# PhishCascade

Cost-aware cascaded multi-agent triage for on-premises phishing detection.

---

## The idea in one paragraph

Multi-agent LLM phishing detectors run every agent on every email. That is
fine when inference is a cloud API call billed in fractions of a cent, and
untenable when a SOC must run the model on-premises because it cannot ship
customer mailboxes to a third party. PhishCascade routes instead: a free
deterministic rule engine resolves the confident tail, a small local model
handles the ordinary middle, and the full five-agent panel is spent only on
the residue that neither could settle. The contribution is the router and the
resulting accuracy-versus-compute frontier, not the agents.

## Why on-premises is the right framing

GDPR Article 44 transfer rules, client confidentiality clauses, and FedRAMP
boundaries all commonly forbid sending message bodies to an external API. That
forces local inference, which makes **compute** the binding budget rather than
dollars. Every cost number in this project is reported as wall-clock inference
seconds and generated tokens per 1,000 emails — the quantities an operator has
to provision for.

This is also the gap versus prior work. MultiPhishGuard (CCS '25) and
PhishDebate assume cloud APIs and optimise accuracy alone; neither reports a
compute budget, and neither routes.

---

## Architecture

```
                    ┌──────────────────────────────┐
   email ─────────► │ Stage 0 · PhishGuard rules   │  0 LLM calls
                    │ deterministic, ~0 ms         │
                    └──────┬───────────────┬───────┘
             score ≥ τ_high │               │ score < τ_low
              ┌─────────────┘               └─────────────┐
              ▼                                           ▼
          PHISHING                                     BENIGN
                            │ otherwise
                            ▼
                    ┌──────────────────────────────┐
                    │ Stage 1 · small model triage │  1 LLM call
                    │ qwen2.5:3b, JSON verdict     │
                    └──────┬───────────────┬───────┘
             |p−0.5| ≥ δ/2 │               │ uncertain
                           ▼               ▼
                       verdict     ┌──────────────────────────────┐
                                   │ Stage 2 · 5-agent panel      │  5 LLM calls
                                   │ qwen2.5:14b                  │
                                   │  header_forensics            │
                                   │  url_infrastructure          │
                                   │  social_engineering          │
                                   │  brand_impersonation         │
                                   │  → adjudicator               │
                                   └──────────────────────────────┘
```

Three thresholds define a policy. Two degenerate settings recover the
baselines the paper compares against:

| Setting | Recovers |
|---|---|
| `tau_low=-1, tau_high=∞, delta=1.01` | always-deliberate (prior work) |
| `tau_low=∞, tau_high=-1` | heuristics only (no LLM) |
| `tau_low=-1, tau_high=∞, delta=0.0` | small model only |

---

## Install

```bash
# 1. Ollama
brew install ollama          # or: curl -fsSL https://ollama.com/install.sh | sh
ollama serve                 # leave running in its own terminal

# 2. Models (~10 GB total)
ollama pull qwen2.5:3b
ollama pull qwen2.5:14b      # use qwen2.5:7b if you have <16 GB RAM

# 3. Python
pip install langgraph
```

No API key. Nothing leaves the machine.

Model choice is env-overridable, so the same code runs on bigger hardware:

```bash
export PC_SMALL_MODEL=qwen2.5:7b
export PC_LARGE_MODEL=qwen2.5:32b
```

---

## Run

Data comes from `../phishguard_dataset_builder/` — run `prepare_dataset.py`
there first if `phishguard_dev.csv` doesn't exist.

```bash
# Sanity check the plumbing without touching the GPU (verdicts are nonsense)
python experiments/pareto.py --split dev --backend mock --limit 50

# The main experiment: one oracle pass, then every policy replayed free
python experiments/pareto.py --split dev

# Fix the policy from dev, then measure once on test
python experiments/run_cascade.py --split test --policy cascade
python experiments/run_cascade.py --split test --policy always_deliberate
python experiments/run_cascade.py --split test --policy heuristics_only
```

Runs are resumable — a 3,000-email pass over a 14B model is measured in hours.
Kill it and rerun; it picks up from the checkpoint.

### The oracle trick

Sweeping N policies naively means N full runs. Instead every email visits all
three stages **once**; per-stage verdicts and measured costs are cached, and
any policy is then replayed over that cache for free. This is exact rather
than approximate, because generation is deterministic (`temperature=0`, fixed
seed) — the replayed verdict is what a live run would have produced. The cost
model charges only the stages a policy actually visits.

---

## SIEM output

`phishcascade/siem.py` emits Elastic Common Schema documents that Wazuh
ingests directly, with the routing decision and per-email compute cost
attached:

```json
{"phishcascade": {"verdict": "phishing", "exit_stage": "stage2",
                  "llm_calls": 6, "compute_seconds": 4.12, ...}}
```

```xml
<localfile>
  <log_format>json</log_format>
  <location>/var/ossec/logs/alerts/phishcascade.json</location>
</localfile>
```

Alert levels are mapped so only rule-certain or fully adjudicated phishing
reaches level 12+, the usual paging threshold. A cascade that pages on every
cheap-model guess is worse than no cascade.

---

## Layout

```
phishcascade/
  config.py    model tiers, RoutingPolicy — the object the paper is about
  schema.py    CascadeResult and cost accounting
  llm.py       Ollama + deterministic mock, disk cache, token/latency metering
  agents.py    5 specialist prompts + adjudicator
  stages.py    stage0 / stage1 / stage2 and the routing decisions
  graph.py     LangGraph state machine + sequential reference implementation
  siem.py      ECS / Wazuh alert emission
experiments/
  run_cascade.py   full run over a split, resumable
  pareto.py        oracle pass + policy sweep + frontier
```

`graph.py` implements the cascade twice — as a LangGraph state machine and as
plain sequential code — and an equivalence test asserts they agree. Debugging
a state machine is expensive; a 30-email equivalence check is not.

---

## Status

Working and verified end-to-end on the mock backend, including LangGraph
equivalence and all three exit paths. **No real measurements have been taken
yet** — that requires an Ollama run on your hardware. Nothing in
`docs/PAPER_SKELETON.md` is filled in with estimated numbers, deliberately.

---

## Update: the benchmark audit

Building the reviewer-mandated baselines turned up something that changed the
project. See `docs/PAPER_SKELETON.md` for the full account. In brief:

| View given to the classifier | Standard pairing | Single-source control |
|---|---|---|
| Full email | 99.82% F1 | 98.50% |
| **Date header only** | **99.91%** | **63.43%** |
| Cross-corpus transfer | **31.15%** | — |

The Date header alone classifies the standard Nazario-vs-SpamAssassin pairing
better than the entire email, because the two corpora were collected in
non-overlapping epochs. Sanitizing it away moves the headline by 0.4 points;
the confound is distributed across the collection process, not carried by one
field.

Reproduce:

```bash
pip install scikit-learn
python experiments/make_transfer_set.py           # out-of-distribution set
python experiments/baselines.py --cross-corpus    # 99.8 in / 31.2 out
python experiments/leakage_probe.py               # the audit
python experiments/leakage_probe.py --corpus ceas # the control
python experiments/sanitize.py                    # attempt a fix
python experiments/leakage_probe.py --sanitized   # confirm it doesn't work
```

The cascade is still the system contribution, but it now has to be evaluated
under a corrected protocol rather than against inflated numbers.
