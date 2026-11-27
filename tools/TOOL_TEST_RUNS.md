# provenance_report.py — test runs (cloud, 24 Sep 2026, 3 seeds)

Data: the six Kaggle CSVs (phishguard_dataset_builder). These are tests of the tool. **Use your own uncapped runs on the Mac for anything in the paper.**

| Run | Cap | Key numbers | Warnings raised |
|---|---|---|---|
| Nazario phishing + SA ham, transfer vs CEAS ham | none | supervised F1 99.73±0.23; transfer F1 97.61±0.58 (null 42.13); year rule bal. acc. 100.00 | single-class sources; SEVERE transfer; SEVERE date |
| same | 3000 | supervised 99.36; transfer 98.02; year 100.00 | same |
| Six-corpus compilation, `--target Nazario` | 3000 | supervised 98.65; source-majority oracle 65.32; label-free 66.10; corpus-ID 94.26; separability class 1 87.83 (chance 16.67; null 49.62), class 0 95.15 (chance 25; null 50.88); leave-Nazario-out 74.57% flagged; Nazario exposure 71.85 → 98.59 (k=0 → 500) | single-class sources; provenance does NOT explain the overall score; sources separable; leave-out rows |
| CEAS alone | 3000 | supervised 99.39; oracle 66.67; year rule 50.17 | provenance does not explain the score |

## Why the capped compilation numbers differ from compilation_audit / followups
The cap keeps at most 3,000 emails per source × class, which changes the class and source mix. That explains why leave-Nazario-out is 74.57% here but 60.32% in the uncapped audit, the Nazario k=0 point is 71.85±18.98 here but 61.13±2.42 in Part G, and the year rule is 70.88 here but 56.73 in H1. The qualitative verdicts are the same. This is why the default is now **no cap**.

## Bugs found and fixed during testing
1. Purity and the source-majority labels came from the capped sample (CEAS showed 50.0%). They now use the full counts (CEAS 55.8%).
2. The datasheet showed min–max years, which one mis-dated email distorts (SpamAssassin ham "2002–2028"). It now shows the 1st–99th percentile, with min–max in brackets.
3. When every source is single-class, the "label-free" row is the supervised model under another name. The report now says so and points to `--transfer`.
4. A single source gave a wrong leave-one-out message. It now says "only one source".
5. The date F1 was unstable when the year has one value (±92). The report now shows balanced accuracy only; F1 stays in the JSON.
6. The default cap was 3000; it is now none.

## Run time (2-core cloud machine)
- Benchmark pair, uncapped: 1 min 48 s.
- CEAS capped: about 2 min.
- Compilation capped: about 40 min.

Expect the uncapped compilation to take longer; on your Mac, add `--seeds 3` and let it run.
