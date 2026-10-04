# Paired Full-Benchmark Comparison

- Baseline: `phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42`
- Candidate: `phase1_i16_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42`
- Macro Avg@k delta: `-1.6665%`
- Problem-bootstrap 95% CI: `[-3.2627%, -0.1820%]`
- P(delta > 0): `0.0133`
- Scope: single-seed screening evidence; multi-seed confirmation is still required.

| Benchmark | Baseline | Candidate | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| aime24 | 25.3125% | 20.8333% | -4.4792% | [-11.6667%, +1.5625%] |
| aime25 | 21.2500% | 17.5000% | -3.7500% | [-8.4375%, +0.0000%] |
| amc23 | 58.9844% | 56.5625% | -2.4219% | [-5.7812%, +0.7812%] |
| hmmt_feb | 8.8542% | 9.7917% | +0.9375% | [-0.9375%, +3.1250%] |
| math500 | 63.2000% | 63.6500% | +0.4500% | [-1.3500%, +2.2500%] |
| minerva | 20.4963% | 19.7610% | -0.7353% | [-2.9412%, +1.3787%] |
