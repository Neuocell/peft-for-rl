# Paired Full-Benchmark Comparison

- Baseline: `phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42`
- Candidate: `phase1_i8_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42`
- Macro Avg@k delta: `+3.5999%`
- Problem-bootstrap 95% CI: `[+2.0847%, +5.1859%]`
- P(delta > 0): `1.0000`
- Scope: single-seed screening evidence; multi-seed confirmation is still required.

| Benchmark | Baseline | Candidate | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| aime24 | 25.3125% | 27.8125% | +2.5000% | [-3.0208%, +8.1250%] |
| aime25 | 21.2500% | 22.5000% | +1.2500% | [-1.8750%, +5.8333%] |
| amc23 | 58.9844% | 69.9219% | +10.9375% | [+6.3262%, +15.8594%] |
| hmmt_feb | 8.8542% | 9.7917% | +0.9375% | [-1.3542%, +3.8542%] |
| math500 | 63.2000% | 69.4500% | +6.2500% | [+4.3500%, +8.2000%] |
| minerva | 20.4963% | 20.2206% | -0.2757% | [-2.3897%, +1.8382%] |
