# Paired Full-Benchmark Comparison

- Baseline: `phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42`
- Candidate: `phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42`
- Macro Avg@k delta: `+4.6317%`
- Problem-bootstrap 95% CI: `[+3.0682%, +6.3484%]`
- P(delta > 0): `1.0000`
- Scope: single-seed screening evidence; multi-seed confirmation is still required.

| Benchmark | Baseline | Candidate | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| aime24 | 25.3125% | 30.4167% | +5.1042% | [-0.4167%, +11.6667%] |
| aime25 | 21.2500% | 23.5417% | +2.2917% | [-1.4583%, +7.1875%] |
| amc23 | 58.9844% | 70.4688% | +11.4844% | [+6.3281%, +16.9531%] |
| hmmt_feb | 8.8542% | 10.5208% | +1.6667% | [-0.3125%, +3.9583%] |
| math500 | 63.2000% | 69.8000% | +6.6000% | [+4.7500%, +8.5000%] |
| minerva | 20.4963% | 21.1397% | +0.6434% | [-1.4706%, +2.7574%] |
