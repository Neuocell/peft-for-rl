# PPT method trace audit (2026-10-09)

Audit scope: `/home/wangls/peft-for-rl` at commit `f6bad923dcdc01f851aeb2f3b8c8df7d2e3fd247` on branch `experiments/spar-lora-v0`. This is a read-only evidence audit. `/root/peft-for-rl`, `/data/peft-for-rl-runtime`, and `/data/peft-for-rl-runtime/outputs/full_bench_eval` were not accessible on this machine. A missing object is therefore reported as `UNKNOWN` or `PARTIAL`, never as proof that it did not exist.

Confidence policy: `CONFIRMED` requires at least two evidence classes (normally implementation/config plus artifact/checkpoint/evaluation); `PARTIAL` means that the mapping is supported but at least one identity-critical object is unavailable; `CONTRADICTED` means the requested/PPT interpretation conflicts with runtime evidence; `UNKNOWN` means no unique evidence-backed mapping was found.

## 1. Executive findings

1. The PPT must separate seven modules: gradient observation, token scope/statistical unit, candidate construction, atom score/order, rank allocation, LoRA materialization, and training/evaluation identity. Labels such as “full-gradient,” “Gain-LCB,” and “adaptive rank” currently collapse different modules.
2. `0.358565@50` is the strongest locally reconstructable older chain: all 7,248 full-bench records are present and recompute to the stated six-task Macro. It maps to full-weight GRPO prompt-group gradients, mean-SVD candidates, Gain-LCB ordering, and uniform r8. The referenced checkpoint and original candidate tensors are under inaccessible `/root`, so the end-to-end chain remains `PARTIAL`, not `CONFIRMED`.
3. `0.340163@100` is not proven to be optimizer-continuous with step50 or to reuse the exact same candidate artifact. Historical summaries support the number, but no exact step50-to-100 launcher/checkpoint identity was found. It must not be shown as a guaranteed continuation curve.
4. `0.375749@150` maps to the same mean/Gain-LCB method family with uniform r32, alpha64, dropout0, seed42, and start-from-scratch training. The checkpoint/evaluation lives under inaccessible `/data`; this is `PARTIAL`. It is not a clean rank-only comparison with r8 because checkpoint step differs and there is no matching r32 step50/100 full-bench.
5. The stable-energy uniform/adaptive comparison is training-only. It reuses the full-gradient mean candidate family but recomputes `P*R`, ignoring artifact `U=max(Gain-LCB,0)` for selection. Its `Reward AUC1:50` and capture numbers are not full-bench. No formal stable-energy uniform-r8 full-bench was found, and `0.358565` must not be attached to this row.
6. Centered versus uncentered covariance is not a causal centering ablation: calibration masking differs, and centered step50-to-100 warm-starts model/global-step/data without optimizer state while uncentered is one optimizer-continuous run.
7. Phase-1 is the strongest artifact-identity experiment. I0/I8/I16/I32 use one frozen Phase-0.5 P2 ordered basis protocol, identical rank32/alpha64/dropout0/B0/trainability, and one deterministic Gaussian generator protocol. However, final random complement rows are not bitwise identical because residualization depends on the retained signal prefix.
8. Random-B uniform-r8 and original near-r32 came from separate probe runs. Width, capacity, bins, observation count, and seed match, but rollout/probe draws and artifact hashes differ; r8 is not proven to be a prefix of the near-r32 candidate.
9. Historical RLPO/CHR/GradTop/Compact/Chord/Dominant values are numerically documented, but inventory grade D means their request/model-snapshot identity differs from the frozen Phase-1 protocol. They are descriptive historical results, not protocol-matched leaderboard entries. The RLPO Macro is additionally unsafe because its six-task source used a corrupted HMMT input; a repaired HMMT exists only as a separate one-task evaluation.

## 2. Result-to-run mapping

`Macro` below always means six-task full-bench Avg@k. Capture and Reward AUC are explicitly labeled and are not Macros.

| Method/result | Run and checkpoint | Evaluation/artifact | Runtime, seed, continuity | Confidence |
|---|---|---|---|---|
| Full-gradient mean/Gain-LCB uniform-r8, Macro `0.358565@50` | `full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_50_seed42`; `/root/peft-for-rl/runs/full-gradient-v1/ckpts/verl/DAPO-Math-17k/.../global_step_50/actor/peft_adapter` | Imported records: `runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42`; 7,248 records recompute exactly | Qwen-1.5B family, 4 GPU launcher, seed42, resume disabled; checkpoint/candidate inaccessible | PARTIAL |
| Same label, Macro `0.340163@100` | Exact training run/checkpoint lineage `UNKNOWN`; a later script stages a step100 adapter with `RESUME_MODE=disable` | Historical summary only; exact formal eval identity not locally complete | Optimizer/scheduler/RNG continuity `UNKNOWN` | PARTIAL |
| Full-gradient mean/Gain-LCB uniform-r32, Macro `0.375749@150` | `full_gradient_mean_gain_lcb_uniform_r32_b64m16n8_270_seed42_v1`; `/data/.../global_step_150/actor/peft_adapter` | `/data/.../full_gradient_mean_gain_lcb_uniform_r32_b64m16n8_step150_fullbench_32768_seed42_v1` | seed42, dropout0, rank32/alpha64, from scratch, save25; `/data` inaccessible | PARTIAL |
| Stable-energy uniform-r8, 9,232,384 params, calib `0.843433`, audit `0.838146`, Reward AUC `0.346875` | `full_gradient_stable_energy_uniform_r8...` training comparison; exact full-bench checkpoint `UNKNOWN` | Allocation SHA `abd503...`; comparison JSON SHA `7149c0...`; audit value only in audit docs because score tensors inaccessible | 50 steps, B64/M16/N8, dropout0, seed42, resume disabled | PARTIAL |
| Stable-energy adaptive-eqr8, 9,231,360 params, rank `2/14.265/32`, calib `0.848773`, audit `0.838414`, Reward AUC `0.3571875` | `full_gradient_stable_energy_adaptive_eqr8...`; exact full-bench checkpoint `UNKNOWN` | Allocation SHA `44f5ef...`; residual budget 1,024 params; no formal full-bench found | Same 50-step training contract | PARTIAL |
| SPAR positive-probe adaptive eq-r8, Macro `0.348220@50` | Historical SPAR formal run; exact local checkpoint path `UNKNOWN` | Candidate/score/allocation tensors and hashes `UNKNOWN` locally | seed/config family supported by launchers; exact identity unavailable | PARTIAL |
| Windowed consensus hybrid uniform-r8, Macro `0.348194@50` | Historical formal run; exact checkpoint unavailable | Probe SHA `d85067...`; allocation SHA `d2a599...`; rank-map SHA `ab0bfd...`; candidate tensors missing | seed42; exact resume/checkpoint evidence unavailable | PARTIAL |
| Random-B energy uniform-r8, Macro `0.358323@50` | Historical uniform-r8 run; checkpoint/eval under unavailable runtime | Source probe `runs/random-b-rank8-v1/analysis/random_b_energy_probe_b16n8_w8_s6to12_seed42_v2`; allocation SHA `6b4492...` | rank8/alpha16, dropout0.05, 50 steps, seed42, resume disabled | PARTIAL |
| Original random-B near-r32, Macro `0.367304@50` | `gradtop_probe12_r8to32_original_random_b_a2_b64m16n8_270_seed42_repro_v1/global_step_50` | `runs/random-b-original-repro-v1/outputs/full_bench_eval/gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1`; summary SHA `767e13...` | separate probe artifact, seed42; summary present, adapter historical `/root` | PARTIAL |
| Uncentered covariance adaptive eq-r28, Macro `0.350770@50` / `0.369779@100` | One `RESUME_MODE=auto` 270-step run; global steps 50/100 | Formal `/data` full-bench directories, inaccessible | rank16..32, eq-r28 budget, dropout0.05; optimizer-continuous | PARTIAL |
| Centered covariance adaptive eq-r28, Macro `0.375451@50` / `0.382131@100` | step50 run plus staged step50-to-100 run | Formal `/data` full-bench directories, inaccessible | rank16..32, dropout0.05; step100 restores model/global-step/data only, fresh optimizer/scheduler/RNG | PARTIAL |
| Phase-0 P0/P1/P2/P3 diagnostics | `runs/phase0-gradient-diagnostics-v1/analysis/phase0_shared_rollout_d64c32a16_r32_seed42_v1` | Shared diagnostic artifact, SHA `331fcf...`; no Macro claim | 64/32/16 prompt groups, shared rollout, audit excluded from selection | CONFIRMED |
| Phase-1 I0 seed42, Macro `0.330162`; seed43 `0.327747` | `phase1_p2_i0.../global_step_50`; seed43 local adapter hash `003799...` | Eval summary SHAs `442e93...`, `f2942b...` | rank32/alpha64/dropout0, seed-specific rollout/data, resume disabled | CONFIRMED |
| Phase-1 I8 seed42, Macro `0.366161`; seed43 `0.341696` | `phase1_p2_i8.../global_step_50`; seed43 local adapter hash `0f180e...` | Eval summary SHAs `213cfd...`, `0fff1d...`; seed43 formal result is v4, not incomplete v3 | Same training contract | CONFIRMED |
| Phase-1 I16 seed42, Macro `0.313498` | `phase1_p2_i16.../global_step_50` | Eval summary SHA `f9db8e...` | Same training contract; seed43 formal result not found | PARTIAL |
| Phase-1 I32 seed42, Macro `0.376479`; seed43 `0.333201` | `phase1_p2_i32.../global_step_50`; seed43 local adapter hash `2fbfdf...` | Eval summary SHAs `136599...`, `db4b81...` | Same training contract | CONFIRMED |
| Stable-SNR diagnostic/checkpoint | `stable_snr_v3_trainableA_pbudget16.../global_step_50` | Probe SHA `8360cd...`; adapter SHA `a23531...`; no formal full-bench found | rank `8/16/32`, dropout0.05, alpha/r2 | CONFIRMED as diagnostic/checkpoint; no Macro |
| Dominant-atoms, Macro `0.373654@50` | dominant-atoms equivalent-r8 run, global step50 | Summary SHA `7db12a...`, records SHA `03fe4e...`; adapter SHA `b49b81...` | rank `4/10.673/16`, dropout0.05, alpha/r2 | PARTIAL: historical grade D |
| CHR/D1, Macro `0.380842@200`, `0.393234@270` | `chr_rlpo_v0_d1act95_r8to28_mean16p78_a2_b64m16n8_270_4gpu_boxed_v3`, steps 200/270 | Eval summary SHAs `757e8b...`, `ecefd8...`; records SHAs `2745bf...`, `9a5217...` | activation-prefix rank map from RLPO step270; historical grade D | PARTIAL |
| Compact, Macro `0.397868@150` | `gradtop_step100_act90_compact_r8to28_mean18p29_a2_b64m16n8_rem170_v2/global_step_150` | Summary SHA `b4e42d...`; compact adapter SHA `32f0b2...` | warm-started from compressed GradTop step100; optimizer continuity not established here | PARTIAL |
| Chord lambda1.25, Macro `0.403345` | materialized adapter `runs/initializers/gradtop_step100_270_chord_lambda1p25` | Summary SHA `f2b031...`; adapter SHA `802a35...` | inference-time checkpoint chord; no optimizer state | PARTIAL |
| RLPO-init r32, historical Macro `0.378684@270` | `rlpo_init_r32a64_b64m16n8_270_4gpu_boxed_v2/global_step_270`; adapter SHA `1a9a78...` | Summary SHA `1c48ca...`; repaired HMMT is separate one-task eval SHA `2b2834...` | r32/alpha64/dropout0.05, base-weight top-r SVD init | CONTRADICTED as a clean six-task Macro because HMMT source was corrupted |
| GradTop, Macros `0.391198/0.403326/0.404574/0.401638/0.399401/0.399411` at steps `50/100/150/200/250/270` | `gradtop_probe12_r8to32_mean31p63_a2_b64m16n8_270_v1/global_step_{step}` | Local full-bench summaries and records; summary SHAs listed in section 4 | optimizer-continuous trajectory; original probe tensors may be absent; historical grade D | PARTIAL |

## 3. Complete method module table

### 3.1 Observation and candidate modules

| Method family | Gradient source | Statistical unit | Token scope | Group advantage and reduction | Gradient clipping | Observation/sketch | Data split | Candidate constructor/rank | Centering |
|---|---|---|---|---|---|---|---|---|---|
| Full-gradient mean/Gain-LCB and stable-energy | Dense frozen-base linear-weight GRPO policy gradient | Prompt group; response tokens averaged inside group | Response tokens | Full rollout-group advantages are fixed before capture; token mean then prompt-group observation | Whole-model gradient norm, threshold discovery median ×2.5; one observed scale `0.993655...` below 1 | No random-B; dense `G`; covariance uses deterministic Nyström `Omega` | 16/16/8 discovery/calibration/audit captured prompt observations, mutually exclusive; artifact also records 80 generated prompts/640 rollouts | Mean: right singular vectors of mean `G`; covariance: Nyström right basis; hybrid combines bases; rank32, 196 all-linear modules | Uncentered `E[G^T G]Omega`; centered subtracts `E[G]^T(E[G]Omega)` |
| SPAR | Teacher-forced CE on selected positive responses; LoRA random-B factor probe represents `G^T B` | One selected response per unique prompt | Response-only | No GRPO group reduction in probe; CE token mean per response | Global projected-gradient sample norm, discovery median ×2.5 | rank32: A=0, deterministic Gaussian B/sqrt(r), installed once at observation0; read `grad_A.T/scaling=G^T B` | 32/32 selected prompts by default; reward>=0.5, parse success, prompt unique | Incremental factor/SVD, candidate A rank32; B reset to zero | N/A |
| Windowed consensus | Dense full-weight GRPO gradient with unbiased single-response estimator | One temporal window of 16 prompts | Response tokens of one deterministic hash-selected response per prompt | Signed GRPO contribution, inverse-probability factor `response_count/window_prompts` | Whole-window norm, discovery median ×2.5 | No random-B; offline CPU virtual Adam, beta1=.9 beta2=.999 eps=1e-8, bias corrected | 8/2/2 windows; 512 prompts/4096 rollouts generated | Four right-singular atoms/window from raw momentum and virtual-Adam update; hybrid pools locally normalized weighted atoms then SVD | N/A |
| Random-B / GradTop | GRPO policy gradient projected through LoRA factors; no factor optimizer updates during probe | Actor optimizer-update/mini-batch observation | Response-only training loss | Normal actor GRPO reduction | No additional stable whole-model clipping established for historical v1 | A=0, B Gaussian/sqrt(r); `grad_A.T/scaling=G^T B`; B refreshed after every captured optimizer-update observation | 12 observations, seed42; no independent discovery/calibration/audit in v1 | Accumulate/sketch then SVD; width8, capacity64, energy target.95, bins 8..32 | N/A |
| Uncentered covariance eq-r28 | Dense full-weight prompt-group GRPO gradient | Prompt group | Discovery top50% surprisal, min128 and force final128; calibration also masked; audit unmasked | Full group advantage fixed before mask; prompt-group reduction | Full-gradient collector clipping contract | No random-B; Nyström width40 | 64/32/16 | Uncentered covariance right basis rank32 | `E[G^T G]Omega` |
| Centered covariance eq-r28 | Same | Prompt group | Discovery mask only; calibration/audit unmasked | Same | Same | No random-B; width40 | 64/32/16 | Centered covariance right basis rank32 | `E[G^T G]Omega-E[G]^T(E[G]Omega)` |
| Phase-0 P0 | Dense full-weight GRPO gradient | Prompt group | No mask | Full 8-response group advantage fixed before selection | Full-gradient collector clipping | No random-B, width40 | 64/32/16, shared rollouts, disjoint | Centered covariance rank32 | Centered |
| Phase-0 P1 | Same | Prompt group | Deterministic density-matched random mask; same requested count as P2 | Same | Same | Same | Same; calibration/audit unmasked | Centered covariance rank32 | Centered |
| Phase-0 P2 | Same | Prompt group | Top50% surprisal, min128, force final128 in discovery | Same | Same | Same | Same; calibration/audit unmasked | Centered covariance rank32 | Centered |
| Phase-0 P3 | Same | Prompt group then deterministic response-half assignment | Stable band: excludes top5% surprisal tail when possible, ranks by `abs(advantage)*entropy` | Advantage tensor is computed on all 8 responses before cross-fit split | Same | Symmetric centered cross-fit | Same; audit not used for selection | Cross-fit centered covariance rank32 | Centered per half |
| Phase-1 I0/I8/I16/I32 | No new gradient observation; consumes frozen P2 basis | N/A | N/A | N/A | N/A | Deterministic Gaussian generator per module for complements | Training seed changes rollouts/data only, not allocation | 0/8/16/32 P2 signal atoms plus 32/24/16/0 orthogonal random atoms | N/A |
| Stable-SNR | Random-B projected GRPO observations | Window of observations | Response-only | Actor GRPO reduction | Probe-specific window normalization | 5 discovery windows×3 observations; 2 calibration+2 validation; 27 total | Window split as stated | Consensus/stable-SNR candidate; heterogeneous r8..32 | N/A |
| Dominant-atoms | Random-B projected GRPO observations | Window | Response-only | Actor GRPO reduction | Probe-specific | 4 discovery windows×2; 1 calibration+1 validation; local top2 atoms | 12 observations total | Dominant local atoms, heterogeneous candidate r4..16 | N/A |
| RLPO-init / CHR | No gradient probe | N/A | CHR rank diagnostic uses response hidden states | N/A | N/A | No sketch | CHR uses 24 calibration +24 validation sequences sampled from RLPO eval records | RLPO A is frozen base-weight top-r right singular subspace; CHR only changes r by activation-output energy | N/A |
| Compact | No pretraining gradient probe; post-hoc factor `Delta W=BA` | Module/checkpoint | Held-out response hidden states for output-energy diagnostic | N/A | N/A | Factor QR+small SVD | Calibration/validation activation records in source diagnostic | Reorders/truncates current `Delta W` singular components to 90% activation-output energy, bins 8..28 | N/A |
| Chord | No gradient probe; combines trained checkpoints | Module/checkpoint | N/A | N/A | N/A | `DeltaW_100 + lambda(DeltaW_270-DeltaW_100)`, projected to anchor span | N/A | Uses historical GradTop anchor subspaces, preserving module rank | N/A |

### 3.2 Score, allocation, materialization, and training modules

| Method family | Atom score/artifact fields | Local ordering | Rank allocation and cost | Rank bounds/budget | LoRA materialization/trainability | Scaling/dropout | Training continuity | Identity/confidence/PPT correction |
|---|---|---|---|---|---|---|---|---|
| Full-gradient mean/Gain-LCB | `F=E||proj G||^2`; `S=||E proj G||^2`; `R=S/(F+eps)`; `P=F/sum(F)`; `gain_lcb=gain-z*SE`; saved `U=max(gain_lcb,0)`, each `(32,)` float32 | Descending saved U | Uniform top-r; per-rank A+B cost `d_in+d_out` | r8: 9,232,384 params; r32: 36,929,536 | Candidate rows copied to A, B=0; A/B trainable | alpha=2r, effective2; dropout0 | From scratch for formal r8-50 and r32 run; step100 continuity UNKNOWN | PARTIAL; say “mean-SVD + Gain-LCB ordering + uniform rank,” not only “full-gradient” |
| Full-gradient stable-energy | Reads F/P/R, recomputes utility `P*R`; saved Gain-LCB U ignored for selection | Selection by descending P*R; writer subsequently orders selected rows by saved U, a materialization-order subtlety | Uniform top-r or global greedy `utility/(d_in+d_out)` after r_min | uniform r8 budget; adaptive r2..32, mean14.265, 9,231,360 params, 1,024 residual | Same candidate-to-A/B0; A/B trainable unless freeze flag | alpha=2r, dropout0 | 50-step from scratch | PARTIAL; training-only, no proven Macro |
| SPAR | `F`, `S`, `R=S/F`, `P=F/sum(F)`, exactly `U=P*R`; code-confirmed, tensors unavailable | Descending U | Global greedy `U/(d_in+d_out)` | r2..32, eq-uniform-r8 budget; exact realized map UNKNOWN locally | Candidate to A, B0, A/B trainable | effective2; exact formal dropout/config UNKNOWN | UNKNOWN | PARTIAL; call it positive-CE random-B candidate, not full-gradient GRPO |
| Windowed consensus | Held-out `P=mean relative energy`; `P_lcb=P-z*SE`; `U=recurrence*max(P_lcb,0)` | Descending U | Uniform top8 | r8, 9,232,384 | Candidate A/B0, A/B trainable | alpha/r2; dropout from exact formal checkpoint UNKNOWN | UNKNOWN | PARTIAL |
| Random-B uniform-r8 | Historical artifact has ordered subspace/rank map but no retained per-atom F/U | First eight saved candidate rows | Uniform r8 | r8, 9,232,384 | A=candidate prefix, B0, A/B trainable | alpha16/r8=2; dropout0.05 | From scratch | PARTIAL; not same candidate as near-r32 |
| Random-B near-r32 | Energy-prefix threshold from each independent probe artifact | Saved energy-prefix order | Per-module energy threshold, bins | source r8/28/32, mean31.612 | Candidate rows to A/B0 | alpha/r2; dropout0.05 | From scratch | PARTIAL |
| Covariance centered/uncentered | Stable-energy `P*R` | Descending P*R | Global greedy utility/cost | r16..32, eq-r28 budget | Candidate A/B0; A/B trainable | alpha/r2; dropout0.05 | Centered warm-start is not optimizer-continuous; uncentered is | PARTIAL; not clean centering ablation |
| Phase-0 P0/P1/P2/P3 | Score artifact stores `F,P,U,audit_F,audit_P,audit_U,spectrum`, all `(32,)` float32; no R field | Saved U/order under diagnostic builder | Diagnostic only | r32 candidates | Not a training checkpoint by itself | N/A | N/A | CONFIRMED diagnostic; do not claim U=P*R where R is absent |
| Phase-1 | Composition count is the controlled variable, not a learned score | Signal prefix preserved exactly; random rows residualized/orthonormalized | Uniform r32 | 36,929,536 params each | A=constructed basis, B0; A/B trainable | alpha64/r32=2, dropout0 | 50 steps, resume disabled | CONFIRMED for available rows; “same generator/draw protocol,” not “identical random rows” |
| Stable-SNR | Window stability/SNR utility; heldout capture mean .107348, validation .117490 | Artifact order | Parameter-budget16 heterogeneous map | r8/16/32 | A=candidate/B0; A/B trainable | alpha/r2; dropout.05 | step50 checkpoint | CONFIRMED diagnostic, no formal full-bench |
| Dominant-atoms | Recurrence/dominance across local top2 atoms | Artifact order | Equivalent-r8 budget | r4/10.673/16 | A=candidate/B0; A/B trainable | alpha/r2; dropout.05 | step50 | PARTIAL due grade D formal eval |
| RLPO-init | Base singular value order, no atom utility artifact | Descending base singular values | Uniform r32 | r32 | A=top-r `Vh`, B0; ordinary two-factor A/B training; no orthogonal loss | alpha64/r32=2; dropout.05 | From scratch | Implementation/checkpoint confirmed; Macro contradicted by HMMT issue |
| CHR/D1 | Activation-weighted output energy of trained RLPO components | Original base singular-value prefix (“activation_prefix”) | Per-module smallest prefix reaching .95, bins | r8/16.776/28, 18,496,512 params | New RLPO base top-r A/B0, not copied trained RLPO `Delta W`; A/B trainable | alpha/r2; training dropout.05 | New 270-step run | PARTIAL historical; call it activation-derived static rank + RLPO init |
| Compact | Held-out activation-output energy of trained GradTop `Delta W` components | Current `Delta W` singular components | Smallest prefix for .90 energy, quantized bins | r8/18.286/28, 22,005,760 params | Re-factorized nonzero A and B then continued training | constant effective2 | Warm-start model; optimizer continuity UNKNOWN | PARTIAL; post-hoc compression, not initialization allocator |
| Chord | No atom score | N/A | No rank reallocation | Preserves source ranks | Materializes projected checkpoint chord; A/B encode nonzero extrapolated update | source effective scaling | Inference-only materialization | PARTIAL; behavior-control/extrapolation point, not adaptive rank |

## 4. Artifact identity and hashes

### 4.1 Full-gradient and stable-energy

| Object | Identity/invariants | SHA256 |
|---|---|---|
| Full-gradient probe summary | 196 modules, candidate r32; 40 captured groups in 16/16/8 split; metadata 80 prompts/640 rollouts; maximum mean-basis row-orthogonality error `1.7881393432617188e-6` | `beda7e7b1166ff201e2f6bb58cc550bf3a13adc4f3874d5157da8620ae1571a2` |
| Mean uniform-r8 allocation / rank map | rank 8/8/8; recomputed A+B params 9,232,384 | `4d36a5f7fc986f9a65849a6470560a9f8f252f8925fe6bc4027aea6b00be7ef5` / `e2b43216724e80f0758ee4352fc5d258bd4b020a8d135f47e2bd7b2c1c61f92e` |
| Mean uniform-r32 allocation / rank map | rank 32/32/32; params 36,929,536 | `b6665e05cdde39d1b7bcb5ccb43800219bb65b14a638e2e1536f4e96fa022464` / `f787b269d484ec6c86b25faa3bac5577425bce24e8a25fe2b8d56f6bf06366e0` |
| Stable-energy uniform / adaptive allocation | Selection uses recomputed P*R; rank-map stats 8/8/8 and 2/14.265306/32 | `abd50377f7164198d35124a22c0c141d5d1e3ce389f4fb3508edbd98112a8f4b` / `44f5ef27dbc0dc8420748ec3f7380ba0236ccdc1cf6958631f436c219a44f15a` |
| Stable-energy rank maps | Recomputed A+B params 9,232,384 and 9,231,360 | `25b92279e38ceb60bbefed9c58b164aabdd8045ac3a812d0b5b975190188aa9c` / `5ad5e16a4083f7e235088e275d718e32f93b946b2bcfdf0573a8d850bc878176` |
| Step50 full-bench records | 7,248 JSONL rows; task Avg@k `0.311458,0.212500,0.652344,0.101042,0.670000,0.204044`; Macro `0.358565` | `3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da` |

Original candidate/score safetensors reference inaccessible `/root` paths. Therefore `U=max(Gain-LCB,0)` is implementation-confirmed but not locally tensor-revalidated; stable-energy audit capture values are document-backed only.

### 4.2 Phase-0 and Phase-1 tensor invariants

The validated Phase-0.5 P2 candidate is `runs/phase0-gradient-diagnostics-v1/analysis/phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3/candidates_P2.safetensors`, SHA `a9e33ccadf14a0e6253ae22919fac86a4a812df6d4db27ee22a26ca852c4b9dc`. It has 196 float32 finite tensors: 28 of shape `(32,8960)` and 168 of shape `(32,1536)`, with maximum row-orthogonality error `1.1920928955078125e-6` and no NaN/Inf. Scores SHA `a2fcd4416d2a740abd252a19cb0dc52f7ae2547e13a995c3a8b9befc6ce70564` contains 1,372 float32 finite vectors (`196×7`): `F,P,U,audit_F,audit_P,audit_U,spectrum`, each `(32,)`; there is no R field.

| Phase-1 method | Allocation | Rank map | Subspace |
|---|---|---|---|
| I0 | `7d793a718f4640e54d6e10ca8201352a4eb2c06f80f61b7192fcc935621ecc88` | `9276de94a217c1fb1297d8531236b4d8a7113bafaa8f9482c31cb7f496c95927` | `30e0a7eba1ff06b6798805c2feeb3eb21501c79ea53a71906889ea7051a2e276` |
| I8 | `0c7ba14adca289fd1d88377d6c91afaab0f1b26914c03b088b0bddf78e55e131` | `2f20e70333a66f979d7630c83823304d9b9c196d65e86e74ce329200a87478f5` | `63842778dad6064beeae7c5de6a497e2bfd58d83ee93c2f5f567592045950135` |
| I16 | `f4648ec41a7148ed822bb094ec16ac344351cf47253596cc05d18f3545c4bafe` | `cd40feb4847364f0c946450757a8cba4af9b27f00011b268ec831eb415705c89` | `9452bbe57076291a4f92b833277d2263b36c3d491091c5d9794680a22115ed35` |
| I32 | `028045d33910be3ace15fc373ce9efadc6f5be58b48431f7bc51a9fd928ddcc8` | `c6c7fbd5e765f2367d745c8380da862b3e6a84e34fb89a59989ef1fb3472182d` | `a9e33ccadf14a0e6253ae22919fac86a4a812df6d4db27ee22a26ca852c4b9dc` |

Local I8 and I32 subspaces both satisfy the same 196-tensor shape/dtype/finite/orthogonality invariant. Preparation JSON SHA is `d60fc87775a5b1a0e1d27f922b6c4f2b100f0e8614faeb261afaf2c1c88dd386`.

### 4.3 Other artifacts

| Family | Artifact facts | SHA256 |
|---|---|---|
| Windowed consensus | Probe / allocation / rank map | `d8506795be016a8ca9837411a8816070d652f05ee17b08c9f307484aa6893200` / `d2a599d6f3107e8cc1f4bdbb05019d238ceda74bb10827b3272cb33c2c5f7857` / `ab0bfd6f6b5ce0bbae50c3d731a802ddb3dd6eb00335172fc877f2790038d4f7` |
| Random-B r8 source | source summary / source rank map / uniform-r8 allocation / output rank map | `23bd74d04217627849d3fbd54ab0e83ebbd6cbf1bf48b51cea6f1d910044ae3a` / `27513a1eff7e2991f514d399dd6cc1988db49265499e3f0de3c470a2c0c4e803` / `6b44925daa31cb7eaaa2bd6405930bf452b3907874f69477840b78937ef30c58` / `54d399bae3b6e082d23c73231b2644bc651b43de965559a74ec44c2b218907fa` |
| Near-r32 reproduction | summary / rank map / eval summary | `863a114ac29c2c595e08fad1ac015d7cea0c42d3ec65bbe47266c5c8b42b5f20` / `b518a283f68947e9c72df4d98898604a65b219f8b82919150fcd8af3cee8f52e` / `767e134f3190a84dfa161b786955ca322b9947fafe2f856e1ee3fef48751f737` |
| Stable-SNR | 196 finite float32 heterogeneous tensors; max orthogonality `1.6093254089355469e-6`; summary/rank map/subspace | `8360cdf678aabf9e19738e9cdbe26c33e77e82991d46323c451cde7b29f3a8e9` / `fc42245671085c87feca63c6f3bbc4db766cef0b719dbb77c722e8652353af8a` / `5c48f83f4a8f9556a57d43aafc29924625ccb1e4f66d6772eb069ec373c35351` |
| Dominant | 196 finite subspaces, max orthogonality `1.1920928955078125e-6`; summary/rank map/subspace/window atoms/sketches | `94e5b3ab43ce6636aaba3670506863f30678d008dc268f38cd31cfb042980d85` / `9dcdee0e7098c87c93fcc473301ebc1ab770f1a3dcb1192f3f0609841f5153fb` / `8ecc50d6d4dd93dcdfad5798bda24dac8624adf18bad8712393db1fe9ea2197c` / `7ac676bbee2c910c6f48cb2756840d2936bfa12d82e708d5adac07c02dc225a9` / `955eea5c9fb671f60ec2272ab6a9b98bde5ddc1d7e5f31258bd23547c3afed20` |
| CHR/D1 | diagnostic summary / activation-prefix rank map; r8/16.7755/28, 18,496,512 params | `5c94b7548a1ec072cf465413c9137c781a9e5ae0eede36a6481c861e1cc82634` / `2be2cb4abc8d723fc9377c96ceb9dde353f82878878c60a2e085fa0e1b332e5b` |
| Compact | build report / rank map / materialized adapter | `0ec436ffa39dc0bca89cbf3407d25b6d16c29a4a373e598076196b22b19dc0a8` / `dfcd4e95cadf30523f34b0d22e4ddd9a23ddb6b54edfa88fc7ed135d1c286b6f` / `32f0b29dd77dc99028c608cb710ca9fceaa03d0f1ad9dca927b2d332d985b02d` |
| Chord lambda1.25 | global projection residual `.00170748`, max module `.00368706`; summary / adapter | `805558dc78b758e4ffd8ffead2f940a2dceb429043d82e21b38735cbd60a1581` / `802a356938f8817516711cc4595ea901ed9f1e92809abceeaad0b09f9e5894e3` |
| RLPO step270 | adapter config / adapter tensor | `64fe010fddb5339d2d28c4344998b33987a8507cf4942603cb9ab362f27547d9` / `1a9a78ebb31bba7d5154d04242e47933bd17a5f3363d925b2901acbb36fb237b` |

## 5. Contradictions with current PPT

| PPT-style claim | Evidence-backed correction |
|---|---|
| “Full-gradient method” identifies the whole method | Full-gradient specifies only observation. Add candidate constructor, atom score, allocator, and rank: e.g. “dense GRPO prompt-group gradient → mean-SVD → Gain-LCB → uniform r8.” |
| Stable-energy uniform-r8 has Macro `0.358565` | Unsupported. `0.358565` is the Gain-LCB uniform-r8 run. Stable-energy has capture/AUC evidence only; its formal full-bench is UNKNOWN. |
| Artifact U is P×R throughout | False. Full-gradient artifact U is clipped Gain-LCB. Stable-energy ignores U for selection and recomputes P×R. SPAR code defines U=P×R, but its tensors are unavailable. |
| Centered beats uncentered in a centering-only ablation | False. Calibration mask scope and step100 optimizer continuity differ. Present as two compound protocols. |
| Random-B r8 versus near-r32 is a rank-only ablation | False. The probes are separate artifacts with different hashes and module maps; only code/config family is shared. |
| Phase-1 methods use identical random complement rows | Too strong. They use the same deterministic Gaussian generator/draw protocol, but residualization against different signal prefixes changes final rows. |
| P1 random mask and P2 top-surprisal differ only in which tokens, with identical density | Supported for this config: both use the same requested keep count; P1 does not reserve final tokens, P2 forces final128. Calibration/audit are unmasked. |
| RLPO-init is orthogonality-constrained training | False. It initializes A with base-weight top-r right singular vectors, zeros B, then trains ordinary A/B LoRA; there is no orthogonal loss or rank allocator. |
| CHR is oracle gradient-rank allocation | False. It is a static activation-output-energy rank map derived from a trained RLPO checkpoint, followed by a fresh base top-r RLPO initialization. |
| Compact is a training-time adaptive-rank method | False. It is post-hoc compression of GradTop step100 `Delta W`, followed by continued training. |
| Historical results form one matched leaderboard | False. Inventory grade D rows differ from the frozen Phase-1 snapshot/protocol and must be labeled descriptive historical results. |

## 6. Unresolved items

1. The exact candidate and score safetensors behind formal full-gradient results are referenced under inaccessible `/root`; tensor-level Gain-LCB U verification is therefore `UNKNOWN` locally.
2. The exact optimizer/scheduler/RNG lineage and candidate identity for `0.340163@100` are `UNKNOWN`. A later warm-start script cannot prove the formal run's history.
3. The SPAR formal candidate, scores, allocation, rank map, checkpoint, and their hashes are `UNKNOWN` locally. Code behavior is auditable; formal artifact identity is not.
4. The exact windowed-consensus candidate tensor and formal step50 checkpoint are unavailable, although probe/allocation/rank-map metadata are present.
5. Stable-energy audit captures `0.838146/0.838414` cannot be tensor-recomputed because the original score tensors are unavailable; the values remain `PARTIAL`.
6. The four centered/uncentered formal checkpoint/eval directories under `/data` are inaccessible, so historical Macros cannot be locally regenerated.
7. I0 and I16 subspace files are not physically local even though immutable preparation manifests/hashes exist. I16 seed43 formal full-bench was not found.
8. Historical GradTop's original main probe tensor identity is not proven identical to the locally reproduced near-r32 probe. Treat the reproduction only as code/protocol evidence.
9. No git commit was embedded in most historical checkpoints/evaluations; repository HEAD is not evidence that those runs used the same commit.
10. Model/device/runtime for inaccessible historical runs is inferred only where launchers/manifests explicitly record it; all other cases remain `UNKNOWN`.

## 7. Minimal PPT replacement text

Use the following wording verbatim where space is limited:

- **Mean/Gain-LCB r8, step50:** “Dense GRPO prompt-group weight gradients; mean-SVD rank-32 candidates; held-out Gain-LCB atom ordering; uniform rank 8. Six-task Macro 0.358565 (7,248 records reverified; source checkpoint/candidate path currently inaccessible).”
- **Mean/Gain-LCB r32, step150:** “Same method family with uniform rank 32, alpha64, dropout0, seed42, start from scratch; Macro 0.375749. Not a matched-step rank ablation.”
- **Stable-energy allocation:** “Same mean-candidate family; atom selection recomputes P×R rather than using Gain-LCB U. Uniform-r8 versus adaptive eq-r8 improves training Reward AUC 0.346875→0.357188 at matched parameter budget; formal full-bench not established.”
- **Centered/uncentered covariance:** “Compound protocol comparison, not centering-only: calibration mask scope and step100 optimizer continuity differ.”
- **Phase-0/1:** “P2 uses discovery-only top-50% surprisal masking (min/final 128), with unmasked held-out scoring. Phase-1 freezes this ordered basis and varies 0/8/16/32 signal atoms within identical rank32 LoRA contracts; complements share a deterministic generator protocol, not identical final rows.”
- **SPAR:** “Positive, parse-valid, prompt-unique teacher-forced CE responses → rank32 random-B sketch → U=P×R → adaptive eq-r8. Formal artifact identity remains incomplete.”
- **Random-B:** “Random-B r8 and near-r32 are separate probe realizations; report as related protocols, not a strict rank ablation.”
- **Historical line:** “RLPO/CHR/GradTop/Dominant/Compact/Chord are descriptive historical results under a non-matched snapshot. Compact is post-hoc compression; Chord is checkpoint extrapolation; neither is a training-time adaptive-rank allocator.”

## 8. Evidence index

### Implementation

- `verl/utils/full_gradient_rl_probe.py:60-167`: token-mask construction; `:256-274`: centered/uncentered covariance action; `:501-571`: deterministic signal/random subspace construction; `:1118-1145`: F/S/R/P/Gain-LCB; `:1783-1835`: cross-fit response split.
- `verl/utils/peft_gradient_subspace.py:1361-1397`: random-B capture and refresh; `:1588-1621`: optional fixed-A behavior; `:1631-1690`: candidate-to-LoRA materialization.
- `verl/utils/peft_spar_lora.py:270-275`: SPAR F/S/R/P/U; surrounding collector code implements positive-response selection and random-B candidate construction.
- `verl/utils/peft_rlpo.py:60-134`: base-weight SVD, A=top-r Vh, B=0.
- `scripts/analysis/build_full_gradient_uniform_allocation.py:86-107`: stable-energy P×R selection and utility/cost allocation.
- `scripts/analysis/build_gradient_probe_uniform_allocation.py:35+`: saved random-B candidate prefix to uniform-r allocation.
- `verl/workers/actor/dp_actor.py:263-304`: SPAR integration; `:497-657`: full-gradient discovery/held-out capture flow.
- `verl/trainer/ppo/ray_trainer.py:1469-1530`: probe lifecycle/training orchestration.

### Configurations and launchers

- `scripts/local/run_token_mask_covariance_adaptive_experiment.sh`
- `scripts/local/run_token_mask_covariance_step50_to100_then_eval.sh`
- `scripts/local/run_token_mask_uncentered_covariance_adaptive_270.sh`
- `scripts/local/run_token_mask_uncentered_step50_eval.sh`
- `scripts/local/run_token_mask_uncentered_step100_eval.sh`
- `scripts/local/run_d1_activation_weighted_rank.sh`
- `examples/verl_train/run_dapo_math_boxed_chr_rlpo_v0_1p5b_4gpu_8k.sh`
- `scripts/local/run_original_random_b_reproduction.sh`

### Canonical local artifacts and evaluations

- `runs/phase0-gradient-diagnostics-v1/analysis/phase0_shared_rollout_d64c32a16_r32_seed42_v1/probe_summary.json`
- `runs/phase0-gradient-diagnostics-v1/analysis/phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3/`
- `runs/phase1-signal-random-v1/analysis/training_allocations/phase1_training_preparation.json`
- `runs/phase1-signal-random-v1/analysis/local_fullbench_evidence_inventory_20261004.json`
- `runs/random-b-rank8-v1/analysis/random_b_energy_probe_b16n8_w8_s6to12_seed42_v2/`
- `runs/random-b-original-repro-v1/analysis/rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_seed42_original_repro_v1/`
- `runs/windowed-adam-consensus-v1/analysis/windowed_adam_consensus_probe_b64n8_seed42_v2/`
- `runs/analysis/rl_gradient_probe_stable_snr_w5x3_c2_v2_pbudget16_v3/`
- `runs/analysis/rl_gradient_probe_dominant_atoms_w4x2_c1v1_pbudget8_v1/`
- `runs/analysis/d1_rlpo_r32_step270_activation_rank/`
- `runs/initializers/gradtop_step100_act90_compact/`
- `runs/initializers/gradtop_step100_270_chord_lambda1p25/`
- `runs/outputs/full_bench_eval/` and `runs/phase1-signal-random-v1/outputs/full_bench_eval/`

### Accessibility boundary

- Accessible: local repository/data/evaluation paths listed above and `/media/nfs_share/wangls/peft-for-rl-external-validity-v1`.
- Inaccessible during audit: `/root/peft-for-rl`, `/data/peft-for-rl-runtime`, `/data/peft-for-rl-runtime/outputs/full_bench_eval`.
