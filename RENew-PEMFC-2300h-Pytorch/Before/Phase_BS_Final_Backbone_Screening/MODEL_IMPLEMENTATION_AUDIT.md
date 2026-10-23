# Model implementation audit

All models use the same externally standardized Raw F2 `[B,48,14]` input and produce only the 24-step voltage channel for the common DeltaV loss. The outer wrapper performs no decomposition and supplies no future covariates.

## Official source fingerprints

| model | repository | expected_commit | observed_commit | license | relative_path | expected_canonical_sha256 | observed_canonical_sha256 | pass |
|---|---|---|---|---|---|---|---|---|
| TimesNet | https://github.com/thuml/Time-Series-Library | 4e938a1767106324dd753b2a44832bf870a0252e | 4e938a1767106324dd753b2a44832bf870a0252e | MIT | models/TimesNet.py | f64c4bed1fd7347090044a0163bd4209c9f9ec1c1b19ceff47842df36b64bba7 | f64c4bed1fd7347090044a0163bd4209c9f9ec1c1b19ceff47842df36b64bba7 | True |
| TimesNet | https://github.com/thuml/Time-Series-Library | 4e938a1767106324dd753b2a44832bf870a0252e | 4e938a1767106324dd753b2a44832bf870a0252e | MIT | layers/Conv_Blocks.py | 16d9f2d9e4fa094dc357901e32beecda9839709bcca02625e6447186933ce4e1 | 16d9f2d9e4fa094dc357901e32beecda9839709bcca02625e6447186933ce4e1 | True |
| SimpleTM | https://github.com/vsingh-group/SimpleTM | 3c77d820837b726afb03c943235ea95bc924243d | 3c77d820837b726afb03c943235ea95bc924243d | NO_LICENSE_FILE_AT_FROZEN_COMMIT | model/SimpleTM.py | 210ff3ff49d2bbf737b3f9cc27611d564b163474a0c999d991115ca9572334af | 210ff3ff49d2bbf737b3f9cc27611d564b163474a0c999d991115ca9572334af | True |
| SimpleTM | https://github.com/vsingh-group/SimpleTM | 3c77d820837b726afb03c943235ea95bc924243d | 3c77d820837b726afb03c943235ea95bc924243d | NO_LICENSE_FILE_AT_FROZEN_COMMIT | layers/SWTAttention_Family.py | 842d5fd4b6ce5bcbf4a732532943759afd5274530e5d3bb757dd65784b25cb17 | 842d5fd4b6ce5bcbf4a732532943759afd5274530e5d3bb757dd65784b25cb17 | True |
| DUET | https://github.com/decisionintelligence/DUET | dcc6e6780a9138731b64b9b5398a94a1d97033f0 | dcc6e6780a9138731b64b9b5398a94a1d97033f0 | MIT | ts_benchmark/baselines/duet/models/duet_model.py | 840690b13fecdd08fdc4a602f3d700ede940731a29ad0fe1de7830b7bb9b22b7 | 840690b13fecdd08fdc4a602f3d700ede940731a29ad0fe1de7830b7bb9b22b7 | True |
| DUET | https://github.com/decisionintelligence/DUET | dcc6e6780a9138731b64b9b5398a94a1d97033f0 | dcc6e6780a9138731b64b9b5398a94a1d97033f0 | MIT | ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py | 46ec8d4986e5579ce93066a205f4986ec2bfbc90a7fc221cd6a29ef0726754e1 | 46ec8d4986e5579ce93066a205f4986ec2bfbc90a7fc221cd6a29ef0726754e1 | True |
| ModernTCN | https://github.com/luodhhh/ModernTCN | 56a9a2c018385cd5acef015378cae7f084d1b11c | 56a9a2c018385cd5acef015378cae7f084d1b11c | MIT | models/ModernTCN.py | daf74722e40d928e5ed6bc871b528a5d5d2c9090f30cafb89110cc88dc60fa30 | daf74722e40d928e5ed6bc871b528a5d5d2c9090f30cafb89110cc88dc60fa30 | True |
| ModernTCN | https://github.com/luodhhh/ModernTCN | 56a9a2c018385cd5acef015378cae7f084d1b11c | 56a9a2c018385cd5acef015378cae7f084d1b11c | MIT | models/ModernTCN_Layer.py | b2291534c6a11ebd80882bed89f7cf8c76c721b8e0495294659f2c5b0fe88cea | b2291534c6a11ebd80882bed89f7cf8c76c721b8e0495294659f2c5b0fe88cea | True |

## Synthetic forward/backward and batch-64 resource preflight

| model | status | device | parameter_count | synthetic_input_shape | voltage_output_shape | finite_output | finite_loss | input_gradient_finite | parameter_gradients_with_values | parameter_gradients_finite | common_mse | official_auxiliary_loss | batch64_step_pass | batch64_step_time_s | peak_gpu_memory_MiB | swt | times_block | duet_cluster | duet_channel | external_feature_decomposition | simpletm_swt_internal_only | moderntcn_use_multi_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TimesNet | PASS | cuda | 149046 | [4,48,14] | [4, 24] | True | True | True | 19 | True | 3.449 | 0.000 | True | 0.044 | 34.211 | 0 | 2 | 0 | 0 | False | False | False |
| SimpleTM | PASS | cuda | 9000 | [4,48,14] | [4, 24] | True | True | True | 26 | True | 3.259 | 0.000 | True | 0.013 | 31.195 | 6 | 0 | 0 | 0 | False | True | False |
| DUET | PASS | cuda | 54761 | [4,48,14] | [4, 24] | True | True | True | 36 | True | 3.035 | 0.021 | True | 0.025 | 28.428 | 0 | 0 | 2 | 2 | False | False | False |
| ModernTCN | PASS | cuda | 421264 | [4,48,14] | [4, 24] | True | True | True | 84 | True | 3.075 | 0.000 | True | 0.055 | 82.807 | 0 | 0 | 0 | 0 | False | False | False |

SimpleTM's SWT calls are internal official `GeomAttentionLayer` operations, not DataLoader features. ModernTCN runs the official `use_multi_scale=false` path; no N2.6R top-down activation adapter is present.
