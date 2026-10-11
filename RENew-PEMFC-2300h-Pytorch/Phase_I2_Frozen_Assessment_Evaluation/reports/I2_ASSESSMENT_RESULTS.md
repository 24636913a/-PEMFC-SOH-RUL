# I2 Assessment results

Reused Development Assessment; n=3 frozen seeds. Not independent Test. All 9072 queries per run retained. No new training, no new model selection.

Forecast-query-weighted RMSE (mV) is primary; MAE/R2 auxiliary. 81648 predictions total. Independent scalar/math.fsum recomputation versus frozen project c.metric: 72 checks PASS; maximum difference 3.552713678800501e-15.

| run_id | model | seed | H_run_id | RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | max_SSE_target_row | max_SSE_target_age_h | max_SSE_target_multiplicity | max_SSE_target_share | top5_target_SSE_share | independent_target_rows | predictions | origins | csv_sha256 | raw_npz_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I2_F0_Seed42 | F0 | 42 | H-F0-Seed42 | 18.9127603 | 8.03137732 | 0.990649539 | 9.12823434 | 18.9539386 | 28.352991 | 316.389423 | 19.201285 | 2952 | 1793.5 | 19 | 0.394923789 | 0.769940034 | 447 | 9072 | 378 | ed5eff0dd5581627929e8589a1914c2d4f704aac0f6141eff421a739fc2d6e6c | 1348fd3a28f8d52275984b5935823898b61d625591c963a91cbeab805a426ebf |
| I2_F0_Seed52 | F0 | 52 | H-F0-Seed52 | 19.2552378 | 8.50039228 | 0.990307832 | 29.8818975 | 20.4752755 | 39.4939157 | 506.37866 | 19.2597895 | 2952 | 1793.5 | 19 | 0.307472912 | 0.677400508 | 447 | 9072 | 378 | c6dd5c1b39f9783fa5f158ab4fd97bd2212f276db1518c276cef95d457131477 | 414a45f8b11846b22d2294479ad985d9dc4c647cb67e1526ec9ff02c4bc2f3f5 |
| I2_F0_Seed62 | F0 | 62 | H-F0-Seed62 | 27.0684818 | 5.79056932 | 0.980846376 | 16.3442244 | 9.79365464 | 21.272327 | 312.359873 | 26.2708042 | 2779 | 1667 | 24 | 0.25400316 | 0.97130058 | 447 | 9072 | 378 | 515fb03515e88be321311bc4d8a9564dab237856d97e8c43126537e41dffbccb | 649c013db3aba5e1c3f0b0221ae6bfd5661d871216ae01372ba314a6a705d31d |
| I2_F1_Seed42 | F1 | 42 | H-F1-Seed42 | 18.4695218 | 6.15200885 | 0.991082678 | 6.46682981 | 14.9044276 | 23.3334845 | 311.106961 | 18.6161043 | 2953 | 1794 | 20 | 0.390174868 | 0.863722273 | 447 | 9072 | 378 | d81f8c3d1a62211418c3715c95682a6bfd499e46fa9f651b0e3951e6aa2f6348 | 891d8bbc0b636d727c000c0b0a6d0fb4ff655ecba7722ffcd3ff9b9b31e93f93 |
| I2_F1_Seed52 | F1 | 52 | H-F1-Seed52 | 38.1570434 | 9.07682004 | 0.961939665 | 44.9759761 | 16.4603471 | 123.376205 | 536.042709 | 35.7438111 | 2503 | 1529 | 24 | 0.302609684 | 0.832147656 | 447 | 9072 | 378 | decac9ff8d59a363d1cd562028343c87e9bd5786b714a9c8b858d3796469520c | a47ad512cfb55faf7be957083d6d84ced213c4e7935178c3e641d6754f1c8bed |
| I2_F1_Seed62 | F1 | 62 | H-F1-Seed62 | 12.0520698 | 6.66892124 | 0.996202949 | 16.0438659 | 17.8068645 | 28.1674254 | 281.286787 | 11.6198879 | 2780 | 1667.5 | 24 | 0.202697731 | 0.403824207 | 447 | 9072 | 378 | bdac2ef66913e91c6bfc890d4356e9d18ba2fa54bd758b5d20e44082236b42bf | d15de508c80cec4b102d5269f4d474229a6589ac84b8c2ada4ce1016c81dd961 |
| I2_F2_Seed42 | F2 | 42 | H-F2-Seed42 | 17.6615155 | 6.15146927 | 0.991845842 | 7.78761891 | 14.7793761 | 23.3615881 | 312.415384 | 17.9434797 | 2952 | 1793.5 | 19 | 0.451409778 | 0.846976224 | 447 | 9072 | 378 | 090bbeab2f5f098bc87aaff87e93ccda355e0b3e56a4922fd6d6effe16240a35 | 8d7fdfe2c5ea04f85c56a26c081c28ea4fcb624d124f38f0500afe57b20c7c8d |
| I2_F2_Seed52 | F2 | 52 | H-F2-Seed52 | 13.7643155 | 6.21743244 | 0.99504741 | 9.1174145 | 16.3314349 | 27.573671 | 318.240993 | 13.6318998 | 2952 | 1793.5 | 19 | 0.240558357 | 0.613489225 | 447 | 9072 | 378 | f75ddde243ba37038543019dc21ac03bf7aaf3b6bef7ba38e768708c74eddd1e | abf0a012aaffcf86bfbca31efd0348a45fcab3fc8e9ac323cdfc15c03ba8b725 |
| I2_F2_Seed62 | F2 | 62 | H-F2-Seed62 | 5.16532586 | 3.07210955 | 0.999302541 | 4.30255197 | 8.18669703 | 14.3087863 | 237.186823 | 5.31912467 | 2952 | 1793.5 | 19 | 0.239438248 | 0.375371359 | 447 | 9072 | 378 | 21ce126f1915484de3cafcb8d5f59961419096ad05f4d0013430bce8ad471e8b | 6fff6ad921d1ed4bfec96987a97524aa73594fbc868b693b3a917c64d43c1f0a |

## Validation versus Assessment

| model | seed | Validation_RMSE_mV | Assessment_RMSE_mV | absolute_gap_mV | relative_gap_percent |
| --- | --- | --- | --- | --- | --- |
| F0 | 42 | 8.30716072 | 18.9127603 | 10.6055996 | 127.668164 |
| F0 | 52 | 10.3815442 | 19.2552378 | 8.87369361 | 85.4756619 |
| F0 | 62 | 8.51398487 | 27.0684818 | 18.5544969 | 217.929644 |
| F1 | 42 | 9.21274323 | 18.4695218 | 9.25677858 | 100.477983 |
| F1 | 52 | 8.00545073 | 38.1570434 | 30.1515926 | 376.638289 |
| F1 | 62 | 10.2486002 | 12.0520698 | 1.8034696 | 17.5972285 |
| F2 | 42 | 6.2651692 | 17.6615155 | 11.3963463 | 181.900056 |
| F2 | 52 | 7.03463242 | 13.7643155 | 6.72968312 | 95.6650286 |
| F2 | 62 | 8.43639894 | 5.16532586 | -3.27107308 | -38.7733333 |

## Rank reversals

| seed | comparison | Validation_difference_mV | Assessment_difference_mV | rank_reversal |
| --- | --- | --- | --- | --- |
| 42 | F1-F0 | 0.905582504 | -0.443238521 | True |
| 42 | F2-F1 | -2.94757403 | -0.808006329 | False |
| 42 | F2-F0 | -2.04199152 | -1.25124485 | False |
| 52 | F1-F0 | -2.37609345 | 18.9018056 | True |
| 52 | F2-F1 | -0.970818314 | -24.3927278 | False |
| 52 | F2-F0 | -3.34691176 | -5.49092225 | False |
| 62 | F1-F0 | 1.73461535 | -15.016412 | True |
| 62 | F2-F1 | -1.81220127 | -6.88674396 | False |
| 62 | F2-F0 | -0.0775859266 | -21.9031559 | False |

Later-period gaps do not uniquely establish overfitting, distribution shift or a physical mechanism.
