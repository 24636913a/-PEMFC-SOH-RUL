# H versus I execution comparison

Only fixed 32 Validation origins are inferred. No Assessment, Test, training or model changes. Exact parity remains np.array_equal; diagnostic tolerances do not change the PASS rule.

| path | construct_load | context | requires_grad | predictor |
| --- | --- | --- | --- | --- |
| A | CUDA construct then strict original state load | torch.no_grad | original | original e.predict -> c.predict |
| B | same as A | torch.inference_mode | original | same one-batch predict_delta arithmetic loop |
| C | same as A | torch.no_grad | False | original e.predict -> c.predict |
| D | CPU construct/load then CUDA transfer | torch.no_grad | original | original e.predict -> c.predict |
| E | CPU construct/load then CUDA transfer | torch.inference_mode | False | unmodified original I.predict_frozen |

H creates on CUDA, strict-loads, and calls e.predict -> c.predict under no_grad with original gradient flags. I creates/loads on CPU, transfers to CUDA, disables flags and calls its unchanged predict_frozen under inference_mode. Each of the 45 paths runs in an independent Python child. No additional diagnostic forward calls are performed.

| H_run_id | model | seed | checkpoint_sha256 | best_epoch | parameter_count | saved_prediction_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| H-F0-Seed42 | F0 | 42 | 82d3cbfa50a41ac381907a06598bb62ff4cffd58a5e39aabda26330271797528 | 277 | 323483 | 12f2e5a5a2f70be6fd0bd36f6290d24a3d644102f218cd0c3409a9a369e5ab18 |
| H-F0-Seed52 | F0 | 52 | 8d9a9450c8d868577c67e2b8c75c7b0c2bbba336eb95816624ad74792a4bab25 | 213 | 323483 | 9f4b1086085aa43f78b4eb3c11d0ec301f6369e79626845d8d98cc908ccb1d10 |
| H-F0-Seed62 | F0 | 62 | 10ab47ca4415465b4d11c14cdeca0bc33cee954038d8d0cd78346595eebb26b1 | 353 | 323483 | f9e879b2dc909f75265f28fe895eac67bdbeac70b9a3df800c73cd3ef32b4b76 |
| H-F1-Seed42 | F1 | 42 | 9eb9ffa33685e0ab00d082c04fbe566944ec8b7d7ded28f434c0de8bd9f6c599 | 204 | 332379 | d3187963a53b3fd3067b23b84ecd81e76129ac01923ec0a716ce1c5e48c11211 |
| H-F1-Seed52 | F1 | 52 | 9eea6aef92052b0e6b641fa99d7554d990a1e35baa8c8db3fa93d37f1e4d2fd3 | 125 | 332379 | d5628f0d7caf149d59aba391869e467824fe10c987d73846235ed4d946b0e568 |
| H-F1-Seed62 | F1 | 62 | 35ee1029482df179c79955d2258d770d2675162c5aec5d1a465456e82743c1c0 | 176 | 332379 | 8f212889e5c36e946295d39ea879211a0f8192df361bd601898f8853ace60594 |
| H-F2-Seed42 | F2 | 42 | 0b28f576e1f5172875203849ad452a40e702040d0c3609b12074a506dbeea974 | 204 | 332436 | 2400d5586be892382f5f0e5765309c1a6b352ea269f4c9dd45630eb40c401b1a |
| H-F2-Seed52 | F2 | 52 | 8100c0199fe43e967a082fb78b6222b9a2ec73d40bedb845098e4fe3b7d3d4ed | 223 | 332436 | 59c355b70d7cd85561b23eafaeefba99d0b90616c1cd4bd1761c64126fca2932 |
| H-F2-Seed62 | F2 | 62 | bd727a697fc99cd6fd3391bd6ae347b3d077ec19941db28b4ad62cd223cd36f5 | 324 | 332436 | cfe3090cf39c1cae74514ca91ef99272a5345a65553dc3086bc977b1268ed4d9 |

Input preprocessing uses the inherited C Validation builder and E EWMA/scaler-transform with persisted scaler, no fit. Full Validation input/label/origin/target-row hashes match H for all nine checkpoints. Only the first 32 are cached and predicted. I.load_protocol is deliberately not called because its broad source hashes include Assessment caches; I protocol/code/ledger identities are independently verified without reading those caches.
