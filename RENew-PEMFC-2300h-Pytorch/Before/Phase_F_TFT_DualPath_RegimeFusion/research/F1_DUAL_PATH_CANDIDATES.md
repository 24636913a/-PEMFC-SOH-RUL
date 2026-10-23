# F1 Dual-Path Candidates and Decision

## Comparison criteria

The comparison is architectural only. It uses no Assessment scores, no Test data and no additional models. The desired branch must complement the post-VSN LSTM representation, remain causal within the fixed 48 historical bins, preserve the native decoder state handoff and create a simple F0→F1 ablation.

| Candidate | Two paths | Expected value for dynamic PEMFC voltage | Main risks | Causal / 48-step fit | VSN/GRN/attention overlap | Decision |
|---|---|---|---|---|---|---|
| A | Native LSTM + two-scale depthwise causal convolution | LSTM summarizes ordered context; convolutions expose short (3-bin) and medium (9-bin) load/voltage transients directly | Extra capacity may overfit; fixed fusion may dilute a dominant path | Yes; left padding only, no sequence extension | Low: VSN selects variables; conv encodes local temporal motifs; attention stays unchanged | **Recommended** |
| B | Fine-scale causal convolution + pooled/coarse temporal branch | Explicitly separates 0.5 h detail from a wider scale | Downsampling/alignment choices are additional design degrees; pooling can erase startup/load steps in only 48 bins | Possible but less clean; interpolation/alignment needed | Medium; coarse branch partly overlaps attention/LSTM global handling | Not selected |
| C | Native LSTM + handcrafted dynamic-change branch (differences, local amplitudes) | May highlight transitions and voltage changes | Introduces feature engineering, duplicates existing current/transition inputs, and blurs whether gains arise from representation or manual features | Causal, but weakly justified as a distinct encoder | High overlap with Raw F2, VSN and load-transition variable | Not selected |

## Why candidate A is the narrowest informative contrast

The existing TFT already combines recurrent local processing and global attention. It does not, however, contain a causal local-convolution path applied to the VSN-selected historical representation. Candidate A changes only that missing inductive bias. It uses no downsampling, no precomputed decomposition, no additional source feature and no future covariate. The LSTM still owns the decoder initialization, so the direct multi-horizon contract stays intact.

The two convolution branches use kernel 3 at dilations 1 and 4. Their receptive fields are 3 and 9 half-hour bins (1.5 h and 4.5 h). They are deliberately not offered as a claim that these are physical PEMFC constants; rather they make the minimal local/mid-local contrast available inside the already-frozen 24 h history. Attention and the LSTM remain responsible for broader history.

The causal-convolution motivation is established sequence-modeling practice, not a novel physics law: Bai, Kolter and Koltun (2018) describe causal dilated convolution as a way to extend temporal receptive field; see [arXiv:1803.01271](https://arxiv.org/abs/1803.01271). The project-specific contribution would only be supported after the preregistered F0–F2 ablation, not by this design choice itself.
