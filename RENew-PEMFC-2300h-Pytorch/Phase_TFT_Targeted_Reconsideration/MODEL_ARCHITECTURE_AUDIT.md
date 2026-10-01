# TFT architecture audit

All candidates retain continuous variable selection, GRNs, causal attention, gating, zero-token strict-history decoder and the absolute-scaled-voltage head. Only the local recurrent encoder-decoder varies. sLSTM and mLSTM use the frozen NX-AI v2.0.4 native state APIs; their encoder state is passed directly to the decoder. The official sLSTM pure-PyTorch `vanilla` backend is used because the host has no CUDA Toolkit for compiling the repository's optional CUDA extension; the GPU remains used for tensor execution.
