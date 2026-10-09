# Training and error diagnosis

Each fold first used the final chronological 15% of its outer-train origins as an inner stopper. The selected epoch then governed a same-seed reinitialised refit on the entire outer train; outer validation was not used for epoch selection. The maximum available epochs were 300 with patience 30, AdamW (0.001, weight decay 1e-5), batch size 64, and gradient clipping 1.0.

Stage 0 verified the actual PTF architecture, a finite GPU forward/backward pass, gradient finiteness, extractable VSN/attention tensors, and prediction invariance to unavailable future measurements. The completed model has 86,883 trainable parameters. The formal overall RMSE is 26.158 mV.
