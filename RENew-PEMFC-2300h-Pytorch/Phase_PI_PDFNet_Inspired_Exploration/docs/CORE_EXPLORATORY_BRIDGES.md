# Core exploratory bridges

The following choices are indispensable to make the published equations into a
closed implementation. They are *not* claimed to be original PDFNet design:

1. `patch → GPRU`: patch convolution, right-pad to `D`, `W_V`, then `E @ value`.
2. Base graph: use attention `E` as `A` rather than adding static node embeddings or a correlation graph.
3. Bidirectional propagation: evaluate `A` and `Aᵀ` with shared weights and average outputs.
4. Intra-to-inter connection: masked patch mean plus final patch hidden state feeds one period-level shared GatedDG.
5. Multi-period fusion: two learned scalar logits only.

Good performance can therefore support only the statement that this whole
preregistered PDFNet-inspired architecture is promising. It cannot separately
validate any one bridge or Official PDFNet.
