# Future-condition architecture

Official TimeXer commit `76011909357972bd55a27adba2e1be994d81b327` remains the historical backbone. FC-TimeXer adds a 9→d_model MLP with positions, 24 horizon queries, one MultiheadAttention cross-condition layer, residual LayerNorm/correction head, and learnable alpha initialized at 0.1.

- parameter_count_original: 115096
- parameter_count_future_branch: 20674
- parameter_count_total: 135770
- future branch / original: 17.9624%
