# P2 occurrence and conditional-q reconstruction

**PASS.** The actual source implementation defines `p_raw` as the occurrence-classifier probability, `q_plus` as the conditional Ridge output clipped to [0,1], and `q_hard=q_plus` only where `p_raw>=0.5`; otherwise `q_hard=0`. All 22,656 Validation prediction points reconstruct the saved P2 q exactly (maximum absolute residual 0.0) and reconstruct the frozen future-condition tensor within the prescribed float tolerance.

This confirms the requested separation among occurrence probability, conditional q and hard P2 q. Evidence: `outputs/scenarios/p2_component_reconstruction.csv` and its audit JSON.
