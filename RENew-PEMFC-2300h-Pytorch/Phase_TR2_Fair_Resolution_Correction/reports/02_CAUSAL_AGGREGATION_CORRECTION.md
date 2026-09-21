# Causal aggregation correction

Experiment C uses a fixed equal arithmetic mean of three predicted 10-minute q values. Future valid-current sample counts never weight a prediction. Actual raw counts remain valid only for construction of the true q label. The automated test changes future observation counts while holding predictions fixed and verifies identical output.
