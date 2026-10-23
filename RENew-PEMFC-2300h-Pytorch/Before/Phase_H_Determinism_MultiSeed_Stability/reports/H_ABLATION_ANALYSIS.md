# Paired Validation ablation

[
  {
    "seed": 42,
    "D10_F1_minus_F0_mV": 0.9055825044927701,
    "D21_F2_minus_F1_mV": -2.9475740288799344,
    "D20_F2_minus_F0_mV": -2.0419915243871642
  },
  {
    "seed": 52,
    "D10_F1_minus_F0_mV": -2.3760934499978834,
    "D21_F2_minus_F1_mV": -0.9708183135354123,
    "D20_F2_minus_F0_mV": -3.3469117635332957
  },
  {
    "seed": 62,
    "D10_F1_minus_F0_mV": 1.7346153455505426,
    "D21_F2_minus_F1_mV": -1.81220127218387,
    "D20_F2_minus_F0_mV": -0.07758592663332742
  }
]

[
  {
    "comparison": "D10_F1_minus_F0_mV",
    "mean_difference_mV": 0.08803480001514312,
    "sample_std_mV": 2.173883595612303,
    "improved_seed_count": 1,
    "max_degradation_mV": 1.7346153455505426
  },
  {
    "comparison": "D21_F2_minus_F1_mV",
    "mean_difference_mV": -1.9101978715330723,
    "sample_std_mV": 0.9920147628184847,
    "improved_seed_count": 3,
    "max_degradation_mV": 0
  },
  {
    "comparison": "D20_F2_minus_F0_mV",
    "mean_difference_mV": -1.8221630715179291,
    "sample_std_mV": 1.645711477894464,
    "improved_seed_count": 3,
    "max_degradation_mV": 0
  }
]

F2_VALIDATION_MULTISEED_GAIN_OBSERVED

Differences are F1−F0, F2−F1 and F2−F0; a negative value means F1/F2 (the minuend) is better. F2 gains relative to F1 alone do not establish superiority to F0. No model or gate redesign.
