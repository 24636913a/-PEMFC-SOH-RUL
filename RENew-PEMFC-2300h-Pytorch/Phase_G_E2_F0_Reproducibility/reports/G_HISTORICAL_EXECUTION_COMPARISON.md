# Historical E2/F0 execution comparison

{
  "first_divergence": {
    "epoch": 4.0,
    "E2_train_MSE": 1.0782116692174564,
    "E2_validation_RMSE": 0.9963804483413696,
    "F0_train_MSE": 1.0782116759907117,
    "F0_validation_RMSE": 0.9963801503181458,
    "train_MSE_absolute_difference": 6.773255289971303e-09,
    "validation_RMSE_absolute_difference": 2.980232238769531e-07
  },
  "E2_epochs": 327,
  "F0_epochs": 205,
  "E2_best_epoch": 287,
  "F0_best_epoch": 165,
  "first_three_epochs_exact": true,
  "E2_first_batch_evidence": "HISTORICAL_FIRST_BATCH_FINGERPRINT_UNAVAILABLE",
  "source_git_blobs": {
    "RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py": {
      "current_blob": "3a50fed2d7a0de740202f06c63376e616f6acb57"
    },
    "RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/scripts/run_f_training.py": {
      "current_blob": "c449a64471654d5abfb938409b806f8a7c8c0709"
    },
    "RENew-PEMFC-2300h-Pytorch/Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py": {
      "current_blob": "d27e2a57ec91e40fd1f30e7a133622542eb12b5c"
    },
    "RENew-PEMFC-2300h-Pytorch/Phase_TFT_Original_Baseline/models/canonical_tft.py": {
      "current_blob": "dd7587a6ab1bcb29b90f1865f51e620527155a3a"
    }
  }
}

Source stage comparison and available runtime evidence are in the CSV tables. Source order has been reviewed: mathematical forward/loss/backward/clip/step and validation expressions agree. Dataset ID tracking, finite checks, in-memory best clones versus frequent serialization, logs and explicit CPU thread control differ. None is presently a proven historical cause.

E2 historical CPU threads, Python, CUDA/cuDNN, GPU, TF32, and deterministic-algorithm flag were not separately saved. Absence of an explicit setting does not identify its actual value. The E2 first-batch hash is unavailable; reconstructed hashes cannot substitute for historical evidence.

Only existing Assessment caches were compared. No new Assessment windows or inference were run. Final checkpoint differences are expected at best epochs 287 versus 165. Historical files are preserved.
