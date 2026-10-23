"""Predeclared finite budget. No optimization over holdout values."""
INITIAL_SEEDS=(42,202,212)
FINAL_SEEDS=(42,103,193,202,210,212)
BASE_LR=0.0009605449687795866
WEIGHT_DECAY=6.95942232671908e-6
MAX_EPOCHS=450
PATIENCE=40
BATCH_SIZE=32
CONFIGS_A={
    'A0':dict(fusion='native',learning_rate=BASE_LR,dropout=.1,loss='MSE'),
    'A1':dict(fusion='native',learning_rate=.0006,dropout=.1,loss='MSE'),
    'A2':dict(fusion='native',learning_rate=BASE_LR,dropout=.05,loss='MSE'),
}
QUALIFICATION=dict(mean_Validation_RMSE_ratio_max=.95,paired_RMSE_wins_required=3,
    mean_Validation_MAE_ratio_max=1.,mean_Validation_P99_ratio_max=1.05,
    worst_seed_MAE_ratio_max=1.10,worst_seed_P99_ratio_max=1.10,
    require_finite_gradients=True,extra_repeats_to_rescue_candidate=0)
STAGE_B_GATE=dict(require_A_complete=True,all_A0_CNN_grad_nonzero=True,
    all_A0_LSTM_grad_nonzero=True,median_A0_g_max=.95,
    mean_A0_fused_to_lstm_RMS_max=.98,
    # Attenuation is a hypothesis, not proof that the CNN caused forecast error.
    CNN_off_Validation_RMSE_worse_seeds_required=2)
STAGE_C_GATE=dict(require_A_B_complete=True,qualified_preC_candidate_required=True,
    min_Validation_unique_event_targets=5,event_weight=2.,
    event_RMSE_ratio_max=.95,non_event_RMSE_ratio_max=1.02,P99_ratio_max=1.)
SELECTION_RULE='Lowest three-seed Validation mean RMSE among qualified candidates; tie <=1e-6 mV: earlier/simpler stage, then candidate ID.'
MAX_FRESH_TRAININGS=27  # A9 + B6 + C6 + D6; conditional stages can reduce this.
