PHASE_I_EXECUTION_BLOCKED

Traceback (most recent call last):
  File "E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_I_Frozen_Checkpoint_Assessment\scripts\run_i_assessment.py", line 28, in execute
    if not np.array_equal(val,saved):raise RuntimeError('FROZEN_VALIDATION_RELOAD_MISMATCH:'+spec['run_id'])
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: FROZEN_VALIDATION_RELOAD_MISMATCH:I_F0_Seed42
