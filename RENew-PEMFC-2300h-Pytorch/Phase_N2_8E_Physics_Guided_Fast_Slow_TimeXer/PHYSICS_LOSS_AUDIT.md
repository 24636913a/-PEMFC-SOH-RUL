# Physics-loss audit

Losses are evaluated in standardized DeltaV target space. The first training batch may rescale only a raw regularizer whose magnitude differs from data MSE by more than 100×; the detached normalizers used are listed below. The fixed relative multipliers remain smooth=0.01, current=0.001, mono=0.001. A fold would fail if the sum of weighted physics losses exceeded data loss for three consecutive epochs. All completed runs are balanced.

| model | fold | seed | physics_mode | smooth_normalizer | current_normalizer | mono_normalizer |
|---|---|---|---|---|---|---|
| SA_TimeXer_Zero | fold_1 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Zero | fold_2 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Zero | fold_3 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Ridge | fold_1 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Ridge | fold_2 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Ridge | fold_3 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Oracle | fold_1 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Oracle | fold_2 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Oracle | fold_3 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Shuffled | fold_1 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Shuffled | fold_2 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| SA_TimeXer_Shuffled | fold_3 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| FS_TimeXer_NoPhysics | fold_1 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| FS_TimeXer_NoPhysics | fold_2 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| FS_TimeXer_NoPhysics | fold_3 | 42 | none | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_Smooth | fold_1 | 42 | smooth | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_Smooth | fold_2 | 42 | smooth | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_Smooth | fold_3 | 42 | smooth | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_SmoothCurrent | fold_1 | 42 | smooth_current | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_SmoothCurrent | fold_2 | 42 | smooth_current | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_SmoothCurrent | fold_3 | 42 | smooth_current | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_Full | fold_1 | 42 | full | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_Full | fold_2 | 42 | full | 1.000000 | 1.000000 | 1.000000 |
| PGFS_TimeXer_Full | fold_3 | 42 | full | 1.000000 | 1.000000 | 1.000000 |

| model | fold | seed | max_consecutive_physics_dominance_epochs | physics_loss_status |
|---|---|---|---|---|
| FS_TimeXer_NoPhysics | fold_1 | 42 | 0 | BALANCED |
| FS_TimeXer_NoPhysics | fold_2 | 42 | 0 | BALANCED |
| FS_TimeXer_NoPhysics | fold_3 | 42 | 0 | BALANCED |
| PGFS_TimeXer_Full | fold_1 | 42 | 0 | BALANCED |
| PGFS_TimeXer_Full | fold_2 | 42 | 0 | BALANCED |
| PGFS_TimeXer_Full | fold_3 | 42 | 0 | BALANCED |
| PGFS_TimeXer_Smooth | fold_1 | 42 | 0 | BALANCED |
| PGFS_TimeXer_Smooth | fold_2 | 42 | 0 | BALANCED |
| PGFS_TimeXer_Smooth | fold_3 | 42 | 0 | BALANCED |
| PGFS_TimeXer_SmoothCurrent | fold_1 | 42 | 0 | BALANCED |
| PGFS_TimeXer_SmoothCurrent | fold_2 | 42 | 0 | BALANCED |
| PGFS_TimeXer_SmoothCurrent | fold_3 | 42 | 0 | BALANCED |
| SA_TimeXer_Oracle | fold_1 | 42 | 0 | BALANCED |
| SA_TimeXer_Oracle | fold_2 | 42 | 0 | BALANCED |
| SA_TimeXer_Oracle | fold_3 | 42 | 0 | BALANCED |
| SA_TimeXer_Ridge | fold_1 | 42 | 0 | BALANCED |
| SA_TimeXer_Ridge | fold_2 | 42 | 0 | BALANCED |
| SA_TimeXer_Ridge | fold_3 | 42 | 0 | BALANCED |
| SA_TimeXer_Shuffled | fold_1 | 42 | 0 | BALANCED |
| SA_TimeXer_Shuffled | fold_2 | 42 | 0 | BALANCED |
| SA_TimeXer_Shuffled | fold_3 | 42 | 0 | BALANCED |
| SA_TimeXer_Zero | fold_1 | 42 | 0 | BALANCED |
| SA_TimeXer_Zero | fold_2 | 42 | 0 | BALANCED |
| SA_TimeXer_Zero | fold_3 | 42 | 0 | BALANCED |
