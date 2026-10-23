# Label identity audit

Official parent mixed label equals one only when parent zero count is positive and strictly below parent valid-current count. All common targets are reconstructed from children and checked against direct parent raw statistics before model fitting.

| check | status | targets_checked |
| --- | --- | --- |
| parent q equals summed child raw counts | PASS | 125664 |
| parent mixed label equals summed child raw counts | PASS | 125664 |

Tests cover all-on, all-zero, q0 plus q1, internally mixed, and incomplete children.
