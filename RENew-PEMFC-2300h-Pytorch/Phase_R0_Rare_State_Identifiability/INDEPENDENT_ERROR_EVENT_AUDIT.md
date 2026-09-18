# Independent error-event audit

Global de-duplicated train OOF events: E1=66, E2=9. Development Validation events: E1=9, E2=2. Events merge repeated origins at a target timestamp and only merge adjacent half-hour timestamps of the same type within a temporal segment.

Train E1 has mean q_true=0.000, q_pred=0.405, probability=0.828, current absolute error=83.80 A. Validation E1 is similar: 0.000, 0.392, 0.706, 72.22 A. Evidence: event CSVs in `outputs/train_oof` and `outputs/events`.
