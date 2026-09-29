# Phase PI — PDFNet-inspired exploratory architecture

This is a single preregistered **exploratory** architecture experiment for
RealHyFC's 48→24 Development-only voltage forecasting task. Its only model
name is `PDFNetInspired_Exploratory`.

It is not an Official PDFNet implementation, an author reproduction, or a
backbone-promotion experiment. Phase BS and Phase BG remain read-only, and the
formal backbone remains `TIMESNET_FREEZE` regardless of this phase's outcome.

The experiment is closed after its frozen resource, seed-42, and—only if
eligible—multi-seed gates. No Test use, external decomposition, feature
selection, hyperparameter sweep, architecture iteration, or automatic next
phase is permitted.

## Factual closure

Phase PI evaluated one preregistered D=128 `PDFNetInspired_Exploratory`
configuration. Its seed-42 formal RMSE was 32.2135 mV, compared with the
same-protocol TimesNet seed-42 reference of 14.2066 mV. The seed-42
exploratory-potential gate therefore returned `PDFNET_INSPIRED_NO_PROMISE`, so
seeds 2024 and 3407 were not run.

The saved activation diagnostic shows that most outer-period segments contained
one inner patch and that continuous cross-patch GPRU updates were activated
only for a limited fraction of recorded patch updates. This is not a causal
explanation of performance and does not imply that official PDFNet, GPRU, or
periodic recurrent networks failed on RealHyFC. `TIMESNET_FREEZE` remains the
formal backbone, and backbone search is `CLOSED`.
