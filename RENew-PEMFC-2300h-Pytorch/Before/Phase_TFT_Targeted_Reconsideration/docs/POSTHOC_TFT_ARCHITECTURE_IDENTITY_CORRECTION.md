# Posthoc correction: TFT architecture identity

## Historical protocol status retained

The protocol-defined terminal status remains **`TFT_NOT_VIABLE`**. Its scientific interpretation is narrowed to **`STRICT_HISTORY_TFT_ADAPTATION_NOT_VIABLE`**: this project's strict-history TFT-style adaptation did not pass its pre-registered Development feasibility gate.

It must not be read as `TFT_FAMILY_NOT_VIABLE`, nor as evidence that the original Temporal Fusion Transformer is unsuitable for RealHyFC.

## What was actually implemented

The executed file [`../scripts/tft_models.py`](../scripts/tft_models.py) is a project-specific strict-history TFT-style implementation, not a line-by-line or structure-complete reproduction of the original TFT architecture. It deliberately retains continuous variable selection, GRN-style gates, a local recurrent route, causal attention, fixed zero decoder tokens and absolute-scaled-voltage output semantics. However, the following implementation facts delimit its identity:

1. The recurrent encoder and decoder share one `torch.nn.LSTM` instance: the encoder native `(h, c)` state initializes the decoder. This is a state-handoff implementation choice, not a separate original-TFT encoder/decoder module pair.
2. Temporal attention is generic `torch.nn.MultiheadAttention`, rather than an implementation of TFT's interpretable multi-head attention component.
3. The call sets `need_weights=False`; no interpretable attention weights of the original TFT design are produced or analyzed.
4. Static covariates and known-future covariates are absent. For the frozen `STRICT_UNKNOWN_FUTURE` forecasting task this is an appropriate data-availability constraint.
5. The shared recurrent module, generic attention implementation, and disabled attention weights are **not** consequences that strict-history forecasting logically requires. They are additional architecture choices of this adaptation.

## Consequence

The negative gate result qualifies only this exact pre-registered implementation and task adaptation. It neither reruns nor revises any model, prediction, metric, gate, protocol hash, historical phase, SOH construction, or RUL conclusion.
