# Phase T1 — Canonical TFT Hyperparameter Optimization

This independent, Development-only phase evaluates whether a bounded nested
Optuna-TPE search can improve the frozen canonical strict-history TFT without
changing its task, data firewall, information set, or upstream TFT identity.

The study contains three distinct comparators: frozen T0 (300 epochs), a T0
configuration retrained at T1's 450-epoch budget, and an inner-train-selected
Optuna configuration retrained once per outer fold.  Test rows and future
measured operating variables are prohibited.
