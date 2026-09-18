# Reference-condition audit

`Z_ref` is the nine-variable condition vector at the prediction origin’s last fully observed 0.5 h bin. It precedes every future target, uses the identical feature definition and active-area density convention, and is transformed by the existing fold train-only future-condition scaler. No future value is used to construct it.

The absolute P2-current departure from `Z_ref` has only a weak linear association with the absolute P2−P1 correction change across Development points (Pearson r=0.075). This is descriptive, not causal. It shows that reference distance alone does not explain correction magnitude or error change; redundant current/density variables were not interpreted as independent physical effects.

The proposed difference `C(X,Z_P2,h)-C(X,Z_ref)` nevertheless has a clear testable mathematical property: it is exactly zero when `Z_P2,h=Z_ref`. This is a structural motivation, not proof that OPR will improve prediction.
