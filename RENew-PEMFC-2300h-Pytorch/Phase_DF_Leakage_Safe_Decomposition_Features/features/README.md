# Origin-window feature cache contract

Formal SimpleTrend, VMD and EMD representations are stored only as ignored, local `cache/<method>/origin_<row>.npz` files. Every entry is a `[48, 2]` feature block tied to one prediction origin; it is deliberately not a global row-level feature table. The committed cache provenance and local-artifact manifests record their identity, SHA256 and reconstruction commands.
