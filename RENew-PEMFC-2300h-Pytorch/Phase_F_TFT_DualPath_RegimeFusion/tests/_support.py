from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from models import DualPathStrictHistoryTFT, RegimeAwareDualPathTFT  # noqa: E402


MODEL_KWARGS = dict(
    hidden_size=64,
    hidden_continuous_size=32,
    attention_head_size=4,
    lstm_layers=1,
    dropout=0.1,
)


def make_history(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(219)
    return torch.randn(batch, 48, 14, generator=generator)


def make_f1(seed: int = 101) -> DualPathStrictHistoryTFT:
    torch.manual_seed(seed)
    return DualPathStrictHistoryTFT(**MODEL_KWARGS, dual_path_enabled=True)


def make_f2(seed: int = 101) -> RegimeAwareDualPathTFT:
    torch.manual_seed(seed)
    return RegimeAwareDualPathTFT(**MODEL_KWARGS, dual_path_enabled=True)


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def named_state_hash(module: torch.nn.Module, prefix: str) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name.startswith(prefix):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
