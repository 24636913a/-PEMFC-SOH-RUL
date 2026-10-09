"""Optuna bookkeeping contracts for a future separately authorized run."""
from __future__ import annotations
from pathlib import Path
from typing import Any
class OptunaContractError(RuntimeError): pass
def validate_trial_closure(states:list[str],minimum_completed:int=12,budget:int=16)->dict[str,Any]:
    attempted=len(states); completed=states.count('COMPLETE'); unresolved=[s for s in states if s in {'RUNNING','WAITING'}]; invalid=[s for s in states if s not in {'COMPLETE','FAILED','PRUNED','RUNNING','WAITING'}]
    ok=attempted==budget and minimum_completed<=completed<=budget and not unresolved and not invalid
    return {'pass':ok,'attempted':attempted,'completed':completed,'unresolved':unresolved,'invalid':invalid}
def assert_new_study_path(path:Path,phase_root:Path,old_phase_root:Path)->None:
    resolved=path.resolve(); root=phase_root.resolve()
    if root not in resolved.parents or old_phase_root.resolve() in resolved.parents or 'T1R' not in path.name: raise OptunaContractError('STUDY_PATH_ISOLATION_FAILURE')
def fixed_trial_zero(cfg:dict[str,Any])->dict[str,Any]: return dict(cfg['search']['fixed_trial_0'])
