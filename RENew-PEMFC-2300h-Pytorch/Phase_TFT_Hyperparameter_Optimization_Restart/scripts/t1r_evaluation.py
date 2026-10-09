"""Identity-first B/C evaluation artifact checks, without scoring model output."""
from __future__ import annotations
from collections import Counter
from typing import Any
class EvaluationIdentityError(RuntimeError): pass
def validate_pair_rows(rows:list[dict[str,Any]],protocol_sha:str)->dict[str,Any]:
    required={'seed','model','fold','origin_row','target_row','horizon_step','y_true_delta_V','protocol_sha256','run_id'}
    if any(required-set(r) for r in rows): raise EvaluationIdentityError('PAIR_FIELDS_MISSING')
    if any(r['protocol_sha256']!=protocol_sha for r in rows): raise EvaluationIdentityError('PAIR_PROTOCOL_MISMATCH')
    by_model={m:[r for r in rows if r['model']==m] for m in ('B','C')}
    if not by_model['B'] or not by_model['C']: raise EvaluationIdentityError('PAIR_MODEL_MISSING')
    key=lambda r:(r['seed'],r['fold'],r['origin_row'],r['target_row'],r['horizon_step'],r['y_true_delta_V'])
    if Counter(map(key,by_model['B']))!=Counter(map(key,by_model['C'])): raise EvaluationIdentityError('BC_QUERY_OR_TRUTH_MISMATCH')
    return {'pass':True,'paired_queries':len(by_model['B'])}
def validate_multiseed_summary(rows:list[dict[str,Any]],protocol_sha:str)->dict[str,Any]:
    required={(seed,model,fold) for seed in (42,2024,3407) for model in ('B','C') for fold in ('fold_1','fold_2','fold_3')}
    seen=[(r.get('seed'),r.get('model'),r.get('fold')) for r in rows]
    if set(seen)!=required or len(seen)!=len(required) or any(r.get('protocol_sha256')!=protocol_sha for r in rows): raise EvaluationIdentityError('MULTISEED_IDENTITY_INCOMPLETE_OR_DUPLICATE')
    return {'pass':True,'records':len(rows)}
