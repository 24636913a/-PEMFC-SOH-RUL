"""Minimal fusion adapter, fresh synthetic initialization, no training entry."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PHASE=Path(__file__).resolve().parents[1]
RESEARCH=PHASE.parent
REPO=RESEARCH.parent
F=RESEARCH/'Phase_F_TFT_DualPath_RegimeFusion/models'
E=RESEARCH/'Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py'
C=RESEARCH/'Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py'
H=RESEARCH/'Phase_H_Determinism_MultiSeed_Stability/scripts/h_common.py'


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(tensor):
    arr=tensor.detach().cpu().contiguous().numpy()
    return digest_bytes(str((arr.shape,str(arr.dtype))).encode()+arr.tobytes())


def state_digest(state):
    return digest_bytes(json.dumps({k:tensor_digest(v) for k,v in sorted(state.items())},sort_keys=True).encode())


def rng_snapshot():
    return {'cpu':torch.get_rng_state().clone(),
            'cuda':[s.clone() for s in torch.cuda.get_rng_state_all()],
            'python':random.getstate(),'numpy':np.random.get_state()}


def restore_rng(snapshot):
    random.setstate(snapshot['python']);np.random.set_state(snapshot['numpy'])
    torch.set_rng_state(snapshot['cpu']);torch.cuda.set_rng_state_all(snapshot['cuda'])


def rng_digest(snapshot, generator=None):
    n=snapshot['numpy']
    record={'cpu':tensor_digest(snapshot['cpu']),
            'cuda':[tensor_digest(v) for v in snapshot['cuda']],
            'python':digest_bytes(repr(snapshot['python']).encode()),
            'numpy':digest_bytes(str(n[0]).encode()+n[1].tobytes()+repr(n[2:]).encode())}
    if generator is not None:
        record['loader_generator']=tensor_digest(generator.get_state())
    return record


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed):
    saved=rng_snapshot()
    try:
        set_seed(seed)
        yield
    finally:
        restore_rng(saved)


def extract_function(path, name, namespace):
    """Compile only an audited pure function; never execute a legacy module."""
    text=path.read_text(encoding='utf-8');module=ast.parse(text)
    nodes=[n for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(nodes)!=1:
        raise RuntimeError('H_FUNCTION_IDENTITY_UNRESOLVED:'+name)
    node=nodes[0]
    code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec')
    env=dict(namespace);exec(code,env)
    return env[name],digest_bytes(ast.dump(node,include_attributes=False).encode())


def load_frozen_classes():
    name='_p1_frozen_f_models'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,F/'__init__.py',submodule_search_locations=[str(F)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    else:
        module=sys.modules[name]
    return module.DualPathStrictHistoryTFT,module.RegimeAwareDualPathTFT


G0,G2=load_frozen_classes()


class GlobalContextGateTFT(G2):
    """Exactly one override: original gate receives the same 5-vector for all windows."""
    def regime_context(self, encoder_cont):
        if encoder_cont.ndim!=3 or encoder_cont.shape[-1]<14:
            raise RuntimeError('INVALID_ENCODER_CONT')
        return torch.ones((encoder_cont.shape[0],5),dtype=encoder_cont.dtype,device=encoder_cont.device)


CLASSES={'G0':G0,'G1':GlobalContextGateTFT,'G2':G2}


def constructor(group, config):
    if group not in CLASSES:
        raise RuntimeError('UNKNOWN_GROUP')
    p=config['hyperparameters']
    return CLASSES[group](hidden_size=p['hidden_size'],hidden_continuous_size=p['hidden_continuous_size'],
                         attention_head_size=4,lstm_layers=1,dropout=p['dropout'])


def split_state(model):
    core={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('regime_gate.')}
    gate={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('regime_gate.')}
    return core,gate


def build_cohort(seed,config,device,order=('G0','G1','G2')):
    if set(order)!=set(CLASSES) or len(order)!=3:
        raise RuntimeError('THREE_GROUPS_REQUIRED')
    # Fresh random reference templates, not historical checkpoints.
    with isolated_rng(seed):
        prototype=constructor('G0',config);common_core,_=split_state(prototype)
    del prototype
    with isolated_rng(seed+100000):
        prototype=constructor('G2',config);_,common_gate=split_state(prototype)
    del prototype
    answer={}
    for group in order:
        with isolated_rng(seed):
            model=constructor(group,config)
            expected={**common_core,**(common_gate if group!='G0' else {})}
            model.load_state_dict(expected,strict=True)
            answer[group]=model.to(device)
    return answer


def make_index_loader(seed):
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.arange(1380)),batch_size=32,shuffle=True,
                      generator=generator,num_workers=0)
    return loader,generator


def sampler_two_epochs(seed):
    loader,generator=make_index_loader(seed)
    initial=tensor_digest(generator.get_state());epochs=[]
    for _ in range(2):
        indices=torch.cat([batch[0] for batch in loader]).numpy()
        if sorted(indices.tolist())!=list(range(1380)):
            raise RuntimeError('SAMPLER_NOT_FULL_PERMUTATION')
        epochs.append({'order':indices.tolist(),'order_sha256':digest_bytes(indices.tobytes()),
                       'generator_after_sha256':tensor_digest(generator.get_state())})
    return {'initial_generator_sha256':initial,'epochs':epochs}


def synthetic_input(n=65):
    x=np.random.default_rng(991).normal(0,1,(n,48,14)).astype(np.float32)
    x[:,:,[1,2,5,12]]=0.
    return x


_c_predict,_c_predict_ast_sha=extract_function(C,'predict',{'torch':torch,'np':np})
_e_predict,_e_predict_ast_sha=extract_function(E,'predict',{'torch':torch,'np':np,'c':SimpleNamespace(predict=_c_predict)})


def predict_synthetic(model,x,device,batch_size=32):
    if batch_size!=32 or x.dtype!=np.float32 or x.shape[1:]!=(48,14):
        raise RuntimeError('H_PREDICTION_CONTRACT_CHANGED')
    if torch.is_autocast_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError('H_NUMERIC_MODE_CHANGED')
    return _e_predict(model,x,32,device)


def configure_runtime(config):
    environment,env_sha=extract_function(H,'environment',{'torch':torch,'np':np,'os':os,'platform':__import__('platform')})
    configure,config_sha=extract_function(H,'configure_normal',{'torch':torch,'environment':environment})
    result=configure(config)
    if torch.get_default_dtype()!=torch.float32:
        raise RuntimeError('DEFAULT_DTYPE_CHANGED')
    return result,{'H_environment_AST_sha256':env_sha,'H_configure_normal_AST_sha256':config_sha,
                   'C_predict_AST_sha256':_c_predict_ast_sha,'E_predict_AST_sha256':_e_predict_ast_sha}


def save_synthetic_checkpoint(path,model,group,seed,protocol_sha):
    path=Path(path).resolve()
    if not path.is_relative_to((PHASE/'runtime').resolve()) or path.suffix!='.pt':
        raise PermissionError('SYNTHETIC_CHECKPOINT_PATH_ONLY')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'synthetic_only':True,'group':group,'seed':seed,'protocol_sha256':protocol_sha,
                'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},path)
    return digest_bytes(path.read_bytes())


def reload_synthetic_checkpoint(path,group,seed,config,device,protocol_sha):
    path=Path(path).resolve()
    if not path.is_relative_to((PHASE/'runtime').resolve()) or path.suffix!='.pt':
        raise PermissionError('HISTORICAL_CHECKPOINT_LOAD_PROHIBITED')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('synthetic_only') is not True or payload.get('group')!=group or payload.get('seed')!=seed or payload.get('protocol_sha256')!=protocol_sha:
        raise RuntimeError('SYNTHETIC_CHECKPOINT_IDENTITY_MISMATCH')
    with isolated_rng(seed):
        model=constructor(group,config).to(device)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model


def early_stop_improves(validation_standardized_RMSE,best):
    return validation_standardized_RMSE < best - 1e-12


def request_real_training(*args,**kwargs):
    raise PermissionError('ENGINEERING_PREFLIGHT_ONLY_TRAINING_NOT_AUTHORIZED')
