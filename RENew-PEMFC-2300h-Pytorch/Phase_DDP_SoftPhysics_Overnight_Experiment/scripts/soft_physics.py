"""Voltage-domain objective, without altering the frozen dynamic G2 model."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def _scale(value: float) -> float:
    value=float(value)
    if not math.isfinite(value) or value<=0:
        raise ValueError("POSITIVE_FINITE_VOLTAGE_SCALE_REQUIRED")
    return value

def restore_absolute_voltage(delta_std, origin_voltage_V, target_scale_V):
    scale=_scale(target_scale_V)
    if delta_std.ndim!=2 or delta_std.shape[1]!=24:
        raise ValueError("DELTA_CONTRACT_B_24_REQUIRED")
    anchor=origin_voltage_V.reshape(-1,1).to(delta_std)
    if len(anchor)!=len(delta_std) or not torch.isfinite(anchor).all() or not torch.isfinite(delta_std).all():
        raise ValueError("INVALID_ORIGIN_OR_PREDICTION")
    # DeltaV is scaled only: never add the input-voltage scaler mean.
    return anchor+scale*delta_std

def voltage_domain_penalty(voltage_V, scale_V, minimum_V=0.0, maximum_V=None):
    scale=_scale(scale_V)
    if not torch.isfinite(voltage_V).all():
        raise ValueError("NONFINITE_ABSOLUTE_VOLTAGE")
    if minimum_V is None and maximum_V is None:
        raise ValueError("AT_LEAST_ONE_REGISTERED_BOUND_REQUIRED")
    for bound in (minimum_V,maximum_V):
        if bound is not None and not math.isfinite(float(bound)):
            raise ValueError("BOUND_MUST_BE_FINITE_OR_NONE")
    if minimum_V is not None and maximum_V is not None and minimum_V>=maximum_V:
        raise ValueError("BOUND_ORDER")
    penalty=torch.zeros_like(voltage_V)
    if minimum_V is not None:
        penalty=penalty+F.relu((minimum_V-voltage_V)/scale).square()
    if maximum_V is not None:
        penalty=penalty+F.relu((voltage_V-maximum_V)/scale).square()
    return penalty.mean()

def combined_objective(delta_std,target_std,origin_voltage_V,target_scale_V,
                       weight=0.01,minimum_V=0.0,maximum_V=None):
    if delta_std.shape!=target_std.shape:
        raise ValueError("TARGET_SHAPE_MISMATCH")
    if not math.isfinite(float(weight)) or weight<0:
        raise ValueError("NONNEGATIVE_FINITE_WEIGHT_REQUIRED")
    if not torch.isfinite(target_std).all():
        raise ValueError("NONFINITE_TARGET")
    absolute=restore_absolute_voltage(delta_std,origin_voltage_V,target_scale_V)
    data=F.mse_loss(delta_std,target_std)
    physics=voltage_domain_penalty(absolute,target_scale_V,minimum_V,maximum_V)
    # Exact scalar and gradient equivalence for lambda=0.
    total=data if weight==0 else data+weight*physics
    return total,data,physics,absolute

def physics_activity(voltage_V,scale_V,minimum_V=0.0,maximum_V=None):
    scale=_scale(scale_V)
    violations=torch.zeros_like(voltage_V,dtype=torch.bool)
    excess=torch.zeros_like(voltage_V)
    if minimum_V is not None:
        violations|=voltage_V<minimum_V
        excess+=torch.relu(minimum_V-voltage_V)
    if maximum_V is not None:
        violations|=voltage_V>maximum_V
        excess+=torch.relu(voltage_V-maximum_V)
    return {"n_forecasts":int(voltage_V.numel()),
            "violations":int(violations.sum().item()),
            "activation_fraction":float(violations.float().mean().item()),
            "minimum_predicted_voltage_V":float(voltage_V.min().item()),
            "maximum_predicted_voltage_V":float(voltage_V.max().item()),
            "mean_violation_V":float(excess.mean().item()),
            "maximum_violation_V":float(excess.max().item()),
            "physics_loss_dimensionless":float(voltage_domain_penalty(
                voltage_V,scale,minimum_V,maximum_V).item())}

def require_training_admission(decision):
    if decision.get("formal_training_admitted") is not True:
        raise PermissionError("SOFT_PHYSICS_FORMAL_TRAINING_NOT_ADMITTED")
    required=("source_identity_pass","physical_domain_pass",
              "train_activity_pass","weight_balance_pass","model_parity_pass")
    if any(decision.get(key) is not True for key in required):
        raise PermissionError("SOFT_PHYSICS_ADMISSION_GATE_FAILED")
