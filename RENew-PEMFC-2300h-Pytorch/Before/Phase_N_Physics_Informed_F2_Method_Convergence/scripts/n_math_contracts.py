"""Pure synthetic contracts: no files, arrays from the project, or models."""
import math


def polarization(j, E, b, log_j0, asr, B, jlim):
    if not 0 < j < jlim or min(b, asr, B) < 0:
        raise ValueError("PHYSICAL_DOMAIN_UNCLOSED")
    return E - b * (math.log(j) - log_j0) - j * asr + B * math.log1p(-j / jlim)


def loss_student(s, y, t, lam):
    if lam < 0:
        raise ValueError("NEGATIVE_WEIGHT")
    return (s - y) ** 2 + lam * (s - t) ** 2


def completed_square(s, y, t, lam):
    target = (y + lam * t) / (1 + lam)
    return (1 + lam) * (s - target) ** 2 + lam / (1 + lam) * (y - t) ** 2


def output_gradient(s, y, t, lam):
    return 2 * (s - y) + 2 * lam * (s - t)


def order_penalty(v_low, v_high):
    return max(0, v_high - v_low) ** 2


def physical_residual(u, anchor_V, scale_V, teacher_V):
    if scale_V <= 0:
        raise ValueError("INVALID_SCALE")
    return (anchor_V + scale_V * u - teacher_V) / scale_V


def require_training_qualification(gates):
    # Missing keys cannot silently qualify a physics route.
    required = {"equation", "identifiability", "domain", "information", "isolation", "independence"}
    if set(gates) != required or not all(value is True for value in gates.values()):
        raise ValueError("NO_PHYSICS_F2_TRAINING_ROUTE_QUALIFIED")


def student_interface(fields):
    if set(fields) != {"history48", "relative_horizon", "origin_anchor"}:
        raise ValueError("INFORMATION_SET_MISMATCH")


def crossfit_contract(teacher_dependencies, receiving_dependencies, train_end,
                      latest_teacher_row, earliest_receiving_row):
    """Symbolic source identities, not real row parsing.

    Dependencies include hist48, anchor, target24, raw dwell and diagnostic IDs.
    All teacher fitted scalers/parameters share the same cutoff.
    """
    if (not teacher_dependencies or not receiving_dependencies
            or teacher_dependencies & receiving_dependencies
            or latest_teacher_row >= earliest_receiving_row
            or not 0 <= latest_teacher_row < train_end
            or not 0 <= earliest_receiving_row < train_end):
        raise ValueError("CROSSFIT_ISOLATION_FAILED")
    return True
