"""Shared real-model synthetic fixture; invoked only by the guarded P1 runner."""
CONFIG=None
PROTOCOL=None
PROTOCOL_SHA=None
DESTINATION=None
RESULTS={}

def evidence(seed):
    if CONFIG is None:
        raise RuntimeError('RUN_TESTS_THROUGH_GUARDED_P1_ENTRYPOINT')
    if seed not in RESULTS:
        from p1_checks import check_seed
        RESULTS[seed]=check_seed(seed,CONFIG,PROTOCOL_SHA,DESTINATION)
    return RESULTS[seed]
