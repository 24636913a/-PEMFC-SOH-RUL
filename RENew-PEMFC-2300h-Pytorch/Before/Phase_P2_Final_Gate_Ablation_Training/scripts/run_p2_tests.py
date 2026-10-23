"""No formal optimizer steps; production guard tests only."""
import sys
import unittest
from p2_common import PHASE,install_firewall
if __name__=='__main__':
    install_firewall()
    suite=unittest.defaultTestLoader.discover(str(PHASE/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
