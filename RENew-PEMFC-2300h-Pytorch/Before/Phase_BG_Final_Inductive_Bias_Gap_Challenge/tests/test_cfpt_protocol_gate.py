"""Exercise the production CFPT source-only gate without any data access."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from run_cfpt_protocol_gate import analyze_protocol_source, load_config, source_fingerprints  # noqa: E402


class CFPTProtocolGateTest(unittest.TestCase):
    def test_verified_official_sources_and_fail_closed_protocol_gate(self) -> None:
        cfg = load_config()
        hashes = source_fingerprints(cfg)
        result = analyze_protocol_source(cfg)
        self.assertTrue(hashes["pass"].all())
        self.assertEqual(result["route"], "INCOMPATIBLE")
        self.assertTrue(result["cfpt_requires_decoder_timestamp"])
        self.assertFalse(result["timesnet_functionally_consumes_decoder_timestamp"])
        self.assertFalse(result["development_data_read"])
        self.assertFalse(result["test_accessed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
