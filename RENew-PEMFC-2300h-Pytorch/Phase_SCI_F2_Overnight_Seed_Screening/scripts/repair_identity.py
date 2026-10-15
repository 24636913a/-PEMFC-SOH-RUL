"""Immutable, non-scientific logger repair. Failed101 is never retried."""
from screen_common import *
def verify_repair():
 f=P/'configs/LOGGER_REPAIR_AMENDMENT.json';assert sha(f)==f.with_suffix('.sha256').read_text().strip()
 r=json.loads(f.read_text());assert r['original_protocol_sha256']==sha(P/'configs/frozen_seed_screening.json')
 for path,digest in r['code_sha256'].items():assert sha(P/path)==digest,('ENGINEERING_REPAIR_IDENTITY_MISMATCH',path)
 assert r['failed_seed_not_retried']==101 and r['scientific_configuration_changed'] is False
 return r
