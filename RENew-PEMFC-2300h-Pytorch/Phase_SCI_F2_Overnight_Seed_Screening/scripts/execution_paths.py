"""Versioned continuation state; earlier ledgers remain immutable evidence."""
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def progress_path():
 latest=P/'outputs/execution_ledger/PROGRESS_IO_REPAIR.json'
 return latest if latest.exists() else P/'outputs/execution_ledger/PROGRESS_ENGINEERING_REPAIR.json'
