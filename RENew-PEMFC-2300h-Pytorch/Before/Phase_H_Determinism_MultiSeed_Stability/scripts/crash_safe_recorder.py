"""Fsynced stage journal plus atomic cumulative latest-batch state."""
from __future__ import annotations
import json
import os
from pathlib import Path
from h_common import atomic_json

class CrashSafeRecorder:
    def __init__(self,folder,run_id):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.run_id=run_id
        self.state={};self.sequence=0
    def emit(self,stage,**payload):
        if stage=='BATCH_STARTED':self.state={'run_id':self.run_id,'stages':[]}
        self.state.update(payload);self.state['stage']=stage;self.state['stages'].append(stage)
        self.sequence+=1;self.state['sequence']=self.sequence
        event={'run_id':self.run_id,'sequence':self.sequence,'stage':stage,
               'epoch':self.state.get('epoch'),'batch':self.state.get('batch'),**payload}
        # Journal fsync precedes snapshot replace. A crash can leave an older
        # snapshot but never loses the successfully persisted journal event.
        with (self.folder/'batch_events.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        atomic_json(self.folder/'latest_batch.json',self.state)
