import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time
from operator import itemgetter
import io

# Imports the Google Cloud client library
from google.cloud import storage


# Checkpointing and restoring, inspired from CheckFreq

def _to_cpu(ele, snapshot=None):
	#while True:
         
        if snapshot is None:
            snapshot = {}

        if hasattr(ele, 'cpu'):
            snapshot = ele.cpu()
        elif isinstance(ele, dict):
            snapshot = {}
            for k,v in ele.items():
                snapshot[k] = None
                snapshot[k] = _to_cpu(v, snapshot[k])
        elif isinstance(ele, list):
            snapshot  = [None for _ in range(len(ele))]
            for idx, v in enumerate(ele):
                snapshot[idx] = _to_cpu(v, snapshot[idx])
        else:
            return ele
        return snapshot	


class CFCheckpoint():
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.tracking_map = OrderedDict()
        self.spawned=False
        self.chk_process=None
		
        for name, ref in kwargs.items():
			#if hasattr(ref, 'state_dict'):
            self.tracking_map[name] = ref
			#else:
			#	self.logger.info("Skipping object `{}` in CF Checkpointing. \
			#	No state_dict() method exposed".format(name))

        self.num_tracking = len(self.tracking_map.keys())

        if self.num_tracking == 0:
            raise ValueError("Nothing to track")


    def _snapshot(self,active_snapshot, additional_state=None):

        self.latest_snapshot = OrderedDict()
        start = time.time()
		# Snapshot the state of tractable items
        for name, ref in self.tracking_map.items():
            if name not in self.latest_snapshot:
                if hasattr(ref, 'state_dict'):
                    self.latest_snapshot[name] = _to_cpu(ref.state_dict())
                else:
                    self.latest_snapshot[name] = {}
                    for n,r in ref.items():
                        self.latest_snapshot[name][n] = _to_cpu(r)
						#print(n, r._grad.shape)
					#self.latest_snapshot[name] = copy.deepcopy(ref)
            else:
                self.logger.info("Repeated entry for {}".format(name))
                return False

        if additional_state:
            self.latest_snapshot.update(additional_state)

        return True

    def checkpoint(
        self,
		active_snapshot,
		lock,
		change,
		additional_state=None,
		background=False,
                profile_snap = None):


        self.client = storage.Client() # need to set up credentials
        self.bucket = self.client.get_bucket('torchelastic')

        while True:

            if background:
                with lock:
                    if change.value==0:		
                        continue                
            
            print("[{}] START ASYNC".format(time.time()))

            start = time.time()
            success = self._snapshot(active_snapshot.value, additional_state=additional_state)
            end1 = time.time()
            print("-------------- Snapshot took: ", end1-start)

            if success:		
                with lock:
                    active_snapshot.value = 0
            else:
                with lock:
                    change.value=0
                self.logger.error("Cannot persist. Empty snapshot")
                return

        
            snapshot = self.latest_snapshot
            skeys = list(snapshot['optimizer']['state'].keys())
            k = skeys[-1]

            with lock:
                if background and profile_snap.value == 1:
                    # don't checkpoint
                    snapshot={}
                    change.value=0
                    continue

            # do the actual checkpoint here
            print("-- FROM CHECKPOINT: ", k, snapshot['optimizer']['state'][k])

            if not background:
                print(" *** ------------------------------------ TEMPORARY, exit now -----------------------------------")
                return	

            rem = time.time()

            buf = io.BytesIO() # kept in memory
            torch.save(snapshot, buf)
            buf.seek(0)  

            
            epoch = snapshot['epoch']
            it = snapshot['iter']

            new_blob = self.bucket.blob('model_ep'+str(epoch)+'_it'+str(it)+'.chk')
            new_blob.upload_from_file(buf)
            
            print("checkpointing in GCS took: ", time.time()-rem)
            

            with lock:
                snapshot={}
                change.value=0


