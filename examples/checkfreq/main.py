#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Source: `pytorch imagenet example <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_ # noqa B950

Modified and simplified to make the original pytorch example compatible with
torchelastic.distributed.launch.

Changes:

1. Removed ``rank``, ``gpu``, ``multiprocessing-distributed``, ``dist_url`` options.
   These are obsolete parameters when using ``torchelastic.distributed.launch``.

2. Removed ``seed``, ``evaluate``, ``pretrained`` options for simplicity.

3. Removed ``resume``, ``start-epoch`` options.
   Loads the most recent checkpoint by default.

4. ``batch-size`` is now per GPU (worker) batch size rather than for all GPUs.

5. Defaults ``workers`` (num data loader workers) to ``0``.

Usage

::

 >>> python -m torchelastic.distributed.launch
        --nnodes=$NUM_NODES
        --nproc_per_node=$WORKERS_PER_NODE
        --rdzv_id=$JOB_ID
        --rdzv_backend=etcd
        --rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
        main.py
        --arch resnet18
        --epochs 20
        --batch-size 32
        <DATA_DIR>
"""

import argparse
import io
import os
import shutil
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import List, Tuple

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchelastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from google.cloud import storage

from statistics import mean
import math
from operator import itemgetter
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, freeze_support, spawn
import ctypes
import chk_manager

client = storage.Client() # need to set up credentials
bucket = client.get_bucket('torchelastic')

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Elastic ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 32), per worker (GPU)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--dist-backend",
    default="nccl",
    choices=["nccl", "gloo"],
    type=str,
    help="distributed backend",
)
parser.add_argument(
    "--checkpoint-file",
    default="/tmp/checkpoint.pth.tar",
    type=str,
    help="checkpoint file path, to load and save to",
)

parser.add_argument(
    "--local_rank",
    default=0,
    type=int,
    help="local rank - not used",
)


parser.add_argument(
    "--ch-freq",
    default=10,
    type=int,
    help="checkpoint frequency (iterations)",
)

parser.add_argument('--profile', type=bool, default=True,
                    help='whether to profile or not according to CheckFreq')

parser.add_argument('--prof-steps', type=int, default=50,
                    help='number of steps to profile for')

parser.add_argument('--max-overhead', type=int, default=5,
                    help='overhead (%) of checkpointing over the total execution')


def main():
    args = parser.parse_args()
    device_id = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}, global rank = {global_rank}")

    # when using NCCL, on failures, surviving nodes will deadlock on NCCL ops
    # because NCCL uses a spin-lock on the device. Set this env var and
    # to enable a watchdog thread that will destroy stale NCCL communicators
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
    )

    print("-----initialize model------")
    model, criterion, optimizer = initialize_model(
        args.arch, args.lr, args.momentum, args.weight_decay, device_id
    )

    print("----------- init data loader---------")
    train_loader, val_loader = initialize_data_loader(
        args.data, args.batch_size, args.workers
    )

    print("LOAD CHECKPOINT")
    # resume from checkpoint if one exists;
    state = load_checkpoint(
        args.checkpoint_file, device_id, args.arch, model, optimizer
    )

    start_epoch = max(state.epoch, 0)
    start_iter = state.iter + 1
    print(f"=> start_epoch: {start_epoch}, start_iter: {start_iter}")

    print_freq = args.print_freq
    

    # for asynchronous checkpointing - CheckFreq based
    for name,ref in model.state_dict().items():
        make_shm(ref)

    for name,ref in optimizer.state_dict().items():
        make_shm(ref)

    chk = chk_manager.CFCheckpoint(model=model.state_dict(), optimizer=optimizer.state_dict())

    active_snapshot = Value('i', 0)
    lock = Lock()
    mp_manager = Manager()
    profile_snap = Value('i', 0)

    change = Value('i', 0)					
    additional_snapshot = mp_manager.dict()

    print("=> Checkpoint instance initialized!")

    for epoch in range(start_epoch, args.epochs):
        state.epoch = epoch
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr)

        if epoch == start_epoch:
            siter = start_iter
        else:
            siter = 0
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, siter, device_id, print_freq, 
                    args.ch_freq, additional_snapshot, chk, active_snapshot, lock, change, global_rank,
                    args.profile, args.prof_steps, args.max_overhead, profile_snap)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device_id, print_freq)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > state.best_acc1
        state.best_acc1 = max(acc1, state.best_acc1)

        #if device_id == 0:
        #    save_checkpoint(state, is_best, args.checkpoint_file)


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.iter = -1
        self.arch = arch
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """
        print(self.model.state_dict()['module.fc.bias'])

        print(obj.keys())
        self.epoch = obj["epoch"]
        self.iter = obj["iter"]
        self.model.load_state_dict(obj["model"])
        self.optimizer.load_state_dict(obj["optimizer"])

        print(self.model.state_dict()['module.fc.bias'])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def initialize_model(
    arch: str, lr: float, momentum: float, weight_decay: float, device_id: int
):
    print(f"=> creating model: {arch}, device_id: {device_id}")
    model = models.__dict__[arch](num_classes=10)
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id])
    print("after calling DDP")
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer = SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )

    ## initialize optimizer - TODO: this might need to change if non-deterministic/affects training!!
    opt_state = optimizer.state_dict()
    if len(opt_state['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
    optimizer.step()
    return model, criterion, optimizer


def initialize_data_loader(
    data_dir, batch_size, num_data_workers
) -> Tuple[DataLoader, DataLoader]:
    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    arch: str,
    model: DistributedDataParallel,
    optimizer,  # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.

    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(arch, model, optimizer)

    
    allcheckp = []
    for blob in client.list_blobs('torchelastic'):

        print(blob.name, blob.updated)
        allcheckp.append([blob.name, blob.updated])

    allcheckp.sort(key=itemgetter(1), reverse=True)
    print(allcheckp)

    for chk in allcheckp:
        try:
            print("Try to download from object ", chk[0])
            # download
            blob = bucket.blob(chk[0])
            buf = io.BytesIO()
            blob.download_to_file(buf)

            # load
            buf.seek(0)
            state.load(buf, device_id)

            break
        except (RuntimeError, TypeError):
            print("[WARNING!] Checkpoint with name ", chk[0], " not valid. Retrying with older checkpoint!")
    

    print(f"=> done restoring from previous checkpoint")
    return state


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def save_checkpoint(additional_snapshot, chk, active_snapshot, lock, epoch, it, change, profile_snap, prof_snap=False, prof_all=False, sync=False):
    
    start = time.time()
    additional_snapshot['epoch'] = epoch
    additional_snapshot['iter'] = it

    if sync:
        chk.checkpoint()
    else:
        if chk.chk_process is not None:
            while change.value==1:		
                # this means a checkpoint is on progress (wait for process doing the checkpoint to set variable to 0)
                continue

		# Once complete, initiate the next checkpoint

        if prof_snap:
            with lock:
                profile_snap.value = 1
        else:
            with lock:
                profile_snap.value = 0

        with lock:
                change.value = 1

        if not chk.spawned:
            print("------------- START A NEW PROCESS!! ------------")
            keywords = { \
					'background': True,
                                        'profile_snap': profile_snap}
            chk.chk_process = \
					Process(target=chk.checkpoint,	\
						args=[active_snapshot, lock, change, additional_snapshot], kwargs=keywords)
            
            chk.chk_process.start()
            chk.spawned = True
            print("-------- Background checkpoint process with pid: ", chk.chk_process.pid, " started!")

        # wait for the checkpoint/snapshot to complete if needed
        if sync or prof_snap or prof_all:
            while change.value==1:		
                continue

    
    end = time.time()
    print("store checkp took: ", time.time() - start)  

def train(
    train_loader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    optimizer,  # SGD,
    epoch: int,
    start_iter: int,
    device_id: int,
    print_freq: int,
    ch_freq,
    additional_snapshot,
    chk,
    active_snapshot,
    lock,
    change,
    global_rank,
    profile, prof_steps, max_overhead, profile_snap
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    # for CheckFreq-based profiling
    profile_done = False
    iter_times = []
    steps = 0
    steps_since_checkp = 0
    base_iter_time = 0
    skip_iter = False
    monitor=False
    monitor_steps = 0

    end = time.time()
    print("---- Start epoch: ", epoch, " from iteration: ", start_iter)

    train_iter = enumerate(train_loader)

    for _ in range(start_iter):
        next(train_iter)

    for i, (images, target) in train_iter:
        # measure data loading time
        data_time.update(time.time() - end)

        start = time.time()
        
        images = images.cuda(device_id, non_blocking=True)
        target = target.cuda(device_id, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient
        loss.backward()

        # checkfreq - wait for current snapshot to finish
        while active_snapshot.value == 1:
            continue

        # now compute sgd
        optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        skeys = list(optimizer.state_dict()['state'].keys())
        k = skeys[-1]

        if i % print_freq == 0:
            progress.display(i)

        if global_rank == 0:
            
            if profile and epoch==0 and (not profile_done) and i >= 5:
                print("---------- Profile step: ", i)
                iter_times.append(end-start)
                steps+=1
                if steps == prof_steps:
                    # profile
                    base_iter_time, ch_freq = do_profile(iter_times, additional_snapshot, chk, active_snapshot, lock, epoch, i, change, profile_snap)
                    profile_done = True
                    steps_since_checkp = 0
                    iter_times = []
                    skip_iter = True
                    monitor_steps = 0

            elif (profile_done or epoch > 0) and ch_freq>0 and steps_since_checkp==ch_freq:
                print("Checkpoint, at epoch: ", epoch, ", and iteration: ", i)
                save_checkpoint(additional_snapshot, chk, active_snapshot, lock, epoch, i, change, profile_snap)
                steps_since_checkp = 1

            elif (profile_done or epoch > 0) and ch_freq>0:
                steps_since_checkp += 1

            if profile and profile_done and epoch==0:
                if not monitor:
                    monitor= True
                if monitor and not skip_iter:
                    iter_times.append(end-start)
                    monitor_steps += 1
                    #print("Monitor steps is: ", monitor_steps)
                    if monitor_steps == ch_freq:
                        # adapt freq
                        ch_freq = adapt_chfreq(iter_times, base_iter_time, ch_freq, max_overhead)
                        iter_times = []
                        monitor_steps = 0
                if skip_iter:
                    skip_iter = False


def adapt_chfreq(iter_dur, base_iter_time, cfreq, max_overhead):

    cur_iter_mean = mean(iter_dur)
    cur_total = sum(iter_dur)
    old_total = base_iter_time * len(iter_dur)

    overhead_full = cur_total-old_total
    overhead_perc = 100 * overhead_full/old_total

    print("--------------- Iter mean new is: ", cur_iter_mean)
    print("--------------- Overhead is: ", overhead_perc)

    if overhead_perc > max_overhead:
        cfreq += 2
        print("-------------------------------- New Checkpoint Freq found: ", cfreq)

    return cfreq


def do_profile(iter_dur, additional_snapshot, chk, active_snapshot, lock, epoch, it, change, profile_snap):

    print(iter_dur)
    t_i = mean(iter_dur)

    ## first, do a simple checkpoint call to create the background processes
    save_checkpoint(additional_snapshot, chk, active_snapshot, lock, \
                        epoch, it, change, profile_snap, prof_snap=False, prof_all=True)

    ## now measure the time the snapshot takes
    start = time.time()
    save_checkpoint(additional_snapshot, chk, active_snapshot, lock, \
                        epoch, it, change, profile_snap, prof_snap=True, prof_all=False)
    overhead = time.time()-start

    ## finally, measure the time the actual checkpoint (snapshot + persist) takes
    start = time.time()
    save_checkpoint(additional_snapshot, chk, active_snapshot, lock, \
                        epoch, it, change, profile_snap, prof_snap=False, prof_all=True)
    t_f = time.time()-start

    ## Check Freq, to minimize stall at training
    chk_freq = max(math.ceil((t_f - overhead)/t_i), 1)
    print("t_i: ", t_i, " , overhead: ", overhead, " , t_f: ", t_f)
    print("------------ CheckFreq found: ", chk_freq)

    return t_i, chk_freq


def validate(
    val_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    device_id: int,
    print_freq: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if device_id is not None:
                images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg

def make_shm(obj):
    if obj is None:
        return
    if torch.is_tensor(obj):
        obj.share_memory_()
    elif isinstance(obj, dict):
        for name, ref in obj.items(): 
            make_shm(ref)
    elif isinstance(obj, list):
        for x in obj:
            make_shm(x)
    else:
        return

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch: int, lr: float) -> None:
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        print("------------ error!")
    pass
    main()



