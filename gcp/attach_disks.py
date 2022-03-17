import os
import subprocess
import time

zone = "europe-west4-a"
num_workers=4
cmd = "" # command to connect to the cluster

def process_response(res, keyword):

    ret_list = []
    reslines = res.split("\n")[1:]
    
    for r in reslines:
        info = r.split(" ")
        name = info[0]
        if (keyword in name):
            ret_list.append(name)

    return ret_list


def check_VM_running(name, zone):
    command = "gcloud compute instances describe "  + name + " --zone " + zone
    result = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode()
    if "RUNNING" in result:
        return True
    return False

def attach_disk(vm_name, disk):
    command = "gcloud compute instances attach-disk " + vm_name + " --disk " + disk
    while True:
        res = os.system(command)
        if res==0:
            break

def mount_disk(name):
    cmd1 = "'sudo mkdir /mnt/data'"
    cmd2 = "'sudo mount /dev/sdb1 /mnt/data'"

    command = "gcloud compute ssh " + name + " --command " + cmd1
    while True:
        a = os.system(command)
        if a==0:
            break

    command = "gcloud compute ssh " + name + " --command " + cmd2
    while True:
        a = os.system(command)
        if a==0:
            break

# get list of VMs

command = "gcloud compute instances list" 
cmd = command.split()
result = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode()
instance_list = process_response(result, "spot-gpu")
print(instance_list)


# get list of disks
command = "gcloud compute disks list" 
cmd = command.split()
result = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode()
disk_list = process_response(result, "disk-imagenet")
print(disk_list)

assert len(instance_list)==len(disk_list)

disk_vm_map = {}
for i in range(len(instance_list)):
    disk_vm_map[instance_list[i]] = disk_list[i]

print(disk_vm_map)


# attach + mount
for key, ref in disk_vm_map.items():
    attach_disk(key, ref)
    mount_disk(key)

# monitor + reattach if needed

deleted = set()

while True:
    for vm in instance_list:
        status = check_VM_running(vm, zone)
        if (not status):
            deleted.add(vm)

    to_remove=[]
    for vm in deleted:
        status = check_VM_running(vm, zone)
        if status:
            print("reattach disk in ", vm)
            to_remove.append(vm)
            attach_disk(vm, disk_vm_map[vm])
            mount_disk(vm)

    for v in to_remove:
        deleted.remove(v)

    time.sleep(2)
            


