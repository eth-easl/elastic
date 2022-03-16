import os
import subprocess

zone = "europe-west4-a"
num_workers=4
cmd = "" # command to connect to the cluster

def check_VM_running(name, zone):
    command = "gcloud compute instances describe "  + name + " --zone " + zone
    result = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode()
    if "RUNNING" in result:
        return True
    return False

def delete_VM(name, zone):
    command = "gcloud compute instances delete " + name + " --zone " + zone + " --quiet"
    print("delete command is: ", command)
    result = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode()
    print(result)
    if "Deleted" in result:
        return True
    return False

def start_and_join(name):
    command = "gcloud compute instances create " + name + " --zone=europe-west4-a \
        --machine-type=n1-standard-8 \
        --image=imagenet-drivers \
        --image-project=ml-elasticity \
        --boot-disk-size=300GB \
        --boot-disk-type=pd-ssd \
        --accelerator=count=2,type=nvidia-tesla-v100 \
        --preemptible"
    os.system(command)
    print("VM ", name, " created")
    command = "gcloud compute ssh " + name + " --command " + cmd
    while True:
        a = os.system(command)
        if a==0:
            break


vm_list = ['worker0', 'worker1', 'worker2', 'worker3']

'''
for i in range(1,num_workers):
    name = "worker" + str(i)
    start_and_join(name)
    vm_list.append(name)
'''
# monitor the VMs

while True:
    for vm in vm_list:
        status = check_VM_running(vm, zone)
        if (not status):
            delete_VM(vm, zone)
            start_and_join(vm)



