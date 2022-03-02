from kubernetes import client, config, watch
import time
import os

# Configs can be set in Configuration class directly or using helper utility
config.load_kube_config()

v1 = client.CoreV1Api()
ns = 'elastic-job'

while True:
    
    print("Listing pods with their IPs in namespace: ", ns)
    ret = v1.list_pod_for_all_namespaces(watch=False)
    retns = [x for x in ret.items if x.metadata.namespace == ns]

    for i in retns:
        #print(i.status)
        print("%s\t%s\t%s\t%s" % (i.metadata.name, i.status.pod_ip, i.status.phase, i.status.host_ip))
        
        if (('worker' in i.metadata.name) and (i.status.phase == 'Failed')):

            if (i.status.init_container_statuses): # TODO: check if this condition is enough
                # get nvidia-drivers
                ret = v1.list_pod_for_all_namespaces(watch=False)
                retnvidia = [x for x in ret.items if (x.metadata.namespace == 'kube-system' and 'nvidia' in x.metadata.name and x.status.host_ip == i.status.host_ip)]
                for x in retnvidia:
                    # restart driver
                    if (x.status.phase == 'Terminated'):

                        print('Manually restart driver pod: ', x.metadata.name)
                        os.system('kubectl delete pods ' + x.metadata.name + ' -n kube-system')
                        time.sleep(10)
        
                # restart pod
                print('Pod: ', i.metadata.name, ' failed, restart!')
                os.system('kubectl delete pods ' + i.metadata.name + ' -n ' + ns)
    
    time.sleep(10)
    print("----------------------------------------------------------\n")
