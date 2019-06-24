from tensorflow.python.client import device_lib
from multiprocessing import Process, Queue

def get_available_gpus_child_process(gpus_list_queue):
    local_device_protos = device_lib.list_local_devices()
    gpus_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus_list_queue.put(gpus_list)


def get_available_gpus():
    gpus_list_queue = Queue()
    proc_get_gpus = Process(target=get_available_gpus_child_process, args=(gpus_list_queue,))
    proc_get_gpus.start()
    proc_get_gpus.join()
    gpus_list = gpus_list_queue.get()
    return gpus_list
