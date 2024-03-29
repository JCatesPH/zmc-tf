#%% [markdown]
# # Tensorflow ZMCintegral Tests
# 
# Testing of the tensorflow version of ZMCintegral.

import math
import time
import numpy as np
import tensorflow as tf
import ZMCintegralmod as ZMCintegral

#from tensorflow_on_slurm import tf_config_from_slurm

#cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)
#cluster_spec = tf.train.ClusterSpec(cluster)
#server = tf.train.Server(server_or_cluster_def=cluster_spec,
#                         job_name=my_job_name,
#                         task_index=my_task_index)

myconfig = tf.ConfigProto()
myconfig.gpu_options.allow_growth = True
session = tf.Session(config=myconfig)

def testfoo(x):
    vec = tf.Variable([x[0],x[1],x[2],x[3]], tf.float32)
    iden = tf.eye(4)
    y = tf.linalg.matmul(iden, vec)
#    tran = tf.transpose(y)
#    inv = tf.linalg.inv(y)
    z = tf.linalg.det(y)
    return z

MC = ZMCintegral.MCintegral(testfoo, [[0,2], [1,4], [2,3], [4,6]], available_GPU=[0])

start = time.time()
result = MC.evaluate()
end = time.time()

print('=====================================')
print('Result = ', result[0])
print(' Error = ', result[1])
print('=====================================')
print('Time = ', end-start, 'seconds')

