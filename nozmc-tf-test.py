import numpy as np
import time
import math
import tensorflow as tf

a = tf.constant([1.0, -2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

c = tf.matmul(a,b)

myconfig=tf.ConfigProto(log_device_placement=True)
myconfig.gpu_options.allow_growth=True

sess = tf.Session(config=myconfig)

print(sess.run(c))

print('\na = ', a)
print('\nb = ', b)

print('\nc = ', c)
