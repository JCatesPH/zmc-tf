import numpy as np
import math
import scipy
import scipy.special
import time

import tensorflow as tf

import ZMCintegral

N=3
mu = 0.1  # Fermi-level
hOmg = 0.3  # Photon energy eV
a = 4.  # AA
A = 4. # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.3  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.003  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2
V0 = eE0 * A / hOmg
V2 = A * (eE0 / hOmg) ** 2
cent = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)

# Example: Creating a tri-diagonal matrix with no for loops
# N = 3
# xx = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
# dd = np.diag(xx)
# x = np.eye(3, 3, k=-1) + np.eye(3, 3) * 2 + np.eye(3, 3, k=1) * 3

# tstexp = np.exp(x)
# inx = np.linalg.inv(x)

# res = tf.matmul(inx, x)
qy = 0


def ds(x):
    # kx , ky , qx , om
    # x0 , x1 , x2 , x3

    #tf.cast(x, tf.complex64)

    topkq = -V0 * (tf.complex(x[1],  (x[0] + x[2])))
    botkq = -V0 * (tf.complex(x[1], -(x[0] + x[2])))
    innkq = tf.complex((x[3] - A * (tf.square(x[0] + x[2]) + tf.square(x[1])) - V2), Gamm)

    topk = -V0 * (tf.complex(x[1],  x[0]))
    botk = -V0 * (tf.complex(x[1], -x[0]))
    innk = tf.complex((x[3] - A * (tf.square(x[0]) + tf.square(x[1])) - V2), Gamm)

    d = hOmg * tf.matrix_diag(cent)

    #Ginkq = np.eye(N, N, k=1) * topkq + np.eye(N, N, k=-1) * botkq + innkq * np.eye(N, N) - d
    #Gink = np.eye(N, N, k=1) * topk + np.eye(N, N, k=-1) * botk + innk * np.eye(N, N) - d
    
    # Identity
    z=tf.Variable(tf.eye(N,N,[len(x[0])],dtype=tf.complex64))

    # Lower Diag
    xz=tf.cast(tf.zeros_like(x[0]),dtype=tf.complex64)
    zdp=tf.Variable(tf.roll(z,1,1))
    zd=zdp[:,0,2].assign(xz)

    # Upper Diag
    ztp=tf.Variable(tf.roll(z,-1,1))
    zt=ztp[:,2,0].assign(xz)

    Ginkq=tf.transpose(tf.multiply(tf.transpose(z,perm=[2,1,0]),innkq),perm=[2,0,1]) \
        + tf.transpose(tf.multiply(tf.transpose(zd,perm=[2,1,0]),botkq),perm=[2,1,0]) \
        + tf.transpose(tf.multiply(tf.transpose(zt,perm=[2,1,0]),topkq),perm=[2,1,0]) \
        - d

    Gink=tf.transpose(tf.multiply(tf.transpose(z,perm=[2,1,0]),innk),perm=[2,0,1]) \
            + tf.transpose(tf.multiply(tf.transpose(zd,perm=[2,1,0]),botk),perm=[2,1,0]) \
            + tf.transpose(tf.multiply(tf.transpose(zt,perm=[2,1,0]),topk),perm=[2,1,0]) \
            - d

    #Ginkq = tf.matrix_diag(topkq, k=1) + tf.matrix_diag(botkq, k=-1) + tf.matrix_diag(innkq) - d
    #Gink = tf.matrix_diag(topk, k=1) + tf.matrix_diag(botk, k=-1) + tf.matrix_diag(innk) - d

    Grkq = tf.matrix_inverse(Ginkq)
    #Gakq = tf.matrix_transpose(Grkq, conjugate=True)

    Grk = tf.matrix_inverse(Gink)
    #Gak = tf.matrix_transpose(Grk, conjugate=True)

    #fer = np.heaviside(-(d + np.eye(N, N) * (om - mu)), 0)

    cond = tf.less(x[3] - mu, 0)
    muom = tf.where(cond, xz, x[3]-mu)

    fer = tf.transpose(tf.multiply(tf.transpose(z,perm=[2,1,0]),muom),perm=[2,1,0]) \
            - d

    in1 = tf.matmul(Grkq, tf.matmul(Grk, tf.matmul(fer, Grk, adjoint_b=True)))
    in2 = tf.matmul(Grkq, tf.matmul(fer, tf.matmul(Grkq, Grk, adjoint_a=True, adjoint_b=True)))
    tr = np.trace(in1 + in2)
    # HERE i will divide by DOS, multiply by 2 for spin, and divide by (2pi)^3

    dchi = -(4) * Gamm * tr / math.pi ** 2

    return dchi



myconfig = tf.ConfigProto(log_device_placement=True)
myconfig.gpu_options.allow_growth = True
session = tf.Session(config=myconfig)

MC = ZMCintegral.MCintegral(ds, [[0,1], [0,1], [0,1], [0,1]], available_GPU=[0])

start = time.time()
result = MC.evaluate()
end = time.time()

print('=====================================')
print('Result = ', result[0])
print(' Error = ', result[1])
print('=====================================')
print('Time = ', end-start, 'seconds')

