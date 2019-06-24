#%%
import math
import time

import numpy as np
import scipy
import scipy.special
import tensorflow as tf

start = time.time()
mu = 0.1  # Fermi-level
hOmg = 0.3  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.3  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.003  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2
V0 = eE0 * A / hOmg
V2 = A * (eE0 / hOmg) ** 2
cent = tf.Variable([-1., 0., 1.])


def ds(x):
    #   kx   ky   qx   qy   om
    # x[0] x[1] x[2] x[3] x[4]

    topkq = -complex(0, 1) * V0 * ((x[0] + x[2]) - complex(0, 1) * (x[1] + x[3]))
    botkq = complex(0, 1) * V0 * ((x[0] + x[2]) + complex(0, 1) * (x[1] + x[3]))
    innkq = x[4] + complex(0, 1) * Gamm - A * ((x[0] + x[2]) ** 2 + (x[1] + x[3]) ** 2) - V2

    topk = -complex(0, 1) * V0 * (x[0] - complex(0, 1) * x[1])
    botk = complex(0, 1) * V0 * (x[0] + complex(0, 1) * x[1])
    innk = x[4] + complex(0, 1) * Gamm - A * (x[0] ** 2 + x[1] ** 2) - V2


    # cent = tf.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)

    d = hOmg * tf.matrix_diag(cent)

    Ginkq = tf.matrix_diag(topkq, k=1) + tf.matrix_diag(botkq, k=1) + tf.matrix_diag(innkq, k=0) - d
    Gink = tf.matrix_diag(topk, k=1) + tf.matrix_diag(botk, k=1) + tf.matrix_diag(innk, k=0) - d

    Grkq = tf.linalg.inv(Ginkq)
    Gakq = tf.transpose(tf.conj(Grkq))

    Grk = tf.linalg.inv(Gink)
    Gak = tf.transpose(tf.conj(Grk))

    fer = tf.heaviside(-(d + tf.eye(N) * (x[4] - mu)), 0)

    in1 = tf.matmul(Grkq, tf.matmul(Grk, tf.matmul(fer, Gak)))
    in2 = tf.matmul(Grkq, tf.matmul(fer, tf.matmul(Gakq, Gak)))
    tr = tf.trace(in1 + in2)
    # HERE i will divide by DOS, multiply by 2 for spin, and divide by (2pi)^3

    dchi = -(4) * Gamm * tr / math.pi ** 2

    return dchi

    
test = ds([0.1, 0.1, 0.01, 0, 0.09])
end = time.time()
print('Time:',end - start)
print(test)


#%%
