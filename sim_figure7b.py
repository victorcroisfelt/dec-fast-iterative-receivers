import numpy as np
import time
from datetime import datetime

import multiprocessing
from joblib import Parallel
from joblib import dump, load

from comm import *
from commsetup import *
from receivers import *

import matplotlib.pyplot as plt

########################################
# Preamble
########################################

# Obtain the number of processors
num_cores = multiprocessing.cpu_count()

# Random seed
np.random.seed(42)

# Treating errors in numpy
np.seterr(divide='raise', invalid='raise')

########################################
# System parameters
########################################

# Number of antennas
M = 128

# Number of users
K = 16

# Number of effective users
D = 4

########################################
# Environment parameters
########################################

# Define pre-processing SNR
SNRdB = 0
SNR = 10**(SNRdB/10)

########################################
# Simulation parameters
########################################

# Define number of simulation setups
nsetups = 10

# Define number of channel realizations
nchnlreal = 50

# Number of iterations
niter_range = np.arange(1, 9)

########################################
# Running simulation
########################################

# Simulation header
print('--------------------------------------------------')
now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('M-MIMO: BER vs Drange')
print('\t M = '+str(M))
print('\t K = '+str(K))
print('--------------------------------------------------')

# Prepare to save simulation results
ber_zf = np.zeros((nsetups, nchnlreal), dtype=np.double)

ber_sdk = np.zeros((niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_bdk = np.zeros((niter_range.size, nsetups, nchnlreal), dtype=np.double)

ber_sdk_relaxed = np.zeros((2, niter_range.size, nsetups, nchnlreal), dtype=np.double)

# Obtain qam transmitted signals
tx_symbs, x_ = qam_transmitted_signals(K, nsetups)

# Go through all setups
for s in range(nsetups):

    print(f"setup: {s}/{nsetups-1}")

    timer_setup = time.time()

    # Generate communication setup
    H = extra_large_mimo(M, K, D, nchnlreal)

    # Compute received signal
    y_ = received_signal(SNR, x_[s], H)

    # Perform ZF receiver
    xhat_soft_zf = zf_receiver(H, y_)

    ber_zf[s] = ber_evaluation(xhat_soft_zf, tx_symbs[s])

    # Go through all different iteration values
    for it, niter in enumerate(niter_range):

        print(f"\tniter: {it}/{len(niter_range)-1}")

        # Perform standard distributed Kaczmarz receiver
        xhat_soft_sdk = standard_distributed_kaczmarz_receiver(H, y_, SNR, niter=niter, D=D)
        xhat_soft_sdk_previous = standard_distributed_kaczmarz_receiver(H, y_, SNR, mu='previous', niter=niter, D=D)
        xhat_soft_sdk_proposed = standard_distributed_kaczmarz_receiver(H, y_, SNR, mu='proposed', niter=niter, D=D)

        # Perform Bayesian distributed Kaczmarz receiver
        xhat_soft_bdk = bayesian_distributed_kaczmarz_receiver(H, y_, SNR, niter=niter)

        # Evaluate BER performance
        ber_sdk[it, s] = ber_evaluation(xhat_soft_sdk, tx_symbs[s])
        ber_sdk_relaxed[0, it, s] = ber_evaluation(xhat_soft_sdk_previous, tx_symbs[s])
        ber_sdk_relaxed[1, it, s] = ber_evaluation(xhat_soft_sdk_proposed, tx_symbs[s])
        ber_bdk[it, s] = ber_evaluation(xhat_soft_bdk, tx_symbs[s])

    print('[setup] elapsed '+str(time.time()-timer_setup)+' seconds.\n')

now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('--------------------------------------------------')

np.savez('xlmimo_ber_vs_cycles.npz',
    M=M,
    K=K,
    niter_range=niter_range,
    ber_zf=ber_zf,
    ber_sdk=ber_sdk,
    ber_bdk=ber_bdk,
    ber_sdk_relaxed=ber_sdk_relaxed
    )

# Compute average values
ber_zf_avg = (ber_zf.mean(axis=-1)).mean(axis=-1)

ber_sdk_avg = (ber_sdk.mean(axis=-1)).mean(axis=-1)
ber_sdk_relaxed_avg = (ber_sdk_relaxed.mean(axis=-1)).mean(axis=-1)
ber_bdk_avg = (ber_bdk.mean(axis=-1)).mean(axis=-1)

########################################
# Plotting
########################################
fig, ax = plt.subplots()

ax.plot(niter_range, ber_zf_avg*np.ones(niter_range.shape), label='ZF: centralized', color='black', linewidth=2)

ax.plot(niter_range, ber_sdk_avg, label='SDK [1]: $\lambda=1$', linewidth=2, linestyle='dashed', color='black')
ax.plot(niter_range, ber_bdk_avg, label=r'BDK: ${\lambda}^{\star}=1$', linewidth=2, linestyle='dotted')

ax.plot(niter_range, ber_sdk_relaxed_avg[0], label='SDK [1]: $\lambda=0.5\cdot{K}/{M}\cdot\log(4\cdot M \cdot \mathrm{SNR})$ in [1]', linewidth=2, linestyle='dashdot', color='black')
ax.plot(niter_range, ber_sdk_relaxed_avg[1], label=r'SDK [1]: $\lambda$ in (13)', linewidth=2, linestyle=(0, (3, 1, 1, 1)))

ax.legend()

ax.set_xlabel('number of cycles $T$')
ax.set_ylabel('average BER per UE')

ax.set_yscale('log')

plt.show()
