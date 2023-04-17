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

########################################
# Environment parameters
########################################

# Define pre-processing SNR
SNRdB_range = np.arange(-10, 11)
SNR_range = 10**(SNRdB_range/10)

########################################
# Simulation parameters
########################################

# Define number of simulation setups
nsetups = 10000

# Define number of channel realizations
nchnlreal = 100

########################################
# Running simulation
########################################

# Simulation header
print('--------------------------------------------------')
now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('M-MIMO: BER vs SNR')
print('\t M = '+str(M))
print('\t K = '+str(K))
print('--------------------------------------------------')

# Prepare to save simulation results
ber_zf = np.zeros((SNR_range.size, nsetups, nchnlreal), dtype=np.double)
ber_sdk = np.zeros((SNR_range.size, nsetups, nchnlreal), dtype=np.double)
ber_bdk = np.zeros((SNR_range.size, nsetups, nchnlreal), dtype=np.double)
ber_sdk_relaxed = np.zeros((2, SNR_range.size, nsetups, nchnlreal), dtype=np.double)

# Obtain qam transmitted signals
tx_symbs, x_ = qam_transmitted_signals(K, nsetups)

# Go through all setups
for s in range(nsetups):

    print(f"setup: {s}/{nsetups-1}")

    timer_setup = time.time()

    # Generate communication setup
    H = massive_mimo(M, K, nchnlreal)

    # Go through all different SNR values
    for ss, SNR in enumerate(SNR_range):

        print(f"\tsnr: {ss}/{len(SNR_range)-1}")

        # Compute received signal
        y_ = received_signal(SNR, x_[s], H)

        # Perform ZF receiver
        xhat_soft_zf = zf_receiver(H, y_)

        # Perform standard distributed Kaczmarz receiver
        xhat_soft_sdk = standard_distributed_kaczmarz_receiver(H, y_, SNR, niter=1)
        xhat_soft_sdk_previous = standard_distributed_kaczmarz_receiver(H, y_, SNR, mu='previous', niter=1)
        xhat_soft_sdk_proposed = standard_distributed_kaczmarz_receiver(H, y_, SNR, mu='proposed', niter=1)

        # Perform Bayesian distributed Kaczmarz receiver
        xhat_soft_bdk = bayesian_distributed_kaczmarz_receiver(H, y_, SNR, niter=1)

        # Evaluate BER performance
        ber_zf[ss, s] = ber_evaluation(xhat_soft_zf, tx_symbs[s])

        ber_sdk[ss, s] = ber_evaluation(xhat_soft_sdk, tx_symbs[s])

        ber_sdk_relaxed[0, ss, s] = ber_evaluation(xhat_soft_sdk_previous, tx_symbs[s])
        ber_sdk_relaxed[1, ss, s] = ber_evaluation(xhat_soft_sdk_proposed, tx_symbs[s])

        ber_bdk[ss, s] = ber_evaluation(xhat_soft_bdk, tx_symbs[s])

    print('[setup] elapsed '+str(time.time()-timer_setup)+' seconds.\n')

now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('--------------------------------------------------')

np.savez('mmimo_ber_vs_snr.npz',
    M=M,
    K=K,
    SNRdB_range=SNRdB_range,
    ber_zf=ber_zf,
    ber_sdk=ber_sdk,
    ber_bdk=ber_bdk,
    ber_sdk_relaxed=ber_sdk_relaxed
    )

# Compute average values
ber_zf_avg = (ber_zf.mean(axis=-1)).mean(axis=-1)

ber_sdk_avg = (ber_sdk.mean(axis=-1)).mean(axis=-1)
ber_bdk_avg = (ber_bdk.mean(axis=-1)).mean(axis=-1)

ber_sdk_relaxed_avg = (ber_sdk_relaxed.mean(axis=-1)).mean(axis=-1)

########################################
# Plotting
########################################
fig, ax = plt.subplots()

ax.plot(SNRdB_range, ber_zf_avg, label='ZF: centralized', color='black', linewidth=2)

ax.plot(SNRdB_range, ber_sdk_avg, label='SDK [1]: $\lambda=1$, $T = 1$', linewidth=2, linestyle='dashed', color='black')
ax.plot(SNRdB_range, ber_bdk_avg, label=r'BDK: ${\lambda}^{\star}=1$, $T = 1$', linewidth=2, linestyle='dotted')

ax.plot(SNRdB_range, ber_sdk_relaxed_avg[0], label='SDK [1]: $\lambda=0.5\cdot{K}/{M}\cdot\log(4\cdot M \cdot \mathrm{SNR})$ in [1], $T = 1$', linewidth=2, linestyle='dashdot', color='black')
ax.plot(SNRdB_range, ber_sdk_relaxed_avg[1], label=r'SDK [1]: $\lambda$ in (13), $T = 1$', linewidth=2, linestyle=(0, (3, 1, 1, 1)))

ax.legend()

ax.set_xlabel('SNR [dB]')
ax.set_ylabel('average BER per UE')

ax.set_yscale('log')

plt.show()
