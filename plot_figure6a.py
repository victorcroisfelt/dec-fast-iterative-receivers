import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)

# Struggling with the font size of the elements
axis_font = {'size':'12'}

plt.rcParams.update({'font.size': 12})

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

plt.rcParams["font.family"] = "Times New Roman"

########################################
# Loading data
########################################

# Load data
data = np.load('../data/mmimo_ber_vs_snr.npz')

M = data['M']
K = data['K']

SNRdB_range = data['SNRdB_range']

ber_zf = data['ber_zf']
ber_sdk = data['ber_sdk']
ber_bdk = data['ber_bdk']
ber_sdk_relaxed = data['ber_sdk_relaxed']

# # set tick width
# matplotlib.rcParams['xtick.major.size'] = 10
# matplotlib.rcParams['xtick.major.width'] = 2
# matplotlib.rcParams['xtick.minor.size'] = 5
# matplotlib.rcParams['xtick.minor.width'] = 2

# matplotlib.rcParams['ytick.major.size'] = 10
# matplotlib.rcParams['ytick.major.width'] = 2
# matplotlib.rcParams['ytick.minor.size'] = 5
# matplotlib.rcParams['ytick.minor.width'] = 2

########################################
# Loading data
########################################

# Compute average values
ber_zf_avg = (ber_zf.mean(axis=-1)).mean(axis=-1)

ber_sdk_avg = (ber_sdk.mean(axis=-1)).mean(axis=-1)
ber_bdk_avg = (ber_bdk.mean(axis=-1)).mean(axis=-1)

ber_sdk_relaxed_avg = (ber_sdk_relaxed.mean(axis=-1)).mean(axis=-1)

########################################
# Plotting
########################################
fig, ax = plt.subplots(figsize=(3.5, 2.0))

ax.plot(SNRdB_range, ber_zf_avg, label='ZF: centralized', color='black', linewidth=1.5)

ax.plot(SNRdB_range, ber_sdk_avg, label='SDK [1]: $\lambda=1$', linewidth=1.5, linestyle='dashed', color='black')
ax.plot(SNRdB_range, ber_bdk_avg, label=r'BDK: ${\lambda}^{\star}=1$', linewidth=1.5, linestyle='dotted')

ax.plot(SNRdB_range, ber_sdk_relaxed_avg[0], label='SDK: $\lambda$ in Eq. (22) of [1]', linewidth=1.5, linestyle='dashdot', color='black')
ax.plot(SNRdB_range, ber_sdk_relaxed_avg[1], label=r'SDK [1]: proposed $\lambda$ in Eq. (13)', linewidth=1.5, linestyle=(0, (3, 1, 1, 1)))

ax.set_xlabel('SNR [dB]')
ax.set_ylabel('average BER per UE')

ax.set_yscale('log')

ax.legend(fontsize='x-small', framealpha=0.5)
ax.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.show()
