import numpy as np
import time
import functools

########################################
# Private Functions
########################################
def dec2bitarray(in_number, bit_width):
    """
    Convert a positive integer or an array-like of positive integers to NumPy
    array of the specified size containing bits (0 and 1).

    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width).copy()
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width).copy()
    return result

@functools.lru_cache(maxsize=128, typed=False)
def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only work for one integer.

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """
    result = np.zeros(bit_width, np.int8)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result

########################################
# Public Functions
########################################
def qam_transmitted_signals(K, nsetups):
    """ Generate and modulate user transmitted signals by using 16-QAM.

    Parameters
    ----------
    K : int
        Number of users.

    nsetups : int
        Number of different communication setups.

    Returns
    -------
    tx_symbs : 2D ndarray of np.uint
        Integer generated symbols.
        shape: (nsetups,K)

    tx_basedband_symbs : 2D ndarray of np.cdouble
        Complex-modulated symbols according to constellation.
        shape: (nsetups,K)
    """
    # Define 16-qam constellation vector
    constellation = np.array([-3+1j*3, -3+1j*1, -3-1j*1, -3-1j*3,
                              -1+1j*3, -1+1j*1, -1-1j*1, -1-1j*3,
                              +1+1j*3, +1+1j*1, +1-1j*1, +1-1j*3,
                              +3+1j*3, +3+1j*1, +3-1j*1, +3-1j*3], dtype=np.cdouble)

    # Normalize constellation with respect to average constellation power
    constellation *= np.sqrt(1/10)

    # Modulation order
    m = constellation.shape[0]

    # Generate random transmitted symbols for each user
    tx_symbs = np.random.randint(low=0, high=m, size=(nsetups, K))

    # Perform m-qam modulation
    mapfunc = np.vectorize(lambda i: constellation[i])
    tx_baseband_symbs = (mapfunc(tx_symbs.flatten())).reshape(tx_symbs.shape)

    return tx_symbs, tx_baseband_symbs

def received_signal(SNR, x_, H):
    """Generate base station received signal for each channel realization and
    SNR point.

    Parameters
    ----------
    SNR : float or 1D ndarray of np.double
        Signal-to-noise-ratio values in power units.
        shape: (,) or (len(SNR),)

    x_ : 1D ndarray of numpy.cdouble
        Baseband signals.
        shape: (K,)

    H : 3D ndarray of numpy.cdouble
        Collection of nchnlreal channel matrices.
        shape: (nchnlreal,M,K)

    Returns
    -------
    ySNR : 2D or 3D ndarray of numpy.cdouble
        Collection of received signals.
        shape: (nchnlreal,M) or (lenght(SNR),nchnlreal,M)

    Notes
    -----
    Function considers white complex Gaussian noise.
    """
    nchnlreal, M, K = H.shape

    if isinstance(SNR, float):
        SNR = np.array([SNR])

    # Generate white complex-Gaussian noise
    wcgn = np.sqrt(.5)*(np.random.randn(nchnlreal, M) + 1j*np.random.randn(nchnlreal, M))

    # Received signals w/o noise
    no_noise_rx_signal = (H*x_[None, None, :]).sum(axis=-1)

    # SNR's reciprocals
    rec_SNR = np.reciprocal(np.sqrt(SNR))

    # Received signal vector
    ySNR = no_noise_rx_signal[None, :, :] + rec_SNR[:, None, None]*wcgn

    # Get rid of additional dimensions when SNR is a float
    ySNR = np.squeeze(ySNR)

    return ySNR

def qam_received_signals(xsoft):
    """ Perform m-QAM demodulation based on the hard threshold detector.
    Return nearest demodulated symbols.

    Parameters
    ----------
    xsoft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)

    Returns
    -------
    rx_symbs : 2D ndarray of np.uint
        Demodulated symbols.
        shape: (nchnlreal,K)
    """
    # Define 16-qam constellation vector
    constellation = np.array([-3+1j*3, -3+1j*1, -3-1j*1, -3-1j*3,
                              -1+1j*3, -1+1j*1, -1-1j*1, -1-1j*3,
                              +1+1j*3, +1+1j*1, +1-1j*1, +1-1j*3,
                              +3+1j*3, +3+1j*1, +3-1j*1, +3-1j*3], dtype=np.cdouble)

    # Normalize constellation with respect to average constellation power
    constellation *= np.sqrt(1/10)

    # Perform hard-threshold demodulation
    rx_symbs = np.abs(xsoft[:, :, None] - constellation).argmin(-1)

    return rx_symbs

def ber_evaluation(xsoft, tx_symbs):
    """ Count the bit error rate (BER) per user.

    Parameters
    ----------
    xsoft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)

    tx_symbs : 1D ndarray of np.uint
        True integer generated symbols.
        shape: (K,)

    Returns
    -------
    ber_peruser : 1D ndarray of np.cdouble
        BER per user.
        shape: (nchnlreal,)
    """
    nchnlreal, K = xsoft.shape
    num_bits_symb = 4

    rx_symbs = qam_received_signals(xsoft)

    tx_bits = dec2bitarray(tx_symbs, num_bits_symb)
    rx_bits = dec2bitarray(rx_symbs.ravel(), num_bits_symb)

    hamming_distance = np.bitwise_xor(np.tile(tx_bits, nchnlreal), rx_bits).reshape(nchnlreal, num_bits_symb*K).sum(axis=-1)

    ber_peruser = hamming_distance / (num_bits_symb*K)

    return ber_peruser
