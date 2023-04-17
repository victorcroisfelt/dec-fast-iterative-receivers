import numpy as np

import multiprocessing
from joblib import Parallel
from joblib import dump, load
from joblib import delayed

# Obtain the number of processors
num_cores = multiprocessing.cpu_count()

########################################
# Private functions
########################################
def block_kaczmarz(H, y_, mu, niter, order=None):
    """"""

    # Extract dimensions from channel matrix
    M, K = H.shape

    # Canonical order
    canonical_order = np.arange(0, M, 1, dtype=np.int_)

    # Initializations
    A_ = np.eye(K, dtype=np.complex128)
    W_ = np.zeros((M, K), dtype=np.complex128)

    # Compute norms of the channel vectors w.r.t. antenna indexes
    norms = np.linalg.norm(H, axis=1)**2

    # Compute lambda vector for each antenna element
    mu_vec = mu/norms

    # Compute probability vector
    p_ = norms/norms.sum()

    # Randomization based on root indexes
    root_indexes = np.random.choice(M, size=niter, replace=True, p=p_)

    # Go through all iterations
    for iter in range(niter):

        # Check the order
        if order is None:
            morder = canonical_order
        else:
            morder = np.roll(canonical_order, shift=root_indexes[iter])

        # Go through each antenna (processing) element
        for m in morder:

            # Compute update term
            update_term = mu_vec[m] * A_ @ H[m]

            # Update the receive combining vector
            W_[m] += update_term

            # Update A_ matrix
            A_ -= np.outer(update_term, H[m].conj())

    # Compute soft estimate
    xhat_soft = W_.conj().T @ y_

    return xhat_soft

########################################
# Public functions
########################################
def mf_receiver(H, y_):
    """ Obtain soft-estimates for the matched filter (MF) receiver.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection o received signals.
        shape: (nchnlreal,M)

    Returns
    -------
    xhat_soft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)
    """
    Vbar = H / (np.linalg.norm(H, axis=1)**2)[:, None, :]
    xhat_soft = np.squeeze(np.matmul(Vbar.conj().transpose(0, 2, 1), y_[:, :, None]))

    return xhat_soft

def zf_receiver(H, y_):
    """ Obtain soft-estimates for the zero-forcing (ZF) receiver.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection o received signals.
        shape: (nchnlreal,M)

    Returns
    -------
    xhat_soft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)
    """
    from numpy.dual import inv

    # Compute Gramian matrix
    G = np.matmul(H.conj().transpose(0, 2, 1), H)

    # Compute MF soft-estimates
    mf = H.conj().transpose(0, 2, 1)@y_[:, :, None]

    # Compute inverses of the Gramian matrix
    Ginv = inv(G)

    # Compute ZF soft-estimates
    xhat_soft = np.squeeze(Ginv @ mf)

    return xhat_soft

def standard_distributed_kaczmarz_receiver(H, y_, SNR, mu=None, niter=1, D=None):
    """ Obtain soft-estimates for the standard distributed Kaczmarz (SDK) receiver.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection o received signals.
        shape: (nchnlreal,M)

    SNR : float
        Signal-to-noise ratio.

    mu : str
        Relaxation parameter.
            None        -   mu = 1
            'previous'  -   previous
            'proposed'  -   proposed

    Returns
    -------
    xhat_soft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)
    """

    # Extract dimensions from channel matrix
    nchnlreal, M, K = H.shape

    # Check mu input
    if mu is None:
        mu_cons = 1.0

    elif mu is 'previous':
        mu_cons = 0.5*(K/M)*np.log(4*M*SNR)

    # Prepare to store the soft estimates
    xhat_soft = np.zeros((nchnlreal, K), dtype=np.complex128)

    # Go through all channel realizations
    for n in range(nchnlreal):

        # Initialization
        x_ = np.zeros((2, K), dtype=np.complex128)

        # Compute channel norms for each node
        mu_norm = np.reciprocal(np.linalg.norm(H[n], axis=1)**2)

        user_indexes = np.ones(K, dtype=np.complex128)

        # Go through all iterations
        for iter in range(niter):

            # Go through each node
            for m in range(M):

                if mu is 'proposed':
                    num = K*SNR
                    den = (iter+1)*(m+1)

                    mu_cons = np.min([np.sqrt(num/den), 1])

                # New is old
                x_[0] = x_[1].copy()

                # Compute the residual
                residual = y_[n, m] - np.inner(H[n, m], x_[0])

                # Update the soft estimates
                x_[1] = x_[0] + mu_cons * mu_norm[m] * H[n, m].conj() * (y_[n, m] - np.inner(H[n, m], x_[0]))

        # Compute soft estimate
        xhat_soft[n] = x_[1]

    return xhat_soft

def bayesian_distributed_kaczmarz_receiver(H, y_, SNR, mu=None, niter=1):
    """ Obtain soft-estimates for the Bayesian distributed Kaczmarz (BDK) receiver.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection o received signals.
        shape: (nchnlreal,M)

    SNR : float
        Signal-to-noise ratio.

    mu : str
        Relaxation parameter.
            None        -   mu = 1
            'proposed'  -   proposed

    Returns
    -------
    xhat_soft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)
    """

    # Extract dimensions from channel matrix
    nchnlreal, M, K = H.shape

    # Check mu input
    if mu is None:
        mu_cons = 1.0

    elif mu is 'previous':
        mu_cons = 0.5*(K/M)*np.log(4*M*SNR)

    # Compute inverse of the SNR
    xi = 1/SNR

    # Store soft estimate
    xhat_soft = np.zeros((nchnlreal, K), dtype=np.complex128)

    # Go through each channel realization
    for n in range(nchnlreal):

        # Initializations
        x_ = np.zeros((2, K), dtype=np.complex128)
        n_ = np.zeros((2, M), dtype=np.complex128)

        # Compute lambda vector for each antenna element
        mu_norm = np.reciprocal(np.linalg.norm(H[n], axis=1)**2 + xi)

        # Go through all iterations
        for iter in range(niter):

            # Go through each antenna (processing) element
            for m in range(M):

                if mu is 'proposed':
                    mu_cons = np.sqrt((K*SNR+M)/((niter+1)*(m+1)))

                # New is old
                x_[0] = x_[1].copy()
                n_[0] = n_[1].copy()

                # Compute the residual
                residual = y_[n, m] - np.inner(H[n, m], x_[0]) - np.sqrt(xi)*n_[0, m]

                # Update soft estimate
                x_[1] = x_[0] + mu_cons * mu_norm[m] * H[n, m].conj() * residual

                # Update noise estimate
                n_[1, m] = n_[0, m] + mu_cons * mu_norm[m] * np.sqrt(xi) * residual

        # Get final soft estimates
        xhat_soft[n] = x_[1]

    return xhat_soft

# def distributed_kaczmarz_receiver(H, y_, SNR, mu=None, niter=1):
#     """ """
#
#     # Extract dimensions from channel matrix
#     nchnlreal, M, K = H.shape
#
#     # Check if we have to use the optimum value
#     if mu is None:
#         mu = 0.5*(K/M)*np.log(4*M*SNR)
#
#     with Parallel(n_jobs=num_cores) as parl:
#         xhat_soft_raw = parl(delayed(block_kaczmarz)(H[n], y_[n], mu, niter) for n in range(nchnlreal))
#
#     # Store soft estimate
#     xhat_soft = np.array(xhat_soft_raw)
#
#     return xhat_soft

# def randomized_kaczmarz_receiver(H, y_, SNR, mu=None, niter=1):
#     """ """
#
#     # Extract dimensions from channel matrix
#     nchnlreal, M, K = H.shape
#
#     # Check if we have to use the optimum value
#     if mu is None:
#         mu = 0.5*(K/M)*np.log(4*M*SNR)
#
#     with Parallel(n_jobs=num_cores) as parl:
#         xhat_soft_raw = parl(delayed(block_kaczmarz)(H[n], y_[n], mu, niter, order=1) for n in range(nchnlreal))
#
#     # Store soft estimate
#     xhat_soft = np.array(xhat_soft_raw)
#
#     return xhat_soft

# def rzf_receiver(SNR, H, G, y_):
#     """ Obtain regularized zero-forcing (RZF) soft signal estimates. Raw signal
#     estimates are outputted for comparison with methods that emulate the RZF
#     scheme. Soft normalization matrix Dinv is also outputted.
#
#     Parameters
#     ----------
#     SNR : float
#         Signal-to-noise-ratio in power units.
#
#     H : 3D ndarray of numpy.cdouble
#         Collection of channel matrices.
#         shape: (nchnlreal,M,K)
#
#     G : 3D ndarray of numpy.cdouble
#         Collection of channel Gramian matrices.
#         shape: (nchnlreal,K,K)
#
#     y_ : 2D ndarray of numpy.cdouble
#         Collection o received signals.
#         shape: (nchnlreal,M)
#
#     Returns
#     -------
#     xhat_soft : 1D ndarray of numpy.cdouble
#         Soft signal estimates.
#         shape: (nchnlreal,K)
#
#     xhat : 1D ndarray of numpy.cdouble
#         Raw signal estimates.
#         shape: (nchnlreal,K)
#
#     Dinv : 2D ndarray of numpy.cdouble
#         Soft power normalization.
#         shape: (nchnlreal,K)
#     """
#     from numpy.dual import inv
#
#     nchnlreal, M, K = H.shape
#
#     # Constants
#     xi = 1/SNR
#     eyeK = np.eye(K)
#
#     # Store inverted covariance of the received signal
#     Ryy_inv = inv(G + (xi*eyeK)[None, :, :])
#
#     # Compute receive combining matrices
#     V = np.matmul(H, Ryy_inv)
#
#     # Store norm of RZF receive combining
#     D = np.diagonal(np.matmul(Ryy_inv, G), axis1=1, axis2=2)
#     Dinv = np.reciprocal(D)
#
#     # Get RZF signal estimates
#     xhat = np.squeeze(np.matmul(V.conj().transpose(0, 2, 1), y_[:, :, None]))
#
#     # Get soft RZF signal estimates
#     xhat_soft = Dinv*xhat
#
#     return xhat_soft, xhat, Dinv
