import numpy as np

########################################
# Private functions
########################################
def nonstationary(M, K, D):
    """ Output the diagonal matrix regarding non-stationary effects.

    Parameters
    ----------
    M : int
        Number of antennas.

    K : int
        Number of users.

    D : int
        Number of visible antennas per user.

    Returns
    -------
    diag : 3D ndarray of numpy.cdouble
        Collection of diagonal matrices.
        shape: (K,M,M)
    """

    # Non-stationary
    diag = np.zeros((M, K), dtype=np.cdouble)

    if D != M:

        # Generating VRs
        offset = D//2
        centers = np.random.randint(M, size=K, dtype=np.int)

        if D%2 == 0:
            upper = centers + offset
        else:
            upper = centers + (offset + 1)

        lower = centers - offset

        # Check correctness of the sets
        upper[upper >= M] = M-1
        lower[lower < 0] = 0

        # Generating diagonal matrices
        zerosM = np.zeros(M, dtype=np.cdouble)
        for k in range(K):
            els = zerosM.copy()
            els[lower[k]:upper[k]] = 1

            diag[:, k] = els

        # Normalization
        diag *= M/D

    else:

        diag = np.ones((M, K))

    return diag

########################################
# Public Functions
########################################
def massive_mimo(M, K, nchnlreal):
    """ Generate a channel matrix considering equal power control, that is,
    we disregard pathloss and other wireless channel effects.

    Parameters
    ----------
    M : int
        Number of base station antennas.

    K : int
        Number of users.

    nchnlreal : int
        Number of channel realizations.

    Returns
    -------
    H : 3D ndarray of numpy.cdouble
        Collection of nchnlreal channel matrices.
        shape: (nchnlreal,M,K)
    """

    # Generate uncorrelated channels
    H = np.sqrt(1/2)*(np.random.randn(nchnlreal, M, K) + 1j*np.random.randn(nchnlreal, M, K))

    return H

def extra_large_mimo(M, K, D, nchnlreal):
    """ Generate a channel matrix considering equal power control (no pathloss)
    and non-stationarities channels.

    Parameters
    ----------
    M : int
        Number of base station antennas.

    K : int
        Number of users.

    D : int
        Number of effective users  served by each antenna.

    nchnlreal : int
        Number of channel realizations.

    Returns
    -------
    H : 3D ndarray of numpy.cdouble
        Collection of nchnlreal channel matrices.
        shape: (nchnlreal,M,K)
    """

    # Prepare to save the diagonals of the Dm matrix for each antenna m
    diags = np.zeros((M, K), dtype=np.cdouble)

    # Go through all the antenna elements
    for m in range(M):

        # Define a control variable
        control = False

        while not(control):

            # Generate a random sequence of ones
            sequence = np.random.randint(0, 2, size=K)

            # Check the sum of the sequences
            if sequence.sum() == D:

                # Store the sequence obtained
                diags[m] = sequence.copy()

                # Update the control variable
                control = True

    # Generate uncorrelated channels
    Huncorr = np.sqrt((1/2))*(np.random.randn(nchnlreal, M, K) + 1j*np.random.randn(nchnlreal, M, K))

    # Go through all the antenna elements
    for m in range(M):

        # Apply the diagonal matrix structure
        Huncorr[:, m, :] = diags[m] * Huncorr[:, m, :]

    return Huncorr
