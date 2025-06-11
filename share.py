import numpy as np
from utils import *


def quant(z,Lz):
    """
    Quantizes z using a scale factor Lz

    Parameters
    ----------
    z : int
        Input value
    Lz : float
        Scale factor

    Returns
    -------
    int
        lceil z/Lz rfloor
    """
    return np.round(z/Lz).astype(int)

def share(n, q):
    """
    Creat n-shares of zero in Z_q

    Parameters
    ----------
    n : int
        Number of shares
    q : int
        Modulus

    Returns
    -------
    nparray of n-shares
        (s_1,...,s_n)
    """

    s = np.array([get_rand((-q+1)//2, q//2) for _ in range(n - 1)])
    s = np.append(s, mod(-s.sum(),q))
    return s


def generate_masking_parts(G, i, q=2**40):
    """
    Runs Protocol 1

    Parameters
    ----------
    G : Graph
    i : int
        Aggregating agent
    q : int
        Modulus

    Returns
    -------
    tuple of int
        {m_ij(t)} for all agents j who are neighbors of agent i
    """

    # Neighbors of i plus i itself
    N_i_plus = set(G.neighbors(i)) | {i}

    # Precompute N_j^+ for all j in N_i^+
    N_plus_dict = {}
    for j in N_i_plus:
        N_plus_dict[j] = set(G.neighbors(j)) | {j}

    # Step 1: Each j generates shares s^i_{j->l} for l in N_i^+ ∩ N_j^+
    shares_dict = {}  # keys: (j,l), value: share s^i_{j->l}
    for j in N_i_plus:
        common_neighbors = N_i_plus & N_plus_dict[j]
        n_ij = len(common_neighbors)
        zero_shares = share(n_ij, q)
        for l, share_val in zip(common_neighbors, zero_shares):
            shares_dict[(j,l)] = share_val

    # Step 2: Each j computes m_ij(t) = sum_{l in N_i^+ ∩ N_j^+} s^i_{l->j} mod q
    m = {}
    for j in N_i_plus:
        common_neighbors = N_i_plus & N_plus_dict[j]
        s_sum = mod(sum(shares_dict[(l,j)] for l in common_neighbors),q)
        m[j] = s_sum
    return m

def distributed_masking_for_all(G, q):
    """
    Repeats Protocol 1 for all i

    Parameters
    ----------
    G : Graph
    q : int
        Modulus

    Returns
    -------
    tuple of masks
    """
    masks_t = {}
    for i in G.nodes():
        masks_t[i] = generate_masking_parts(G, i, q)
    masks_by_agent = {j: {} for j in G.nodes()}
    for i, m_ij in masks_t.items():
        for j, val in m_ij.items():
            masks_by_agent[j][i] = val
    return masks_by_agent


def privacy_preserving_avg_consensus(zini,G,config, T):
    """
    Runs Protocol 2

    Parameters
    ----------
    zini : float
        Initial value
    G : Graph
    config: class
        Scale factors, modulus, and weight matrix
    T : int
        Maximum number of iterations

    Returns
    -------
    int
        Generated random integer in `[min, max)`.
    """
    q = config["q"]
    Lz = config["Lz"]
    Lw = config["Lw"]
    Wbar = config["Wbar"]
    
    z = zini 
    z_history_pp = [z.copy()]  

    
    for _ in range(T):
        # Storage
        z_next = np.zeros_like(z)
        # Run Protocol 1
        masks_by_agent = distributed_masking_for_all(G, q)
        for i in G.nodes():
            # Qunatize z_i
            z_i_q = quant(z[i], Lz)
            sum_term = 0
            # Aggregation
            for j in G.neighbors(i):
                # mask from agent j for i
                m_ij = masks_by_agent[j][i]  
                z_j_q = quant(z[j], Lz)
                w_ij = Wbar[i, j]

                # Compute ζ_ij(t) = w_ij * Q(z_j) + m_ij
                zeta_ij = w_ij * z_j_q + m_ij

                # Add ζ_ij(t) and subtract w_ij * Q(z_i) mod q
                sum_term = mod(sum_term + zeta_ij - w_ij * z_i_q, q)

            # Add agent i's own mask m_{ii}(t)
            m_ii = masks_by_agent[i][i]  
            total = mod(m_ii + sum_term, q)

            # Update z_i(t+1)
            z_next[i] = z[i] + Lw * Lz * total

        z = z_next
        z_history_pp.append(z.copy())

    return z_history_pp