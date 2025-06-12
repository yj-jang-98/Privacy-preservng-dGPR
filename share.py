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

def share(n, q, p):
    """
    Creat n-shares of zero in Z_q

    Parameters
    ----------
    n : int
        Number of shares
    q : int
        Modulus
    p : int
        Dimension

    Returns
    -------
    nparray of n-shares
        (s_1,...,s_n)
    """

    shares = np.array([
        [get_rand((-q+1)//2, q//2) for _ in range(p)]
        for _ in range(n - 1)
    ])
    last_share = mod_vec(-shares.sum(axis=0), q)
    shares = np.vstack([shares, last_share])
    return shares


def generate_masking_parts(G, i, q, p):
    """
    Runs Protocol 1

    Parameters
    ----------
    G : Graph
    i : int
        Aggregating agent
    q : int
        Modulus
    p : int
        Dimension
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
    shares_dict = {}  # keys: (j,l), value: vector share s^i_{j->l} ∈ Z_q^p
    for j in N_i_plus:
        common_neighbors = N_i_plus & N_plus_dict[j]
        n_ij = len(common_neighbors)
        zero_shares = share(n_ij, q, p)  # shape (n_ij, p)
        for l, share_vec in zip(common_neighbors, zero_shares):
            shares_dict[(j, l)] = share_vec

    # Step 2: Each j computes m_ij(t) = sum_{l in N_i^+ ∩ N_j^+} s^i_{l->j} mod q
    m = {}
    for j in N_i_plus:
        common_neighbors = N_i_plus & N_plus_dict[j]
        s_sum = sum(shares_dict[(l, j)] for l in common_neighbors)
        m[j] = mod_vec(s_sum, q)  # s_sum ∈ Z^p → mod applied elementwise
    return m

def distributed_masking_for_all(G, q, p):
    """
    Runs Protocol 1 for all agents in the graph, producing p-dimensional masks.

    Parameters
    ----------
    G : Graph
        Network graph (e.g., networkx.Graph)
    q : int
        Modulus
    p : int
        Dimension 

    Returns
    -------
    dict
        Nested dict where result[j][i] = m_ij(t) ∈ Z_q^p
    """
    # First, compute all m_ij(t) for each i
    masks_t = {}
    for i in G.nodes():
        masks_t[i] = generate_masking_parts(G, i, q, p)

    # Reorganize: for each agent j, collect m_ij(t) from all i
    masks_by_agent = {j: {} for j in G.nodes()}
    for i, m_ij in masks_t.items():
        for j, val in m_ij.items():
            masks_by_agent[j][i] = val  # val ∈ Z_q^p (np.ndarray)

    return masks_by_agent


def privacy_preserving_avg_consensus(zini, G, config, T):
    """
    Runs Protocol 2 with p-dimensional states and privacy-preserving masking.

    Parameters
    ----------
    zini : np.ndarray
        Initial values of shape (n, p), where n = number of agents, p = dimension
    G : Graph
        Network graph
    config : dict
        Dictionary containing:
            - "q": modulus
            - "Lz": quantization scale
            - "Lw": learning rate / weight scaling
            - "Wbar": doubly stochastic weight matrix of shape (n, n)
    T : int
        Number of consensus iterations

    Returns
    -------
    list of np.ndarray
        History of z values over time, each entry of shape (n, p)
    """
    q = config["q"]
    Lz = config["Lz"]
    Lw = config["Lw"]
    Wbar = config["Wbar"]

    n, p = zini.shape
    z = zini.copy()
    z_history_pp = [z.copy()]

    for _ in range(T):
        z_next = np.zeros_like(z)

        # Run Protocol 1 for all agents
        masks_by_agent = distributed_masking_for_all(G, q, p)

        for i in G.nodes():
            z_i_q = quant(z[i], Lz)  # shape (p,)
            sum_term = np.zeros(p, dtype=int)

            for j in G.neighbors(i):
                m_ij = masks_by_agent[j][i]       # shape (p,)
                z_j_q = quant(z[j], Lz)           # shape (p,)
                w_ij = Wbar[i, j]                 # scalar

                zeta_ij = w_ij * z_j_q + m_ij     # shape (p,)
                sum_term = mod_vec(sum_term + zeta_ij - w_ij * z_i_q, q)

            # Add agent i's own mask
            m_ii = masks_by_agent[i][i]           # shape (p,)
            total = mod_vec(m_ii + sum_term, q)       # shape (p,)

            # Update z_i(t+1)
            z_next[i] = z[i] + Lw * Lz * total    # shape (p,)

        z = z_next
        z_history_pp.append(z.copy())

    return z_history_pp
