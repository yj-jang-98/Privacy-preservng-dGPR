import torch
import numpy as np
from utils import *

def generate_masking_parts(G, i, q, p, device):
    """
    Section 3.4

    Parameters
    ----------
        G : Graph
        i : Aggregating agent
        q : Modulus
        p : Dimension
    Returns
    -------
        {phi_ij(t)} for all agents j who are neighbors of agent i (including itself)
    """

    N_i_plus = set(G.neighbors(i)) | {i}
    N_plus_dict = {j: set(G.neighbors(j)) | {j} for j in N_i_plus}

    # --- Eq. (15a) and (15b) using `share()`
    shares_dict = {}  # keys: (j, l), value: share vector s^i_{j->l} ∈ Z_q^p

    for j in N_i_plus:
        common_neighbors = sorted(N_i_plus & N_plus_dict[j])
        n_ij = len(common_neighbors)

        # Generate (n_ij - 1) random vectors in [-q//2, q//2)
        partial_shares = torch.randint(
            low=(-q + 1) // 2,
            high=q // 2,
            size=(n_ij - 1, p),
            dtype=torch.int64,
            device=device
        )

        # Compute the last share so that their sum is zero mod q
        last_share = -partial_shares.sum(dim=0)
        last_share = mod_vec(last_share, q)

        # Combine all shares
        all_shares = torch.cat([partial_shares, last_share.unsqueeze(0)], dim=0)

        for l, share_vec in zip(common_neighbors, all_shares):
            shares_dict[(j, l)] = share_vec

    # --- Eq. (15c)
    phi = {}
    for j in N_i_plus:
        common_neighbors = N_i_plus & N_plus_dict[j]
        s_sum = torch.zeros(p, dtype=torch.int64, device=device)
        for l in common_neighbors:
            s_sum += shares_dict[(l, j)]
        phi[j] = mod_vec(s_sum, q)

    return phi

def distributed_masking_for_all(G, q, p, device):
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
        masks_t[i] = generate_masking_parts(G, i, q, p, device)

    # Reorganize: for each agent j, collect m_ij(t) from all i
    masks_by_agent = {j: {} for j in G.nodes()}
    for i, m_ij in masks_t.items():
        for j, val in m_ij.items():
            masks_by_agent[j][i] = val.to(device)  # val ∈ Z_q^p (np.ndarray)

    return masks_by_agent


def privacy_preserving_avg_consensus(zini, G, config, T, device):
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
    Wbar = torch.tensor(config["Wbar"], dtype=torch.float32, device=device)

    if isinstance(zini, np.ndarray):
        z = torch.tensor(zini, dtype=torch.float32, device=device)
    else:
        z = zini.to(device=device, dtype=torch.float32)

    M, p = z.shape
    z_history_pp = [z.clone()]

    for _ in range(T):
        z_next = torch.zeros_like(z)

        # Run Protocol 1 for all agents
        masks_by_agent = distributed_masking_for_all(G, q, p, device)

        for i in G.nodes():
            z_i_q = torch.round(z[i] / Lz).to(torch.int64)  # shape (p,)
            sum_term = torch.zeros(p, dtype=torch.int64, device=device)

            for j in G.neighbors(i):
                m_ij = masks_by_agent[j][i]       # shape (p,)
                z_j_q = torch.round(z[j] / Lz).to(torch.int64)    # shape (p,)
                w_ij = Wbar[i, j]                 # scalar

                zeta_ij = (w_ij * z_j_q).to(torch.int64) + m_ij
                sum_term = (sum_term + zeta_ij - (w_ij * z_i_q).to(torch.int64)) % q

                sum_term = torch.where(sum_term >= q // 2, sum_term - q, sum_term)

            # Add agent i's own mask
            m_ii = masks_by_agent[i][i]           # shape (p,)
            total = (m_ii + sum_term) % q
            total = torch.where(total >= q // 2, total - q, total)

            z_next[i] = z[i] + (Lw * Lz * total).float()

        z = z_next
        z_history_pp.append(z.clone())

    return z_history_pp
