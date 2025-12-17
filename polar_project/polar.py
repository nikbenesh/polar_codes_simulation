import numpy as np
import pandas as pd
from pathlib import Path


_rank_df = pd.read_csv("rank.csv")
print(_rank_df.head(5))
print(_rank_df.columns)
NR_RELIABILITY_SEQUENCE = _rank_df["Q"].values


def polar_transform(u):
    """In-place polar transform (Arikan) - returns codeword x (0/1)"""
    n = len(u)
    x = u.copy()
    m = int(np.log2(n))
    # iterative butterfly
    step = 1
    for s in range(m):
        half = step
        step *= 2
        for i in range(0, n, step):
            for j in range(half):
                a = x[i + j]
                b = x[i + j + half]
                x[i + j] = (a ^ b) & 1
                # x[i + j + half] stays as b
        # next stage
    return x

def encode_old(u_info_bits, frozen_mask, reliability_order=None, info_positions=None):
    """
    Build u (length n) given info bits and frozen_mask (boolean mask: True if frozen).
    Place info bits in non-frozen positions in increasing reliability (or by order of indices).
    Then apply polar_transform to get codeword.
    """
    n = len(frozen_mask)
    k = len(u_info_bits)
    u = np.zeros(n, dtype=np.int8)
    # positions where frozen_mask is False are information positions
    if info_positions is None:
        info_positions = np.where(~frozen_mask)[0]
    if len(info_positions) < k:
        raise ValueError("Not enough information positions for k bits")
    u[info_positions[:k]] = u_info_bits[:k]
    x = polar_transform(u)
    return x, u


def encode(u_info_bits, frozen_mask, info_positions_ordered=None):
    """
    Build u (length n) given info bits and frozen_mask (boolean mask: True if frozen).
    If info_positions_ordered is provided, it must be an array of info indices ordered by reliability
    (most reliable first). Then the first k positions of that array will receive the info bits.
    """
    n = len(frozen_mask)
    k = len(u_info_bits)
    u = np.zeros(n, dtype=np.int8)
    if info_positions_ordered is None:
        info_positions = np.where(~frozen_mask)[0]
        if len(info_positions) < k:
            raise ValueError("Not enough information positions for k bits")
        u[info_positions[:k]] = u_info_bits[:k]
    else:
        if len(info_positions_ordered) < k:
            raise ValueError("info_positions_ordered too short")
        u[info_positions_ordered[:k]] = u_info_bits[:k]
    x = polar_transform(u)
    return x, u


def bit_reverse_indices(n):
    m = int(np.log2(n))
    idx = np.arange(n)
    rev = np.zeros(n, dtype=int)
    for i in range(n):
        b = format(i, 'b').zfill(m)[::-1]
        rev[i] = int(b, 2)
    return rev

def gaussian_approx_reliability(n, design_snr=-1.0):
    """
    Compute reliability ordering using Gaussian Approximation (GA) for AWGN.
    Returns an array of indices from least to most reliable (so lower first).
    design_snr in dB (default -1 dB used by some constructions).
    This gives a reasonable reliability ordering when the 5G sequence is not available.
    """
    m = int(np.log2(n))
    N = n
    # initialize channel LLR means for AWGN: use L = 2*Es/N0 approx; choose design Eb/N0
    # We'll work with Bhattacharyya-like recursion using means
    ebn0 = 10**(design_snr/10)
    # initialize with L0 = 2*Es/N0; Es normalized to 1; for convenience
    L0 = 2 * ebn0
    means = [L0]
    # build means through polarization tree
    for _ in range(m):
        new_means = []
        for mu in means:
            # f node approx for mean: phi^-1(1 - (1-phi(mu))^2) approximate with simple transform
            # but simpler: use mu_f = phi_inv(1 - (1 - phi(mu))**2)
            # We'll use crude approximations: mu_upper = phi_inv(1 - (1-phi(mu))**2)
            # For simplicity (and speed), use heuristic:
            mu_u = 2 * np.arctanh(np.tanh(mu/2)**2) if mu>0 else 0.0
            mu_v = 2 * mu
            # handle numeric issues
            if np.isfinite(mu_u) and mu_u>0:
                new_means.append(mu_u)
            else:
                new_means.append(0.0)
            new_means.append(mu_v)
        means = new_means
    means = np.array(means)
    # reliability: larger mean => more reliable; we need ordering from least to most reliable
    order = np.argsort(means)
    return order


def make_frozen_mask(n, k, reliability_order=None, use_nr_sequence=False):
    """
    Return (frozen_mask, info_positions_ordered)
    frozen_mask: boolean array length n: True where frozen.
    info_positions_ordered: numpy array of length n-k? or at least the ordered info indices
    If use_nr_sequence=True, use NR_RELIABILITY_SEQUENCE (must be loaded in module).
    """
    if use_nr_sequence:
        # NR_RELIABILITY_SEQUENCE: array where first element is most reliable (Q[0])
        # seq_full = NR_RELIABILITY_SEQUENCE  # assume loaded at module top
        # # take only positions < n and preserve their order (most->least reliable)
        # seq_n = [int(q) for q in seq_full if int(q) < n]
        # if len(seq_n) < n:
        #     raise ValueError(f"NR sequence does not provide enough indices < {n}")
        # # info positions should be the K most reliable (preserve order)
        # info_positions_ordered = np.array(seq_n[:k], dtype=int)
        # # the remaining (less reliable) become frozen
        # frozen_indices = seq_n[k:n]
        # frozen = np.zeros(n, dtype=bool)
        # frozen[frozen_indices] = True
        # return frozen, info_positions_ordered

        rev = bit_reverse_indices(n)

        # NR sequence is given in bit-reversed domain
        # We must map it back to natural indices
        seq_n = []
        for q in NR_RELIABILITY_SEQUENCE:
            q = int(q)
            if q < n:
                seq_n.append(rev[q])
        
        seq_n = np.array(seq_n, dtype=int)
        
        # info_positions_ordered = seq_n[:k]
        # frozen_indices = seq_n[k:n]

        info_positions_ordered = seq_n[-k:]
        frozen_indices = seq_n[:-k]
        
        frozen = np.zeros(n, dtype=bool)
        frozen[frozen_indices] = True
        
        return frozen, info_positions_ordered

    # old GA-based behaviour (keep backward compat)
    if reliability_order is None:
        reliability_order = gaussian_approx_reliability(n)
    # reliability_order currently is least->most; frozen are least reliable first (N-K positions)
    frozen = np.zeros(n, dtype=bool)
    num_frozen = n - k
    frozen_indices = reliability_order[:num_frozen]
    frozen[frozen_indices] = True
    # Build ordered info positions according to reliability: most -> least
    # reliability_order is least->most, so take the last k entries and reverse to most->least
    info_positions_ordered = np.array(reliability_order[num_frozen:][::-1], dtype=int)
    return frozen, info_positions_ordered



def make_frozen_mask_old(n, k, reliability_order=None, use_nr_sequence=False):
    """
    return frozen_mask (boolean array length n): True where frozen.
    reliability_order: array of indices from least to most reliable (length n).
    If None, compute GA-based order.
    """
    # if use_nr_sequence:
    #     # NR_RELIABILITY_SEQUENCE: from most → least reliable
    #     seq = NR_RELIABILITY_SEQUENCE[:n]
    
    #     # frozen = least reliable = последние (n-k)
    #     frozen_indices = seq[k:n]
    #     frozen = np.zeros(n, dtype=bool)
    #     frozen[frozen_indices] = True
    #     return frozen

    if use_nr_sequence:
        # NR_RELIABILITY_SEQUENCE: from most → least reliable (length 1024)
        seq_full = NR_RELIABILITY_SEQUENCE
    
        # take only positions < n, keep order
        seq_n = [q for q in seq_full if q < n]
    
        if len(seq_n) < n:
            raise ValueError("NR reliability sequence does not cover this n")
    
        # most reliable first → info bits
        info_indices = seq_n[:k]
    
        # rest → frozen
        frozen_indices = seq_n[k:n]
    
        frozen = np.zeros(n, dtype=bool)
        frozen[frozen_indices] = True
        # return frozen
        return frozen, info_indices



    if reliability_order is None:
        reliability_order = gaussian_approx_reliability(n)
    # reliability_order currently is least->most; frozen are least reliable first (N-K positions)
    frozen = np.zeros(n, dtype=bool)
    num_frozen = n - k
    frozen_indices = reliability_order[:num_frozen]
    frozen[frozen_indices] = True
    return frozen

if __name__ == "__main__":
    import numpy as np
    # tiny self-test
    n=8
    k=4
    reliability = np.argsort(np.random.randn(n))  # dummy
    frozen = make_frozen_mask(n,k,reliability_order=reliability)
    info = np.random.randint(0,2,k)
    x,u=encode(info,frozen)
    print("info",info,"u",u,"x",x)