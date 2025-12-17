import numpy as np
from copy import deepcopy
from .polar import polar_transform, encode, make_frozen_mask


def llr_from_awgn(r, sigma2):
    # r: received real values, sigma2: noise variance per dimension
    return 2.0 * r / sigma2

def path_metric_increment(llr, bit):
    """
    Increment of path metric when we choose 'bit' given LLR.
    Use exact log-domain: increment = log(1+exp(-(-1)**bit * llr))
    """
    # To avoid overflow, use stable expression:
    a = -((-1)**bit) * llr
    # log1p(exp(a)) is stable
    return np.log1p(np.exp(a))

def sc_decode_from_llr(llr, frozen_mask):
    """
    Simple SC decoder (L=1) that returns estimated u (length n).
    This implementation uses recursive f/g functions via arrays.
    """
    n = len(llr)
    m = int(np.log2(n))
    # We'll implement iterative SC using recursion on u-hat and LLR arrays
    # For simplicity use a recursive function that computes LLs on the fly
    u_hat = np.zeros(n, dtype=np.int8)

    def recurse(level, idx, llr_sub):
        # level 0..m, idx: starting index in u_hat; llr_sub length = 2**(m-level)
        if len(llr_sub) == 1:
            i = idx
            if frozen_mask[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_sub[0] >= 0 else 1
            return np.array([u_hat[i]], dtype=int)
        half = len(llr_sub)//2
        # compute f for left
        llr_left = np.zeros(half)
        for i in range(half):
            a = llr_sub[i]
            b = llr_sub[i+half]
            # f = sign(a)*sign(b)*min(|a|,|b|) for min-sum approx; but we need exact SC LLRs
            # Use exact f: L_f = 2 * atanh( tanh(a/2) * tanh(b/2) )
            # Handle numeric issues:
            ta = np.tanh(a/2.0); tb = np.tanh(b/2.0)
            val = 0.0
            prod = ta * tb
            # clip
            prod = np.clip(prod, -0.999999999999, 0.999999999999)
            val = 2.0 * np.arctanh(prod)
            llr_left[i] = val
        # left recursion
        left_u = recurse(level+1, idx, llr_left)
        # compute g for right
        llr_right = np.zeros(half)
        for i in range(half):
            a = llr_sub[i]
            b = llr_sub[i+half]
            # g = b + (1 - 2*left_u[i]) * a
            llr_right[i] = b + (1 - 2*left_u[i]) * a
        right_u = recurse(level+1, idx+half, llr_right)
        # combine u's already filled through recursion
        for i in range(half):
            u_hat[idx + i] = (left_u[i] ^ right_u[i]) & 1
            u_hat[idx + half + i] = right_u[i] & 1
        return np.concatenate([left_u, right_u])

    recurse(0, 0, llr.copy())
    return u_hat

def scl_decode_naive(llr, frozen_mask, L=4, return_list=False):
    """
    Naive SCL where for each candidate we keep full SC state and recompute subsequent LLRs.
    This is correct but memory- and time-inefficient; used for clarity.
    llr: input LLR array length n
    frozen_mask: boolean array length n
    returns: estimated u_hat (length n)
    """
    n = len(llr)
    # each path: dictionary {'u_hat_partial': array up to pos i, 'metric': float}
    paths = [{'u': np.zeros(0, dtype=np.int8), 'metric': 0.0}]
    for i in range(n):
        new_paths = []
        for path in paths:
            # build partial llr given path.u and full llr by running SC up to pos i to obtain LLR_i
            # For simplicity, we'll run SC but stopping at position i to get decision LLR
            # Define mask to indicate which bits are frozen/decided: we pretend unknown bits are not set yet
            # We'll use a function that computes decision LLR for bit i given previous decisions
            # Simpler: brute-force: for each candidate bit value b, compute PM increment using approximate LLR at position i
            # Use SC to get the LLR for position i given path.u
            # We'll simulate SC on the tree but with previous bits fixed to path.u
            # Build modified llr array by performing SC f/g using fixed bits when needed
            # For speed, call sc_decision_llr helper:
            llr_i = sc_decision_llr(llr, path['u'], i)
            if frozen_mask[i]:
                b = 0
                new_u = np.concatenate([path['u'], np.array([b], dtype=np.int8)])
                incr = path_metric_increment(llr_i, b)
                new_paths.append({'u': new_u, 'metric': path['metric'] + incr})
            else:
                for b in (0,1):
                    new_u = np.concatenate([path['u'], np.array([b], dtype=np.int8)])
                    incr = path_metric_increment(llr_i, b)
                    new_paths.append({'u': new_u, 'metric': path['metric'] + incr})
        # keep best L paths
        new_paths.sort(key=lambda p: p['metric'])
        paths = new_paths[:L]
    # choose best final path (metric minimum)
    # for path in paths:
    #     if check_crc(path[info_positions]):
    #         return path
    best = min(paths, key=lambda p: p['metric'])
    # best['u'] is full u_hat   

    if not return_list:
        return best['u']
    else:
        return (best['u'], paths)

def sc_decision_llr(llr, decided_bits, pos):
    """
    Compute the decision LLR for bit position pos given previous decided bits (0..pos-1)
    Using recursive tree computations but where nodes requiring prior bits use decided values.
    This is a helper used by the naive SCL; it's not optimized.
    """
    # We'll implement a recursive function that returns LLR at a specific leaf index while
    # taking into account decided bits.
    n = len(llr)
    m = int(np.log2(n))

    # We'll implement recursion that returns pair (llr_array, u_array) at each subtree;
    # but to get LLR at pos, we proceed similarly to SC decoding but using decided_bits when available.
    # For simplicity and correctness we can run full SC but when we encounter positions < len(decided_bits) we use them.
    u_hat = np.zeros(n, dtype=np.int8)
    def recurse_llr(level, idx, llr_sub, start_pos):
        # returns list of LLRs for leaves under this subtree and fills u_hat for decided positions
        if len(llr_sub) == 1:
            i = start_pos
            # if this bit is decided, its LLR still computed, but SC decision uses it
            return llr_sub
        half = len(llr_sub)//2
        # compute left LLRs
        llr_left = np.zeros(half)
        for i in range(half):
            a = llr_sub[i]; b = llr_sub[i+half]
            ta = np.tanh(a/2.0); tb = np.tanh(b/2.0)
            prod = ta * tb
            prod = np.clip(prod, -0.999999999999, 0.999999999999)
            llr_left[i] = 2.0 * np.arctanh(prod)
        left_leaf_llrs = recurse_llr(level+1, idx, llr_left, start_pos)
        # determine left hard decisions if decided bits exist in that range
        left_u = np.zeros(len(left_leaf_llrs), dtype=np.int8)
        for i in range(len(left_leaf_llrs)):
            pos_i = start_pos + i
            if pos_i < len(decided_bits):
                left_u[i] = decided_bits[pos_i]
            else:
                left_u[i] = 0 if left_leaf_llrs[i] >= 0 else 1
        # compute right LLRs
        llr_right = np.zeros(half)
        for i in range(half):
            a = llr_sub[i]; b = llr_sub[i+half]
            llr_right[i] = b + (1 - 2*left_u[i]) * a
        right_leaf_llrs = recurse_llr(level+1, idx+half, llr_right, start_pos+half)
        return np.concatenate([left_leaf_llrs, right_leaf_llrs])

    leaf_llrs = recurse_llr(0, 0, llr.copy(), 0)
    return leaf_llrs[pos]

# Exported functions
__all__ = ['scl_decode_naive', 'sc_decode_from_llr']
