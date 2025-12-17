import numpy as np

def bits_to_bpsk(x):
    # 0->+1, 1->-1
    return 1.0 - 2.0 * x

def awgn_channel(x_bits, ebn0_db, rate):
    """
    x_bits: array of bits length n
    ebn0_db: Eb/N0 in dB
    rate: code rate R = k/n
    returns received r (real) and sigma2
    """
    ebn0 = 10**(ebn0_db/10.0)
    esn0 = ebn0 * rate
    sigma2 = 1.0 / (2.0 * esn0)
    sigma = np.sqrt(sigma2)
    s = bits_to_bpsk(x_bits)
    noise = sigma * np.random.randn(*s.shape)
    r = s + noise
    return r, sigma2

def llr_from_awgn(r, sigma2):
    return 2.0 * r / sigma2