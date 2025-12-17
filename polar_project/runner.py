import numpy as np
from pathlib import Path
from .polar import make_frozen_mask, encode, polar_transform, gaussian_approx_reliability
from .awgn import awgn_channel, llr_from_awgn, bits_to_bpsk
from .scl import scl_decode_naive, sc_decode_from_llr
from .utils import save_results_csv, plot_fer_snr
from .crc import append_crc
from .crc import check_crc
import time
import itertools


def run_one_case(
    n, 
    rate, 
    L, 
    mode='random', 
    ebn0_range=None, 
    stop_errors=30, 
    max_frames=200000, 
    yield_mode=False, 
    save_results=True, 
    use_nr_sequence=True):
    k = int(np.round(n * rate))
    if k <= 0:
        raise ValueError("k must be > 0")
    # prepare frozen mask (GA-based ordering)

    # without rank.csv 
    # reliability = gaussian_approx_reliability(n)
    # frozen_mask = make_frozen_mask(n, k, reliability_order=reliability)

    # 5G NR - using rank.csv 
    frozen_mask, info_positions_ordered = make_frozen_mask(n, k, use_nr_sequence=use_nr_sequence)
    print(f"[debug] n={n}, k={k}, len(info_positions_ordered)={len(info_positions_ordered)}")
    print(f"[debug] first info pos (most reliable): {info_positions_ordered[:10]}")
    print(f"[debug] max index in info positions: {np.max(info_positions_ordered)}; should be < {n}")


    labels = []
    rows_all = []
    results_for_plot = []
    if ebn0_range is None:
        # choose a default range heuristically based on rate
        if rate <= 1/3:
            ebn0s = np.arange(0.0, 4.5, 0.5)
        elif rate <= 0.5:
            ebn0s = np.arange(0.5, 6.0, 0.5)
        else:
            ebn0s = np.arange(1.0, 7.0, 0.5)
    else:
        ebn0s = np.array(ebn0_range)
    fer_list = []
    snr_list = []
    for eb in ebn0s:
        n_frames = 0
        n_errors = 0
        # Run frames until stop_errors reached or max_frames used
        while n_errors < stop_errors and n_frames < max_frames:
            n_frames += 1
            # generate info bits or zero-codeword mode
            if mode == 'random':
                # info = np.random.randint(0,2,k)
                crc_len = 8
                info = np.random.randint(0, 2, k - crc_len)
                info = append_crc(info)

                x, u = encode(info, frozen_mask, info_positions_ordered=info_positions_ordered)
            else:
                # zero-codeword: send all-zero u => codeword zeros
                info = np.zeros(k, dtype=int)
                x, u = encode(info, frozen_mask, info_positions_ordered=info_positions_ordered)
            # send through AWGN using code rate
            r, sigma2 = awgn_channel(x, eb, rate)
            llr = llr_from_awgn(r, sigma2)
            # decode with SCL naive
            # u_hat = scl_decode_naive(llr, frozen_mask, L=L)
            u_hat, paths = scl_decode_naive(llr, frozen_mask, L=L, return_list=True)
            # info_pos = np.where(~frozen_mask)[0][:k]
            info_pos = info_positions_ordered[:k]

            # CRC check 
            for path in paths:
                # info_bits = path['u'][info_pos]
                info_bits = np.asarray(path['u'][info_pos], dtype=int)
                if check_crc(info_bits):
                    u_hat = path['u']
                    break

            # extract info bits from u_hat at info positions (non-frozen)
            info_hat = u_hat[info_pos]
            if not np.array_equal(info_hat, info):
                n_errors += 1
        fer = n_errors / float(n_frames) if n_frames>0 else 1.0
        fer_list.append(fer)
        snr_list.append(eb)
        print(f"n={n}, R={rate:.3f}, L={L}, mode={mode}, Eb/N0={eb:.2f} dB: frames={n_frames}, errors={n_errors}, FER={fer:.3e}")
        if yield_mode:
            yield eb, fer, n_frames, n_errors
    # save CSV
    header = ['EbN0_dB', 'N_frames', 'N_errors', 'FER']
    rows = []
    # We didn't store N_frames per point in detail, but for simplicity store fer and n_frames approximated. 
    for snr, fer in zip(snr_list, fer_list):
        rows.append([snr, '>=stop_errors' if fer>0 else 0, int(np.round(fer*1e3)), fer])
    if save_results:
        outdir = Path('polar_project/results_nr')
        outdir.mkdir(parents=True, exist_ok=True)
        csvname = outdir / f'res_n{n}_R{int(rate*100) }_L{L}_mode_{mode}.csv'
        save_results_csv(csvname, rows, header=header)
        # plot
        plot_fer_snr([(f"L={L} mode={mode}", snr_list, fer_list)], outdir / f'fer_n{n}_R{int(rate*100)}_L{L}_mode_{mode}.png',
                     title=f"n={n} R={rate:.3f} L={L} mode={mode}")
    if not yield_mode:
        return snr_list, fer_list

def quick_demo():
    # quick smoke test: one small case with reduced stop_errors to run fast
    print("Running quick demo (n=128, R=1/2, L=4, stop_errors=6) ...")
    return run_one_case(n=128, rate=0.5, L=4, mode='random', ebn0_range=np.arange(0.5,3.5,0.5), stop_errors=6, max_frames=2000)

if __name__ == '__main__':
    quick_demo()