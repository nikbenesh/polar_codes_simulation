# polar_project/run_all.py
import time
import itertools
from pathlib import Path
import csv

import numpy as np

from .runner import run_one_case
from .utils import plot_fer_snr, save_results_csv

# ============ Параметры, как в задании (запуск без аргументов) ============
NS = [128, 256, 512]
RATES = [1/3, 1/2, 2/3]
LIST_SIZES = [4, 8] # [4, 8, 16]
# Можно еще запустить без 16 - будет быстрее. 

MODE = "zero"            # zero-codeword mode (быстрый и стабильный для сравнения)
STOP_ERRORS = 30
MAX_FRAMES = 200_000

USE_NR_SEQUENCE = True   # использовать rank.csv (5G NR) по умолчанию
RESULTS_DIR = Path("polar_project/results_nr_crc")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================================================


def save_case_results(n, R, L, mode, snr_list, fer_list, frames_list, errors_list):
    """
    Сохраняет CSV и PNG для одного кейса.
    CSV columns: EbN0_dB, N_frames, N_errors, FER
    """
    # prepare rows for CSV
    header = ["EbN0_dB", "N_frames", "N_errors", "FER"]
    rows = []
    for snr, fer, frames, errors in zip(snr_list, fer_list, frames_list, errors_list):
        rows.append([float(snr), int(frames), int(errors), float(fer)])

    # filenames
    rate_int = int(round(R * 100))
    csvname = RESULTS_DIR / f"res_n{n}_R{rate_int}_L{L}_mode_{mode}.csv"
    pngname = RESULTS_DIR / f"fer_n{n}_R{rate_int}_L{L}_mode_{mode}.png"

    # save CSV (use helper for consistent format)
    try:
        save_results_csv(csvname, rows, header=header)
        print(f"[save] CSV -> {csvname}")
    except Exception as e:
        # fallback simple csv write
        with open(csvname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[save-fallback] CSV -> {csvname} (via csv module). Error: {e}")

    # save plot
    try:
        series = [(f"L={L} mode={mode}", snr_list, fer_list)]
        plot_fer_snr(series, pngname, title=f"n={n} R={R:.3f} L={L} mode={mode}")
        print(f"[save] PNG -> {pngname}")
    except Exception as e:
        print(f"[warn] Failed to save PNG {pngname}: {e}")


def run_all():
    total_cases = len(NS) * len(RATES) * len(LIST_SIZES)
    case_idx = 0
    t_start_all = time.time()

    print("=" * 80)
    print("Batch run of polar code FER simulations (yield-mode).")
    print(f"Total cases to run: {total_cases}")
    print(f"NS = {NS}")
    print(f"RATES = {RATES}")
    print(f"LIST_SIZES = {LIST_SIZES}")
    print(f"mode = {MODE}, stop_errors = {STOP_ERRORS}, max_frames = {MAX_FRAMES}")
    print(f"use_nr_sequence = {USE_NR_SEQUENCE}")
    print("Results will be saved to:", RESULTS_DIR.resolve())
    print("=" * 80)

    try:
        for n, R, L in itertools.product(NS, RATES, LIST_SIZES):
            case_idx += 1
            print("\n" + "-" * 80)
            print(f"Case {case_idx}/{total_cases}: n={n}, R={R:.3f}, L={L}, mode={MODE}")
            print("-" * 80)

            snr_list = []
            fer_list = []
            frames_list = []
            errors_list = []

            t_case_start = time.time()

            # Call generator version of run_one_case. We use yield_mode=True so that
            # run_one_case yields one tuple (eb, fer, frames, errors) per SNR point.
            gen = run_one_case(
                n=n,
                rate=R,
                L=L,
                mode=MODE,
                ebn0_range=None,          # use internal heuristic ranges
                stop_errors=STOP_ERRORS,
                max_frames=MAX_FRAMES,
                yield_mode=True,
                save_results=False,       # we will save after case completes (or on interrupt)
                use_nr_sequence=USE_NR_SEQUENCE
            )

            # Iterate generator and collect points as they come.
            # This also allows saving partial results on KeyboardInterrupt.
            try:
                point_idx = 0
                for eb, fer, n_frames, n_errors in gen:
                    point_idx += 1
                    snr_list.append(float(eb))
                    fer_list.append(float(fer))
                    frames_list.append(int(n_frames))
                    errors_list.append(int(n_errors))

                    # print progress for this point
                    print(f"  point {point_idx}: Eb/N0={eb:.2f} dB, frames={n_frames}, errors={n_errors}, FER={fer:.3e}")

                # generator exhausted normally -> full case finished
                t_case_elapsed = time.time() - t_case_start
                print(f"Case finished in {t_case_elapsed:.1f}s, points={len(snr_list)}")

                # Save results for this case
                save_case_results(n, R, L, MODE, snr_list, fer_list, frames_list, errors_list)

            except KeyboardInterrupt:
                # If user interrupts while computing a case, save partial results collected so far.
                print("\n[interrupt] Received KeyboardInterrupt during case computation.")
                if len(snr_list) > 0:
                    print(f"[interrupt] Saving partial results for case n={n}, R={R}, L={L} (points={len(snr_list)})")
                    save_case_results(n, R, L, MODE, snr_list, fer_list, frames_list, errors_list)
                else:
                    print("[interrupt] No points collected for this case; nothing to save.")
                raise  # re-raise to stop batch run

    except KeyboardInterrupt:
        print("\nBatch interrupted by user. Exiting.")
    finally:
        total_elapsed = time.time() - t_start_all
        print("\n" + "=" * 80)
        print(f"Batch run finished (or interrupted). Total elapsed: {total_elapsed/60:.2f} min")
        print("Results folder:", RESULTS_DIR.resolve())
        print("=" * 80)


if __name__ == "__main__":
    run_all()
