import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def save_results_csv(path, rows, header=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)

def plot_fer_snr(results, out_path, title=None):
    """
    results: list of tuples (label, snr_db_list, fer_list)
    """
    plt.figure(figsize=(7,5))
    for label, snrs, fers in results:
        plt.semilogy(snrs, fers, marker='o', label=label)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('FER')
    if title:
        plt.title(title)
    plt.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()