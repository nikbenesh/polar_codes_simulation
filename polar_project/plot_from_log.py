import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ===== настройки =====
log_file = "polar_project/results_log/log_n128_R067_L4.txt"
out_png = "polar_project/results_log/fer_n128_R067_L4.png"
out_csv = "polar_project/results_log/fer_n128_R067_L4.csv"
# =====================

ebn0 = []
fer = []

pattern = re.compile(
    r"Eb/N0=([\d.]+)\s*dB.*FER=([\deE\-.]+)"
)

with open(log_file, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ebn0.append(float(m.group(1)))
            fer.append(float(m.group(2)))

ebn0 = np.array(ebn0)
fer = np.array(fer)

# ---- сортировка по Eb/N0 (на всякий случай) ----
idx = np.argsort(ebn0)
ebn0 = ebn0[idx]
fer = fer[idx]

# ---- сохранение CSV ----
with open(out_csv, "w") as f:
    f.write("EbN0_dB,FER\n")
    for x, y in zip(ebn0, fer):
        f.write(f"{x},{y}\n")

# ---- построение графика ----
plt.figure(figsize=(6, 4))
plt.semilogy(ebn0, fer, marker='o')
plt.grid(True, which="both")
plt.xlabel("Eb/N0, dB")
plt.ylabel("FER")
plt.title("FER vs Eb/N0\nn=128, R=0.667, L=4")
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()

print(f"[OK] График сохранён: {out_png}")
print(f"[OK] Таблица сохранена: {out_csv}")
