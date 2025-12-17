import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import itertools
from collections import OrderedDict

from polar_project.runner import run_one_case
from polar_project.polar import encode, make_frozen_mask
from polar_project.awgn import awgn_channel, llr_from_awgn
from polar_project.scl import scl_decode_naive
from polar_project.utils import save_results_csv, plot_fer_snr


RESULTS_DIR = "polar_project/results_st"
PLOT_FILENAME_TEMPLATE = "overlay_fer_plot.png"


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Polar Codes Simulator (5G NR)",
    layout="wide"
)

# --------------------------------------------------
# Sidebar – navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Навигация",
    [
        "🏠 Обзор",
        "📘 Теория полярных кодов",
        "⚙️ Симулятор (FER)",
        "🔍 Пример одного кадра",
    ]
)

# --------------------------------------------------
# PAGE 1 – Overview
# --------------------------------------------------
if page == "🏠 Обзор":
    st.title("Полярные коды (5G NR)")
    st.markdown("""
    Это интерактивный симулятор для изучения **полярных кодов**, 
    используемых в стандарте **5G NR**.

    ### Возможности:
    - моделирование передачи в канале **AWGN**
    - **SCL-декодирование** с разным размером списка
    - построение кривых **FER(Eb/N0)**
    - пошаговый пример кодирования и декодирования
    """)

    st.success("Проект предназначен для учебных и исследовательских целей.")

# --------------------------------------------------
# PAGE 2 – Theory
# --------------------------------------------------
elif page == "📘 Теория полярных кодов":
    st.title("Теория: как работают полярные коды")

    st.markdown("""
    ## Идея полярных кодов
    Полярные коды основаны на явлении **поляризации каналов**:
    после специального линейного преобразования часть подканалов
    становится очень надёжной, а часть — почти полностью зашумлённой.

    Информационные биты передаются **только по надёжным подканалам**,
    остальные фиксируются (frozen bits).
    """)

    st.markdown("""
    ## Кодирование
    - длина кода: `n = 2^m`
    - кодовое слово:
    ```
    x = u · G_n
    ```
    - на практике реализуется через **бабочковую структуру**
    """)

    st.markdown("""
    ## Декодирование (SCL)
    Алгоритм SCL поддерживает **список из L кандидатов**.
    - при каждом информационном бите путь ветвится
    - сохраняются L наиболее вероятных путей
    - при L=1 → обычный SC-декодер
    """)

    st.info("Увеличение L повышает качество декодирования, но увеличивает сложность.")

# --------------------------------------------------
# PAGE 3 – FER Simulator
# --------------------------------------------------
elif page == "⚙️ Симулятор (FER)":
    st.title("Симуляция FER(Eb/N0)")

    col1, col2 = st.columns(2)

    with col1:
        # n = st.selectbox("Длина кода n", [128, 256, 512])
        # rate = st.selectbox("Скорость кода R", [1/3, 1/2, 2/3])
        # L = st.selectbox("Размер списка SCL (L)", [2, 4, 8, 16])
        # Ls = st.multiselect(
        #     "Размер списка SCL (L)",
        #     [2, 4, 8, 16],
        #     default=[4]
        # )
        # # вместо единичного селекта:
        # n = st.selectbox("n", [128,256,512], index=0)
        
        # добавить множественный выбор:
        ns = st.multiselect("Длины кода n (выберите 1 или несколько)", [128, 256, 512], default=[128])
        rates = st.multiselect("Скорости R (выберите 1 или несколько)", [1/3, 1/2, 2/3], format_func=lambda x: f"{x:.3f}", default=[1/3, 1/2])
        Ls = st.multiselect("Размер списка L (выберите 1 или несколько)", [4, 8, 16], default=[4, 8])


    with col2:
        # mode = st.radio("Тип передаваемых данных", ["zero", "random"])
        modes = st.multiselect("Режим передачи (mode)", ["zero", "random"], default=["zero"])
        snr_min = st.slider("Eb/N0 min (dB)", 0.0, 5.0, 0.5)
        snr_max = st.slider("Eb/N0 max (dB)", 1.0, 8.0, 4.0)
        snr_step = st.selectbox("Шаг Eb/N0", [0.25, 0.5, 1.0])
        use_nr_sequence = st.checkbox(
            "Использовать rank.csv",
            value=True
        )


    stop_errors = st.number_input("Минимум ошибок (FER)", value=10) # 3, 10, 100, 30
    max_frames = st.number_input("Максимум кадров (для FER)", value=200_000) # 500, 10000, 200_000, 300_000

    if st.button("🚀 Запустить симуляцию"):
        snrs = np.arange(snr_min, snr_max + 1e-9, snr_step)

        # with st.spinner("Моделирование..."):
        #     rows = run_one_case(
        #         n=n,
        #         rate=rate,
        #         L=L,
        #         mode=mode,
        #         ebn0_range=snrs,
        #         stop_errors=stop_errors,
        #         max_frames=300_000
        #     )
        # st.success("Симуляция завершена")

        # snr_vals = [r[0] for r in rows]
        # fer_vals = [r[3] for r in rows]

        # fig, ax = plt.subplots()
        # ax.semilogy(snr_vals, fer_vals, marker="o")
        # ax.set_xlabel("Eb/N0 (dB)")
        # ax.set_ylabel("FER")
        # ax.grid(True, which="both")
        # ax.set_title(f"FER: n={n}, R={rate}, L={L}")

        # st.pyplot(fig)



        with st.spinner("Моделирование..."):            
            # fig, ax = plt.subplots()
            yield_mode = True
            for L in Ls:
                # progress_bar = 0
                # progress = st.progress(0)
                snr_vals = []
                fer_vals = []
                plot_area = st.empty()
                table_area = st.empty()
                log_rows = []
                    

                # подготовка списков (если пользователь ничего не выбрал — задать defaults)
                if not ns:
                    ns = [128]
                if not rates:
                    rates = [1/2]
                if not Ls:
                    Ls = [4]
                if not modes:
                    modes = ["zero"]
                
                # создаём список всех комбинаций, которые нужно прогнать
                combos = list(itertools.product(ns, rates, Ls, modes))
                # total_cases = len(combos)
                progress_bar = st.progress(0.0)
                progress_text = st.empty()
                
                # предупреждение о времени
                max_cases = len(combos)
                st.info(f"Будет запущено {max_cases} кейсов (каждый кейс — одна кривa). Это может занять значительное время. ")
                
                # структура для хранения результатов для отрисовки
                all_series = OrderedDict()  # ключ -> (snr_list, fer_list)

                # итерация по всем комбинациям
                case_idx = 0
                for (n_val, r_val, L_val, mode_val) in combos:
                    label = f"n={n_val} R={r_val:.3f} L={L_val} mode={mode_val}"
                    case_idx += 1
                    progress_text.write(f"Выполняется кейс {case_idx} из {max_cases}")
                    progress_bar.progress(case_idx / max_cases)

                    st.write(f"Запуск: {label}")
                    snr_list = []
                    fer_list = []
                    frames_list = []
                    errors_list = []
                
                    gen = run_one_case(
                        n=n_val,
                        rate=r_val,
                        L=L_val,
                        mode=mode_val,
                        ebn0_range=snrs,
                        stop_errors=stop_errors,  # используем переменные из интерфейса
                        max_frames=max_frames,
                        yield_mode=True,
                        save_results=False,       # сохранение сделаем вручную после завершения кейса
                        use_nr_sequence=use_nr_sequence  # предположим, что checkbox у тебя уже есть
                    )
                
                    # собираем точки по мере появления и обновляем общий график
                    point_idx = 0
                    try:
                        for eb, fer, n_frames, n_errors in gen:
                            point_idx += 1
                            snr_list.append(float(eb))
                            fer_list.append(float(fer))
                            frames_list.append(int(n_frames))
                            errors_list.append(int(n_errors))
                
                            # обновляем общий словарь перед отрисовкой
                            all_series[label] = (snr_list.copy(), fer_list.copy())
                
                            # строим общий график (обновляем placeholder)
                            fig, ax = plt.subplots()
                            for lab, (xs, ys) in all_series.items():
                                ax.semilogy(xs, ys, marker='o', label=lab)
                            ax.set_xlabel("Eb/N0 (dB)")
                            ax.set_ylabel("FER")
                            ax.grid(True, which="both")
                            ax.legend(fontsize='small')
                            plot_area.pyplot(fig)
                            plt.close(fig)
                            # table_area.table({
                            #     "SNR": snr_vals,
                            #     "FER": fer_vals,
                            # })
                            # progress.progress((progress_bar + 1) / len(snrs))
                            # progress_bar += 1
                
                            # опционально: печатать текущую точку
                            st.write(f"  {label} point {point_idx}: Eb/N0={eb:.2f} dB, frames={n_frames}, errors={n_errors}, FER={fer:.3e}")
                            log_rows.append({
                                "n": n_val,
                                "R": round(r_val, 3),
                                "L": L_val,
                                "mode": mode_val,
                                "Eb/N0 (dB)": round(float(eb), 2),
                                "Frames": n_frames,
                                "Errors": n_errors,
                                "FER": float(fer)
                            })
                            
                            log_df = pd.DataFrame(log_rows)
                            table_area.dataframe(log_df, use_container_width=True, hide_index=True)

                        # после завершения генератора сохраняем CSV+PNG (как раньше)
                        # используем вспомогательную функцию save_results_csv/plot_fer_snr если есть
                        # здесь простая запись CSV и PNG:
                        # CSV
                        out_csv = Path("polar_project/results_st") / f"res_n{n_val}_R{int(r_val*100)}_L{L_val}_mode_{mode_val}.csv"
                        out_png = Path("polar_project/results_st") / f"fer_n{n_val}_R{int(r_val*100)}_L{L_val}_mode_{mode_val}.png"
                        out_csv.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_csv, "w") as f:
                            f.write("EbN0_dB,N_frames,N_errors,FER\n")
                            for x, fr, er, y in zip(snr_list, frames_list, errors_list, fer_list):
                                f.write(f"{x},{fr},{er},{y}\n")
                        # final per-case PNG (single-series)
                        fig, ax = plt.subplots()
                        ax.semilogy(snr_list, fer_list, marker='o', label=label)
                        ax.set_xlabel("Eb/N0 (dB)")
                        ax.set_ylabel("FER")
                        ax.grid(True, which="both")
                        ax.legend()
                        fig.savefig(out_png, dpi=200)
                        plt.close(fig)
                        st.write(f"Сохранено: {out_csv} и {out_png}")
                
                    except KeyboardInterrupt:
                        st.error("Прервано пользователем; частичные результаты сохранены (если были точки).")
                        # при необходимости — сохранить частичные результаты так же, как выше
                        raise

                    # for eb, fer, n_frames, n_errors in run_one_case(
                    #         n=n,
                    #         rate=rate,
                    #         L=L,
                    #         mode=mode,
                    #         ebn0_range=snrs,
                    #         stop_errors=stop_errors,
                    #         max_frames=max_frames,
                    #         yield_mode=True,
                    #         save_results=False
                    # ):
                    #     snr_vals.append(eb)
                    #     fer_vals.append(fer)
                    
                    #     # обновление графика
                    #     fig, ax = plt.subplots()
                    #     ax.semilogy(snr_vals, fer_vals, marker="o")
                    #     ax.set_xlabel("Eb/N0 (dB)")
                    #     ax.set_ylabel("FER")
                    #     ax.grid(True, which="both")
                    #     plot_area.pyplot(fig)
                    #     plt.close(fig)
                    #     table_area.table({
                    #         "SNR": snr_vals,
                    #         "FER": fer_vals,
                    #     })
                    #     progress.progress((progress_bar + 1) / len(snrs))
                    #     progress_bar += 1


        st.success("Симуляция завершена")
        print("\nSimulation ended successfully\n")

        # st.write(n, rate, L, mode)
        # if st.button("Сохранить результаты"):
        #     outdir = Path('polar_project/st_results')
        #     outdir.mkdir(parents=True, exist_ok=True)
        #     csvname = outdir / f'res_n{n}_R{int(rate*100) }_L{L}_mode_{mode}.csv'
        #     save_results_csv(csvname, zip(snr_vals, fer_vals), header=header)
        #     # plot
        #     plot_fer_snr([(f"L={L} mode={mode}", snr_vals, fer_vals)], outdir / f'fer_n{n}_R{int(rate*100)}_L{L}_mode_{mode}.png',
        #                  title=f"n={n} R={rate:.3f} L={L} mode={mode}")
        #     st.success("Результаты сохранены в папку st_results")


# --------------------------------------------------
# PAGE 4 – One frame demo
# --------------------------------------------------
elif page == "🔍 Пример одного кадра":
    st.title("Пошаговый пример: один кадр")

    n = st.selectbox("n", [128, 256])
    rate = st.selectbox("R", [1/3, 1/2])
    L = st.selectbox("L", [4, 8, 16])
    ebn0 = st.slider("Eb/N0 (dB)", 0.0, 10.0, 4.0)

    if st.button("▶️ Запустить пример"):
        k = int(n * rate)
        frozen = make_frozen_mask(n, k)

        info = np.random.randint(0, 2, k)
        x, u = encode(info, frozen)

        r, sigma2 = awgn_channel(x, ebn0, rate)
        llr = llr_from_awgn(r, sigma2)

        u_hat = scl_decode_naive(llr, frozen, L=L)
        info_pos = np.where(~frozen)[0][:k]
        info_hat = u_hat[info_pos]

        st.subheader("Результат")
        st.write("Совпадение:", bool(np.all(info == info_hat)))

        st.markdown("**Первые 20 бит:**")
        st.code(f"info     = {info[:20]}")
        st.code(f"decoded  = {info_hat[:20]}")
