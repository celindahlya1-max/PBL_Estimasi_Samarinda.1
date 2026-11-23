# -*- coding: utf-8 -*-
"""Fungsi grafik (matplotlib) untuk EDA, evaluasi, dan komponen laporan.

Semua fungsi menyimpan grafik ke folder ``outputs/`` dan mengembalikan Path.
Gaya grafik dibuat seragam agar konsisten di laporan dan dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # aman dijalankan tanpa layar (headless)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402

WARNA_UTAMA = "#1f4e79"
WARNA_AKSEN = "#c0392b"
WARNA_ABU = "#7f8c8d"
PALET = plt.get_cmap("tab10")


def siapkan_gaya() -> None:
    """Atur gaya global matplotlib (ukuran font, grid, latar)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _path(nama: str) -> Path:
    """Siapkan folder output dan kembalikan path tujuan grafik."""
    cfg.FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    return cfg.FOLDER_OUTPUT / nama


def simpan(fig, nama: str) -> Path:
    """Simpan gambar lalu tutup figure (mencegah kebocoran memori)."""
    tujuan = _path(nama)
    fig.savefig(tujuan, bbox_inches="tight")
    plt.close(fig)
    return tujuan


def heatmap_korelasi(data: pd.DataFrame, kolom: list[str],
                     nama: str = "eda_korelasi_heatmap.png") -> Path:
    """Heatmap matriks korelasi Pearson antar kolom numerik."""
    siapkan_gaya()
    korelasi = data[kolom].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    gambar = ax.imshow(korelasi, cmap="RdBu_r", vmin=-1, vmax=1)
    label = [cfg.LABEL_PENDEK.get(k, k) for k in kolom]
    ax.set_xticks(range(len(kolom)))
    ax.set_yticks(range(len(kolom)))
    ax.set_xticklabels(label, rotation=45, ha="right")
    ax.set_yticklabels(label)
    ax.grid(False)
    for i in range(len(kolom)):
        for j in range(len(kolom)):
            nilai = korelasi.iloc[i, j]
            ax.text(j, i, f"{nilai:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(nilai) > 0.55 else "black")
    fig.colorbar(gambar, ax=ax, shrink=0.8, label="Koefisien korelasi Pearson")
    ax.set_title("Matriks Korelasi Indikator Sosial-Ekonomi dan P0\n"
                 "Kabupaten/Kota Provinsi Kalimantan Timur, 2017-2024")
    return simpan(fig, nama)


def tren_p0(data: pd.DataFrame, nama: str = "eda_tren_p0.png") -> Path:
    """Garis tren P0 tiap wilayah per tahun."""
    siapkan_gaya()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, wilayah in enumerate(cfg.WILAYAH):
        sub = data[data[cfg.KOLOM_WILAYAH] == wilayah].sort_values(cfg.KOLOM_TAHUN)
        ax.plot(sub[cfg.KOLOM_TAHUN], sub[cfg.TARGET], marker="o", markersize=3.5,
                linewidth=1.6, color=PALET(i % 10),
                label=cfg.NAMA_TAMPILAN.get(wilayah, wilayah))
    ax.set_xlabel("Tahun")
    ax.set_ylabel("P0 - persentase penduduk miskin (%)")
    ax.set_title("Tren Tingkat Kemiskinan (P0) Menurut Kabupaten/Kota, 2016-2024\n"
                 "(garis putus-putus vertikal: awal periode pemodelan 2017)")
    ax.axvline(2016.5, color=WARNA_ABU, linestyle="--", linewidth=1)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_xticks(sorted(int(t) for t in data[cfg.KOLOM_TAHUN].unique()))
    return simpan(fig, nama)


def boxplot_fitur(data: pd.DataFrame, nama: str = "eda_boxplot_pencilan.png") -> Path:
    """Grid boxplot 12 fitur untuk menampilkan sebaran dan pencilan."""
    siapkan_gaya()
    nkol = 4
    nbar = int(np.ceil(len(cfg.FITUR) / nkol))
    fig, axes = plt.subplots(nbar, nkol, figsize=(13, 2.9 * nbar))
    axes = np.array(axes).reshape(-1)
    for ax, kolom in zip(axes, cfg.FITUR):
        ax.boxplot(data[kolom].dropna(), widths=0.5,
                   medianprops={"color": WARNA_AKSEN})
        titik = data.nlargest(1, kolom).iloc[0]
        ax.scatter([1], [titik[kolom]], color=WARNA_AKSEN, s=18, zorder=5)
        ax.annotate(f"{titik[cfg.KOLOM_WILAYAH]} {int(titik[cfg.KOLOM_TAHUN])}",
                    (1.12, titik[kolom]), fontsize=6.5, color=WARNA_AKSEN,
                    va="center")
        ax.set_title(cfg.LABEL_PENDEK.get(kolom, kolom), fontsize=9)
        ax.set_xticks([])
    for ax in axes[len(cfg.FITUR):]:
        ax.axis("off")
    fig.suptitle("Sebaran 12 Indikator Sosial-Ekonomi, 2017-2024 "
                 "(titik merah: nilai terbesar per indikator)", fontsize=11)
    return simpan(fig, nama)


def scree_plot(varians: np.ndarray, kumulatif: np.ndarray, m: int,
               nama: str = "pca_scree_plot.png") -> Path:
    """Scree plot + variansi kumulatif dan garis ambang 85%."""
    siapkan_gaya()
    komponen = np.arange(1, len(varians) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(komponen, varians * 100, color=WARNA_UTAMA, alpha=0.85)
    ax.set_xlabel("Komponen utama (PC)")
    ax.set_ylabel("Varians dijelaskan (%)", color=WARNA_UTAMA)
    ax.set_xticks(komponen)
    ax2 = ax.twinx()
    ax2.plot(komponen, kumulatif * 100, color=WARNA_AKSEN, marker="o",
             linewidth=1.8)
    ax2.set_ylabel("Varians kumulatif (%)", color=WARNA_AKSEN)
    ax2.axhline(cfg.PCA_VAR_THRESHOLD * 100, color=WARNA_ABU, linestyle="--",
                linewidth=1)
    ax2.annotate(f"Ambang {cfg.PCA_VAR_THRESHOLD:.0%}", (1, cfg.PCA_VAR_THRESHOLD * 100),
                 xytext=(1.1, cfg.PCA_VAR_THRESHOLD * 100 + 4), fontsize=8,
                 color=WARNA_ABU)
    for i, k in enumerate(kumulatif, start=1):
        ax2.annotate(f"{k*100:.1f}%", (i, k * 100), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=7,
                     color=WARNA_AKSEN)
    ax.set_title(f"Scree Plot dan Variansi Kumulatif (m = {m} komponen, "
                 f"kumulatif {kumulatif[m-1]*100:.1f}%)")
    ax.grid(False)
    ax2.grid(False)
    return simpan(fig, nama)


def heatmap_loading(loadings: pd.DataFrame,
                    nama: str = "pca_loading_heatmap.png") -> Path:
    """Heatmap matriks loading komponen utama."""
    siapkan_gaya()
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    nilai = loadings.to_numpy(dtype=float)
    batas = float(np.nanmax(np.abs(nilai))) or 1.0
    gambar = ax.imshow(nilai, cmap="RdBu_r", vmin=-batas, vmax=batas)
    ax.set_xticks(range(loadings.shape[1]))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticks(range(loadings.shape[0]))
    ax.set_yticklabels([cfg.LABEL_PENDEK.get(i, i) for i in loadings.index])
    ax.grid(False)
    for i in range(nilai.shape[0]):
        for j in range(nilai.shape[1]):
            ax.text(j, i, f"{nilai[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(nilai[i, j]) > 0.6 * batas else "black")
    fig.colorbar(gambar, ax=ax, shrink=0.85, label="Loading (korelasi variabel-PC)")
    ax.set_title("Matriks Loading Komponen Utama\n"
                 "(nilai besar = kontribusi indikator pada komponen)")
    return simpan(fig, nama)



def scatter_aktual_vs_estimasi(aktual, estimasi, judul: str,
                               nama: str = "plot_aktual_vs_estimasi.png") -> Path:
    """Scatter aktual vs estimasi dengan garis identitas y = x."""
    siapkan_gaya()
    aktual = np.asarray(aktual, dtype=float)
    estimasi = np.asarray(estimasi, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(aktual, estimasi, s=26, alpha=0.75, color=WARNA_UTAMA,
               edgecolor="white", linewidth=0.4)
    batas = [min(aktual.min(), estimasi.min()) - 0.5,
             max(aktual.max(), estimasi.max()) + 0.5]
    ax.plot(batas, batas, color=WARNA_AKSEN, linestyle="--", linewidth=1.2,
            label="Garis identitas (estimasi = aktual)")
    ax.set_xlim(batas)
    ax.set_ylim(batas)
    ax.set_xlabel("P0 aktual - hasil hitung (%)")
    ax.set_ylabel("P0 estimasi model (%)")
    ax.set_title(judul)
    ax.legend(fontsize=8, frameon=False)
    return simpan(fig, nama)


def plot_residual(nilai_cocok, residual, judul: str,
                  nama: str = "plot_residual.png") -> Path:
    """Grafik residual vs nilai cocok (pemeriksaan asumsi homoskedastisitas)."""
    siapkan_gaya()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(nilai_cocok, residual, s=24, alpha=0.75, color=WARNA_UTAMA,
               edgecolor="white", linewidth=0.4)
    ax.axhline(0, color=WARNA_AKSEN, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Nilai cocok (estimasi P0, %)")
    ax.set_ylabel("Residual (aktual - estimasi, poin persentase)")
    ax.set_title(judul)
    return simpan(fig, nama)


def bar_perbandingan(metrik: pd.DataFrame, kolom: str = "RMSE",
                     nama: str = "perbandingan_model.png") -> Path:
    """Diagram batang perbandingan satu metrik antar model."""
    siapkan_gaya()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    urut = metrik.sort_values(kolom)
    label = urut["Label"] if "Label" in urut.columns else urut["Model"]
    batang = ax.barh(label, urut[kolom], color=WARNA_UTAMA, alpha=0.85)
    for b, nilai in zip(batang, urut[kolom]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {nilai:.4f}",
                va="center", fontsize=8)
    ax.set_xlabel(f"{kolom} (semakin kecil semakin baik)")
    ax.set_title(f"Perbandingan Model Berdasarkan {kolom}")
    ax.invert_yaxis()
    return simpan(fig, nama)



def plot_confusion(matriks: pd.DataFrame, judul: str,
                   nama: str = "knn_confusion.png") -> Path:
    """Heatmap confusion matrix klasifikasi kategori kemiskinan."""
    siapkan_gaya()
    nilai = matriks.to_numpy(dtype=float)
    puncak = max(1.0, float(nilai.max()) if nilai.size else 1.0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    gambar = ax.imshow(nilai, cmap="Blues", vmin=0, vmax=puncak)
    ax.set_xticks(range(matriks.shape[1]))
    ax.set_xticklabels(list(matriks.columns))
    ax.set_yticks(range(matriks.shape[0]))
    ax.set_yticklabels(list(matriks.index))
    ax.grid(False)
    for i in range(nilai.shape[0]):
        for j in range(nilai.shape[1]):
            ax.text(j, i, f"{int(nilai[i, j])}", ha="center", va="center",
                    fontsize=11,
                    color="white" if nilai[i, j] > 0.6 * puncak else "black")
    total = float(nilai.sum())
    akurasi = float(np.trace(nilai) / total) if total else float("nan")
    ax.set_xlabel("Kelas prediksi")
    ax.set_ylabel("Kelas aktual")
    ax.set_title(f"{judul}\nAkurasi = {akurasi:.4f}")
    fig.colorbar(gambar, ax=ax, shrink=0.8, label="Jumlah observasi")
    return simpan(fig, nama)