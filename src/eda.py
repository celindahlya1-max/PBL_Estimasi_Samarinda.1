# -*- coding: utf-8 -*-
"""Fase 1 - persiapan data, EDA, dan penyiapan informasi geospasial.

Menjalankan:  python -m src.eda

Hasil (semua ke ``outputs/``):
- kelengkapan_data.csv, eda_deskriptif.csv, eda_p0_lebar.csv
- eda_korelasi_heatmap.csv / .png, eda_vif_12fitur.csv
- eda_tren_p0.png, eda_boxplot_pencilan.png
- pemetaan_nama_wilayah.csv, centroid_wilayah.csv, jarak_antar_wilayah.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Agar skrip ini tetap jalan walau dijalankan langsung (python src\eda.py)
# maupun sebagai modul (python -m src.eda).
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src import data as mod_data  # noqa: E402
from src import geo as mod_geo  # noqa: E402
from src import plots as mod_plot  # noqa: E402


def tampilkan_judul(teks: str) -> None:
    """Cetak judul bagian dengan garis pemisah."""
    print("\n" + "=" * 78)
    print(teks)
    print("=" * 78)


def statistik_deskriptif(data: pd.DataFrame) -> pd.DataFrame:
    """Statistik deskriptif 12 fitur + P0 dan tulis ke CSV."""
    kolom = cfg.FITUR + [cfg.TARGET]
    deskriptif = data[kolom].describe().T
    deskriptif["variasi_koef"] = (deskriptif["std"] / deskriptif["mean"]).round(4)
    deskriptif = deskriptif.round(4)
    deskriptif.to_csv(cfg.FOLDER_OUTPUT / "eda_deskriptif.csv")
    print("Statistik deskriptif (80 baris, 2017-2024):")
    print(deskriptif.to_string())
    return deskriptif


def hitung_vif(data: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    """VIF tiap fitur; nilai > 10 menandakan multikolinearitas tinggi."""
    x = data[kolom].to_numpy(dtype=float)
    nilai = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    tabel = pd.DataFrame({"Fitur": kolom, "VIF": np.round(nilai, 3)})
    tabel["Keterangan"] = np.where(
        tabel["VIF"] > 10, "multikolinearitas tinggi",
        np.where(tabel["VIF"] > 5, "perlu diperhatikan", "aman"))
    return tabel.sort_values("VIF", ascending=False).reset_index(drop=True)


def korelasi_tinggi(data: pd.DataFrame, kolom: list[str],
                    ambang: float = 0.8) -> pd.DataFrame:
    """Daftar pasangan fitur dengan |korelasi| >= ambang."""
    korelasi = data[kolom].corr().abs()
    baris = []
    for i, a in enumerate(kolom):
        for b in kolom[i + 1:]:
            nilai = float(korelasi.loc[a, b])
            if nilai >= ambang:
                baris.append({"Fitur_A": a, "Fitur_B": b,
                              "Korelasi_abs": round(nilai, 3)})
    return pd.DataFrame(baris).sort_values("Korelasi_abs", ascending=False)


def analisis_kelengkapan(panel: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan kelengkapan sel per tahun dan tulis ke CSV."""
    ringkas = mod_data.ringkasan_data_lengkap(panel)
    ringkas.to_csv(cfg.FOLDER_OUTPUT / "kelengkapan_data.csv", index=False)
    print("Kelengkapan sel per tahun:")
    print(ringkas.to_string(index=False))
    return ringkas


def bagian_eda(data: pd.DataFrame) -> None:
    """Jalankan seluruh langkah EDA dan simpan berkasnya."""
    tampilkan_judul("EDA: STATISTIK DESKRIPTIF")
    statistik_deskriptif(data)

    tampilkan_judul("EDA: KORELASI DAN MULTIKOLINEARITAS")
    kolom_korelasi = cfg.FITUR + [cfg.TARGET]
    data[kolom_korelasi].corr().round(4).to_csv(
        cfg.FOLDER_OUTPUT / "eda_korelasi_heatmap.csv")
    mod_plot.heatmap_korelasi(data, kolom_korelasi)
    print("Heatmap korelasi tersimpan: eda_korelasi_heatmap.png")

    tinggi = korelasi_tinggi(data, cfg.FITUR)
    print("\nPasangan fitur dengan |korelasi| >= 0,80 (indikasi multikolinearitas):")
    print(tinggi.to_string(index=False) if len(tinggi) else "  (tidak ada)")

    vif = hitung_vif(data, cfg.FITUR)
    vif.to_csv(cfg.FOLDER_OUTPUT / "eda_vif_12fitur.csv", index=False)
    print("\nVIF 12 fitur:")
    print(vif.to_string(index=False))

    tampilkan_judul("EDA: TREN P0 DAN PENCILAN")
    lebar = mod_data.tabel_lebar(data, cfg.TARGET)
    lebar.round(4).to_csv(cfg.FOLDER_OUTPUT / "eda_p0_lebar.csv")
    print("P0 (%) per wilayah dan tahun:")
    print(lebar.round(4).to_string())
    mod_plot.tren_p0(data)
    mod_plot.boxplot_fitur(data)
    print("\nGrafik tersimpan: eda_tren_p0.png, eda_boxplot_pencilan.png")

    print("\nBaris yang ditandai secara khusus:")
    print("- Samarinda 2021 : Jumlah_Penduduk_Ribu dikoreksi 83,1 -> 831,0 ribu "
          "jiwa (salah ketik sumber BPS).")
    ppu = data[(data[cfg.KOLOM_WILAYAH] == "Penajam Paser Utara")
               & (data[cfg.KOLOM_TAHUN] == 2024)].iloc[0]
    print(f"- Penajam Paser Utara 2024: Laju_Pertumbuhan_Penduduk = "
          f"{ppu['Laju_Pertumbuhan_Penduduk']:.2f}% (pencilan, dipertahankan); "
          f"P0 = {ppu[cfg.TARGET]:.4f}%")


def bagian_geospasial() -> pd.DataFrame:
    """Hitung centroid, jarak, dan daftar tetangga; simpan ke outputs/."""
    tampilkan_judul("GEOSPASIAL: PEMETAAN NAMA, CENTROID, JARAK, TETANGGA")
    geometri = mod_geo.muat_geometri()
    pemetaan = mod_geo.tabel_pemetaan(geometri)
    pemetaan.to_csv(cfg.FOLDER_OUTPUT / "pemetaan_nama_wilayah.csv", index=False)
    print(f"Wilayah dikenal pada GeoJSON: {len(geometri)} dari 11 fitur "
          "(poligon tanpa nama dibuang)")
    print("Tabel pemetaan nama wilayah GeoJSON <-> dataset:")
    print(pemetaan.to_string(index=False))

    centroid = mod_geo.hitung_centroid(geometri)
    mod_geo.simpan_centroid(centroid)
    print("\nCentroid wilayah (dihitung pada EPSG:32750, diubah ke EPSG:4326):")
    print(centroid.to_string(index=False))

    jarak = mod_geo.matriks_jarak_haversine(centroid)
    jarak.round(2).to_csv(cfg.FOLDER_OUTPUT / "jarak_antar_wilayah.csv")
    pasangan = jarak.where(~np.eye(len(jarak), dtype=bool)).stack().sort_values()
    print("\nJarak haversine tersimpan: jarak_antar_wilayah.csv")
    print(f"Pasangan terdekat : {pasangan.index[0][0]} - {pasangan.index[0][1]} "
          f"({pasangan.iloc[0]:.1f} km)")
    print(f"Pasangan terjauh  : {pasangan.index[-1][0]} - {pasangan.index[-1][1]} "
          f"({pasangan.iloc[-1]:.1f} km)")

    for k in cfg.K_CANDIDATES:
        tetangga = mod_geo.daftar_tetangga_umum(centroid, k)
        print(f"\nTetangga terdekat k={k} (seluruh 10 wilayah):")
        print("  " + "; ".join(f"{w}: {', '.join(t)}" for w, t in tetangga.items()))
    return centroid



def main() -> int:
    """Titik masuk fase 1."""
    cfg.FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    tampilkan_judul("FASE 1 - PERSIAPAN DATA DAN EDA")
    panel = mod_data.muat_panel()
    laporan = mod_data.validasi_panel(panel)
    print("Laporan validasi panel:")
    for kunci, nilai in laporan.items():
        print(f"  {kunci:32s}: {nilai}")

    data = mod_data.siapkan_data_model(panel)
    print(f"\nData pemodelan siap         : {data.shape[0]} baris x {data.shape[1]} kolom")
    print(f"Wilayah                     : {data[cfg.KOLOM_WILAYAH].nunique()}")
    print(f"Tahun                       : {sorted(int(t) for t in data[cfg.KOLOM_TAHUN].unique())}")
    print(f"Sel kosong (fitur+target)   : "
          f"{int(data[cfg.FITUR + [cfg.TARGET]].isna().sum().sum())}")

    audit = mod_data.audit_target_p0(panel)
    audit.to_csv(cfg.FOLDER_OUTPUT / "audit_target_p0.csv", index=False)
    print("\nAudit target P0 (pembilang, penyebut, hasil hitung):")
    kolom_audit = ["Kabupaten_Kota", "Tahun",
                   "Pembilang_Jumlah_Penduduk_Miskin_Ribu",
                   "Penyebut_Jumlah_Penduduk_Ribu", "P0_hitung_ulang",
                   "P0_di_panel", "Selisih_poin"]
    print(audit[kolom_audit].head(10).to_string(index=False))
    selisih = pd.to_numeric(audit["Selisih_poin"], errors="coerce").abs().max()
    print(f"  Selisih absolut maksimum panel vs hitung ulang: {selisih:.6f} poin")
    print("  Berkas: outputs/audit_target_p0.csv (kolom P0_resmi_BPS disediakan "
          "untuk diisi angka resmi BPS).")

    analisis_kelengkapan(panel)
    bagian_eda(data)
    bagian_geospasial()

    print("\nCatatan mutu data:")
    print(mod_data.muat_catatan_mutu().to_string(index=False))
    print("\nFASE 1 SELESAI. Berkas hasil ada di folder outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

