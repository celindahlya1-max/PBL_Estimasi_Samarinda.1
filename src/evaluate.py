# -*- coding: utf-8 -*-
"""Fase 3-5: validasi, perbandingan model, interpretasi, dan model akhir.

Menjalankan:  python -m src.evaluate

Skema validasi
--------------
1. Utama   : leave-one-region-out (10 lipatan; 8 tahun satu wilayah jadi uji).
2. Tambahan: pembagian waktu, latih 2017-2022 dan uji 2023-2024 dengan spatial
   lag baris uji dihitung dari P0 tetangga tahun 2022.

Model yang dibandingkan: baseline rata-rata, MLR tanpa PCA, PCA+MLR tanpa
spatial lag, dan model utama PCA + spatial lag + MLR. Seluruh langkah
(standardisasi, PCA, pemilihan k, spatial lag) hanya di-fit pada data latih.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Agar skrip ini tetap jalan walau dijalankan langsung (python src\evaluate.py)
# maupun sebagai modul (python -m src.evaluate).
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src import data as mod_data  # noqa: E402
from src import geo as mod_geo  # noqa: E402
from src import interpret as mod_tafsir  # noqa: E402
from src import knn as mod_knn  # noqa: E402
from src import model as mod_model  # noqa: E402
from src import plots as mod_plot  # noqa: E402

URUT_MODEL = ["baseline", "mlr_tanpa_pca", "pca_tanpa_lag", "utama"]


def _catat(baris: list, skema: str, lipatan: str, subset: pd.DataFrame,
           nama_model: str, estimasi: np.ndarray) -> None:
    """Simpan satu baris hasil estimasi ke daftar hasil."""
    for (_, baris_data), nilai in zip(subset.iterrows(), estimasi):
        baris.append({
            "Skema": skema,
            "Lipatan": lipatan,
            "Wilayah": baris_data[cfg.KOLOM_WILAYAH],
            "Tahun": int(baris_data[cfg.KOLOM_TAHUN]),
            "Aktual": round(float(baris_data[cfg.TARGET]), 4),
            "Model": nama_model,
            "Estimasi": round(float(nilai), 4),
            "Galat": round(float(baris_data[cfg.TARGET]) - float(nilai), 4),
        })


def validasi_loro(data: pd.DataFrame, jarak: pd.DataFrame,
                  kolom_fitur: list[str] | None = None,
                  skema: str = "LORO",
                  catat_detail: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validasi leave-one-region-out untuk keempat model.

    Returns
    -------
    (tabel_hasil, tabel_detail)
        ``tabel_hasil`` satu baris per wilayah-tahun-model; ``tabel_detail``
        mencatat m komponen dan k terpilih pada tiap lipatan.
    """
    kolom_fitur = list(kolom_fitur or cfg.FITUR)
    baris: list[dict] = []
    detail: list[dict] = []

    for wilayah_uji in cfg.WILAYAH:
        latih = data[data[cfg.KOLOM_WILAYAH] != wilayah_uji]
        uji = data[data[cfg.KOLOM_WILAYAH] == wilayah_uji]

        # pilih k memakai validasi dalam pada data latih saja
        k_utama, tabel_k = mod_model.pilih_k_inner(latih, jarak,
                                                   kolom_fitur=kolom_fitur)
        k_mlr, _ = mod_model.pilih_k_inner(latih, jarak, pakai_pca=False,
                                           kolom_fitur=kolom_fitur)

        baseline = np.full(len(uji), float(latih[cfg.TARGET].mean()))
        _catat(baris, skema, wilayah_uji, uji, "baseline", baseline)

        m_mlr = mod_model.EstimatorP0(k=k_mlr, pakai_pca=False, pakai_lag=True,
                                      kolom_fitur=kolom_fitur).fit(latih, jarak)
        _catat(baris, skema, wilayah_uji, uji, "mlr_tanpa_pca", m_mlr.predict(uji))

        m_nolag = mod_model.EstimatorP0(pakai_pca=True, pakai_lag=False,
                                        kolom_fitur=kolom_fitur).fit(latih, jarak)
        _catat(baris, skema, wilayah_uji, uji, "pca_tanpa_lag", m_nolag.predict(uji))

        m_utama = mod_model.EstimatorP0(k=k_utama, pakai_pca=True, pakai_lag=True,
                                        kolom_fitur=kolom_fitur).fit(latih, jarak)
        _catat(baris, skema, wilayah_uji, uji, "utama", m_utama.predict(uji))

        if catat_detail:
            detail.append({
                "Skema": skema,
                "Lipatan": wilayah_uji,
                "k_utama": k_utama,
                "k_mlr": k_mlr,
                "m_utama": m_utama.m,
                "varians_kumulatif_utama": round(
                    float(m_utama.varians_kumulatif[-1]), 4),
                "m_pca_tanpa_lag": m_nolag.m,
                "RMSE_inner_k": round(float(tabel_k.iloc[0]["RMSE_inner"]), 4),
                "kandidat_k": "; ".join(
                    f"k={int(r.k)}:{r.RMSE_inner:.4f}" for r in tabel_k.itertuples()),
            })

    return pd.DataFrame(baris), pd.DataFrame(detail)


def validasi_waktu(data: pd.DataFrame, jarak: pd.DataFrame,
                   kolom_fitur: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Validasi pembagian waktu: latih 2017-2022, uji 2023-2024.

    Spatial lag baris uji memakai P0 tetangga pada tahun latih terakhir (2022)
    supaya label periode uji tidak dipakai sama sekali.
    """
    kolom_fitur = list(kolom_fitur or cfg.FITUR)
    latih = data[data[cfg.KOLOM_TAHUN].isin(cfg.TAHUN_LATIH_WAKTU)]
    uji = data[data[cfg.KOLOM_TAHUN].isin(cfg.TAHUN_UJI_WAKTU)]
    tahun_lag = cfg.TAHUN_ACUAN_LAG_WAKTU

    k_utama, tabel_k = mod_model.pilih_k_inner(latih, jarak, kolom_fitur=kolom_fitur)
    k_mlr, _ = mod_model.pilih_k_inner(latih, jarak, pakai_pca=False,
                                       kolom_fitur=kolom_fitur)
    baris: list[dict] = []

    baseline = np.full(len(uji), float(latih[cfg.TARGET].mean()))
    _catat(baris, "Waktu", "2023-2024", uji, "baseline", baseline)

    m_mlr = mod_model.EstimatorP0(k=k_mlr, pakai_pca=False, pakai_lag=True,
                                  kolom_fitur=kolom_fitur).fit(latih, jarak)
    _catat(baris, "Waktu", "2023-2024", uji, "mlr_tanpa_pca",
           m_mlr.predict(uji, tahun_lag=tahun_lag))

    m_nolag = mod_model.EstimatorP0(pakai_pca=True, pakai_lag=False,
                                    kolom_fitur=kolom_fitur).fit(latih, jarak)
    _catat(baris, "Waktu", "2023-2024", uji, "pca_tanpa_lag", m_nolag.predict(uji))

    m_utama = mod_model.EstimatorP0(k=k_utama, pakai_pca=True, pakai_lag=True,
                                    kolom_fitur=kolom_fitur).fit(latih, jarak)
    _catat(baris, "Waktu", "2023-2024", uji, "utama",
           m_utama.predict(uji, tahun_lag=tahun_lag))

    detail = {
        "n_latih": int(len(latih)),
        "n_uji": int(len(uji)),
        "k_utama": k_utama,
        "k_mlr": k_mlr,
        "m_utama": m_utama.m,
        "tahun_lag": tahun_lag,
        "kandidat_k": "; ".join(
            f"k={int(r.k)}:{r.RMSE_inner:.4f}" for r in tabel_k.itertuples()),
    }
    return pd.DataFrame(baris), detail


def tabel_metrik(hasil: pd.DataFrame, kunci_gabung: list[str]) -> pd.DataFrame:
    """Metrik gabungan (pooled) per model untuk satu skema validasi."""
    baris = []
    for (skema, model), grup in hasil.groupby(kunci_gabung):
        m = mod_model.metrik_regresi(grup["Aktual"], grup["Estimasi"])
        baris.append({
            "Skema": skema,
            "Model": model,
            "Label": cfg.MODEL.get(model, model),
            "n": m["n"],
            "R2": round(m["R2"], 4),
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "MAPE_persen": round(m["MAPE"], 2),
        })
    return pd.DataFrame(baris).sort_values(["Skema", "RMSE"]).reset_index(drop=True)


def metrik_per_wilayah(hasil: pd.DataFrame) -> pd.DataFrame:
    """Metrik per lipatan/wilayah (untuk LORO dan pembagian waktu)."""
    baris = []
    for (skema, lipatan, model), grup in hasil.groupby(["Skema", "Lipatan", "Model"]):
        m = mod_model.metrik_regresi(grup["Aktual"], grup["Estimasi"])
        baris.append({
            "Skema": skema, "Lipatan": lipatan, "Model": model,
            "n": m["n"], "R2": round(m["R2"], 4), "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
        })
    return pd.DataFrame(baris)


def metrik_per_tahun(hasil: pd.DataFrame) -> pd.DataFrame:
    """Metrik pooled per tahun untuk model utama."""
    baris = []
    sub = hasil[hasil["Model"] == "utama"]
    for (skema, tahun), grup in sub.groupby(["Skema", "Tahun"]):
        m = mod_model.metrik_regresi(grup["Aktual"], grup["Estimasi"])
        baris.append({"Skema": skema, "Tahun": int(tahun), "n": m["n"],
                      "R2": round(m["R2"], 4), "MAE": round(m["MAE"], 4),
                      "RMSE": round(m["RMSE"], 4)})
    return pd.DataFrame(baris)



def tabel_lebar_estimasi(hasil: pd.DataFrame, skema: str) -> pd.DataFrame:
    """Tabel lebar: satu baris per wilayah-tahun, satu kolom per model."""
    sub = hasil[hasil["Skema"] == skema]
    lebar = sub.pivot_table(index=["Wilayah", "Tahun"], columns="Model",
                            values="Estimasi").reset_index()
    lebar.columns = ["Wilayah", "Tahun"] + [
        f"Estimasi_{k}" for k in lebar.columns[2:]]
    aktual = (sub.drop_duplicates(["Wilayah", "Tahun"])[["Wilayah", "Tahun", "Aktual"]])
    return aktual.merge(lebar, on=["Wilayah", "Tahun"], how="left")


def analisis_sensitivitas(data: pd.DataFrame, jarak: pd.DataFrame,
                          metrik_utama: pd.DataFrame) -> pd.DataFrame:
    """LORO model utama tanpa fitur Jumlah_Penduduk_Ribu (penyebut rumus P0)."""
    print("\nAnalisis sensitivitas: model utama tanpa Jumlah_Penduduk_Ribu "
          f"({len(cfg.FITUR_TANPA_PENDUDUK)} fitur)")
    hasil, _ = validasi_loro(data, jarak, kolom_fitur=cfg.FITUR_TANPA_PENDUDUK,
                             skema="LORO_tanpa_penduduk", catat_detail=False)
    sub = hasil[hasil["Model"] == "utama"]
    m = mod_model.metrik_regresi(sub["Aktual"], sub["Estimasi"])
    acuan = metrik_utama[(metrik_utama["Skema"] == "LORO")
                         & (metrik_utama["Model"] == "utama")].iloc[0]
    return pd.DataFrame([
        {"Varian": "Model utama (12 fitur)",
         "Jumlah_fitur": len(cfg.FITUR),
         "R2": acuan["R2"], "MAE": acuan["MAE"], "RMSE": acuan["RMSE"]},
        {"Varian": "Tanpa Jumlah_Penduduk_Ribu (11 fitur)",
         "Jumlah_fitur": len(cfg.FITUR_TANPA_PENDUDUK),
         "R2": round(m["R2"], 4), "MAE": round(m["MAE"], 4),
         "RMSE": round(m["RMSE"], 4)},
    ]).assign(
        Selisih_RMSE=lambda d: (d["RMSE"] - d["RMSE"].iloc[0]).round(4),
        Selisih_MAE=lambda d: (d["MAE"] - d["MAE"].iloc[0]).round(4),
    )


def cetak_tabel(judul: str, tabel: pd.DataFrame, maks: int = 40) -> None:
    """Cetak tabel dengan judul; potong bila terlalu panjang."""
    print(f"\n{judul}")
    print(tabel.head(maks).to_string(index=False))
    if len(tabel) > maks:
        print(f"  ... ({len(tabel) - maks} baris lain tersimpan di berkas)")


def main() -> int:
    """Titik masuk Fase 3-5."""
    cfg.FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg.FOLDER_MODEL.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("FASE 3-5 - VALIDASI, INTERPRETASI, DAN MODEL AKHIR")
    print("=" * 78)

    panel = mod_data.muat_panel()
    data = mod_data.siapkan_data_model(panel)
    centroid = mod_geo.muat_centroid()
    jarak = mod_geo.matriks_jarak_haversine(centroid)
    print(f"Data pemodelan: {len(data)} baris, {data[cfg.KOLOM_WILAYAH].nunique()} "
          f"wilayah, tahun {int(data[cfg.KOLOM_TAHUN].min())}-"
          f"{int(data[cfg.KOLOM_TAHUN].max())}")

    # ---------------------------------------------------------------- Fase 3
    print("\n" + "-" * 78)
    print("FASE 3a - VALIDASI LEAVE-ONE-REGION-OUT (10 lipatan)")
    print("-" * 78)
    hasil_loro, detail_loro = validasi_loro(data, jarak)
    print("Detail pemilihan m dan k per lipatan:")
    print(detail_loro.to_string(index=False))

    print("\n" + "-" * 78)
    print("FASE 3b - VALIDASI PEMBAGIAN WAKTU (latih 2017-2022, uji 2023-2024)")
    print("-" * 78)
    hasil_waktu, detail_waktu = validasi_waktu(data, jarak)
    print("Detail:", json.dumps(detail_waktu, ensure_ascii=False))

    # Pembanding KNN (regresi) dan klasifikasi kategori kemiskinan
    knn = mod_knn.jalankan_knn(data, jarak)
    hasil = pd.concat([hasil_loro, hasil_waktu, knn["regresi"]],
                        ignore_index=True)
    metrik = tabel_metrik(hasil, ["Skema", "Model"])
    metrik.to_csv(cfg.FOLDER_OUTPUT / "perbandingan_model.csv", index=False)
    cetak_tabel("Perbandingan model (metrik gabungan):", metrik)

    per_wilayah = metrik_per_wilayah(hasil)
    per_wilayah.to_csv(cfg.FOLDER_OUTPUT / "metrik_per_wilayah.csv", index=False)
    per_tahun = metrik_per_tahun(hasil)
    per_tahun.to_csv(cfg.FOLDER_OUTPUT / "metrik_per_tahun.csv", index=False)
    detail_loro.to_csv(cfg.FOLDER_OUTPUT / "detail_lipatan.csv", index=False)

    lebar = tabel_lebar_estimasi(hasil, "LORO")
    lebar.to_csv(cfg.FOLDER_OUTPUT / "estimasi_lipatan.csv", index=False)
    hasil.to_csv(cfg.FOLDER_OUTPUT / "estimasi_lipatan_panjang.csv", index=False)

    utama_loro = hasil[(hasil["Skema"] == "LORO") & (hasil["Model"] == "utama")]
    utama_waktu = hasil[(hasil["Skema"] == "Waktu") & (hasil["Model"] == "utama")]
    mod_plot.scatter_aktual_vs_estimasi(
        utama_loro["Aktual"], utama_loro["Estimasi"],
        "Aktual vs Estimasi P0 - Validasi Leave-One-Region-Out (model utama)",
        nama="plot_aktual_vs_estimasi_loro.png")
    mod_plot.scatter_aktual_vs_estimasi(
        utama_waktu["Aktual"], utama_waktu["Estimasi"],
        "Aktual vs Estimasi P0 - Uji 2023-2024 (latih 2017-2022, model utama)",
        nama="plot_aktual_vs_estimasi_waktu.png")
    mod_plot.bar_perbandingan(
        metrik[metrik["Skema"] == "LORO"][["Label", "RMSE"]].rename(
            columns={"Label": "Model"}),
        kolom="RMSE", nama="perbandingan_model_rmse_loro.png")
    mod_plot.bar_perbandingan(
        metrik[metrik["Skema"] == "Waktu"][["Label", "RMSE"]].rename(
            columns={"Label": "Model"}),
        kolom="RMSE", nama="perbandingan_model_rmse_waktu.png")

    print("\nMetrik per wilayah (LORO, model utama):")
    print(per_wilayah[(per_wilayah["Skema"] == "LORO")
                      & (per_wilayah["Model"] == "utama")]
          .round(4).to_string(index=False))

    # ---------------------------------------------------------- sensitivitas
    sensitivitas = analisis_sensitivitas(data, jarak, metrik)
    sensitivitas.to_csv(cfg.FOLDER_OUTPUT / "sensitivitas_tanpa_penduduk.csv",
                        index=False)
    cetak_tabel("Sensitivitas fitur Jumlah_Penduduk_Ribu:", sensitivitas)
    return lanjut_interpretasi(data, jarak, metrik, per_wilayah)



def lanjut_interpretasi(data: pd.DataFrame, jarak: pd.DataFrame,
                        metrik: pd.DataFrame,
                        per_wilayah: pd.DataFrame) -> int:
    """Fase 4 (interpretasi & uji asumsi) dan Fase 5 (model akhir)."""
    out = cfg.FOLDER_OUTPUT
    print("\n" + "-" * 78)
    print("FASE 4 - INTERPRETASI DAN UJI ASUMSI (model akhir, 80 baris)")
    print("-" * 78)
    model, k, tabel_k = mod_tafsir.latih_model_akhir(data, jarak)
    print(f"k terpilih (validasi dalam LORO): {k}")
    print(tabel_k.to_string(index=False))
    print(f"m komponen: {model.m} dengan variansi kumulatif "
          f"{float(model.varians_kumulatif[-1]):.4f}")

    varians = pd.DataFrame({
        "Komponen": [f"PC{i + 1}" for i in range(len(model.pca.explained_variance_ratio_))],
        "Eigenvalue": model.pca.explained_variance_.round(4),
        "Varians_dijelaskan": model.pca.explained_variance_ratio_.round(6),
        "Varians_kumulatif": np.cumsum(
            model.pca.explained_variance_ratio_).round(6),
        "Dipakai_dalam_model": [i < model.m for i in
                                range(len(model.pca.explained_variance_ratio_))],
    })
    varians.to_csv(out / "pca_varians_model_akhir.csv", index=False)
    cetak_tabel("Explained variance komponen utama:", varians)

    loadings = model.tabel_loadings()
    loadings.round(6).to_csv(out / "pca_loadings_model_akhir.csv")
    cetak_tabel("Tabel loading (m komponen dipakai):", loadings.round(3), maks=15)

    tafsir = mod_tafsir.interpretasi_komponen(loadings)
    tafsir["Varians_dijelaskan"] = model.varians_ratio[: model.m].round(6)
    tafsir.to_csv(out / "pca_interpretasi_komponen.csv", index=False)
    cetak_tabel("Interpretasi singkat komponen:", tafsir)

    desain, y = mod_tafsir.desain_final(model, data)
    koef = mod_tafsir.koefisien_statsmodels(desain, y)
    koef.to_csv(out / "koefisien_model_akhir.csv")
    cetak_tabel("Koefisien MLR model akhir (statsmodels OLS):", koef.round(4))

    asli = mod_tafsir.koefisien_ke_variabel_asli(model)
    asli["Koefisien_Lag"] = np.nan
    asli = pd.concat([asli, pd.DataFrame([{
        "Variabel": "SpatialLag", "Label": "Spatial lag (rata-rata P0 tetangga)",
        "Koefisien_efektif": np.nan, "Arah": "", "Koefisien_Lag": model.koef[-1],
    }])], ignore_index=True)
    asli.to_csv(out / "koefisien_variabel_asli.csv", index=False)
    cetak_tabel("Koefisien dikembalikan ke variabel asli (loading x koefisien):",
                asli.round(4), maks=20)

    vif_mentah = mod_tafsir.vif_tabel(data[cfg.FITUR])
    vif_mentah.to_csv(out / "vif_fitur_mentah.csv", index=False)
    vif_komponen = mod_tafsir.vif_tabel(desain.drop(columns="intersep"))
    vif_komponen.to_csv(out / "vif_komponen.csv", index=False)
    cetak_tabel("VIF 12 fitur mentah:", vif_mentah, maks=15)
    cetak_tabel("VIF komponen PCA + spatial lag:", vif_komponen, maks=15)

    uji = mod_tafsir.uji_asumsi(model, data, desain, y)
    uji.to_csv(out / "uji_asumsi.csv", index=False)
    cetak_tabel("Uji asumsi klasik:", uji)

    robust = mod_tafsir.galat_baku_robust(model, data, desain, y)
    robust.to_csv(out / "galat_baku_robust.csv")
    cetak_tabel("Galat baku robust berkelompok per wilayah:", robust.round(4))

    mod_tafsir.simpan_grafik_interpretasi(model, data)
    print("\nGrafik tersimpan: pca_scree_plot.png, pca_loading_heatmap.png, "
          "plot_residual_model_akhir.png")

    print("\n" + "-" * 78)
    print("FASE 5 - MODEL AKHIR DAN TABEL ESTIMASI")
    print("-" * 78)
    metadata = mod_tafsir.simpan_artefak_model(model, k, tabel_k, data, jarak)
    print("Artefak model tersimpan di models/: model_akhir.joblib, "
          "koefisien_mlr.csv, loadings_pca.csv, centroid_wilayah.csv, "
          "metadata.json, daftar_tetangga.csv")

    estimasi = mod_tafsir.tabel_estimasi(model, data)
    estimasi.to_csv(out / "estimasi.csv", index=False)
    m_dalam = mod_model.metrik_regresi(estimasi["P0_Aktual"],
                                       estimasi["P0_Estimasi"])
    print(f"\nMetrik dalam sampel (80 baris): R2={m_dalam['R2']:.4f}, "
          f"MAE={m_dalam['MAE']:.4f}, RMSE={m_dalam['RMSE']:.4f}")
    cetak_tabel("Estimasi P0 per wilayah-tahun (10 baris pertama):",
                estimasi.head(10).round(4))

    utama = metrik[(metrik["Skema"] == "LORO") & (metrik["Model"] == "utama")].iloc[0]
    print("\n" + "=" * 78)
    print("RINGKASAN JUJUR")
    print("=" * 78)
    print(f"Model utama (LORO)      : R2={utama['R2']:.4f}, "
          f"MAE={utama['MAE']:.4f}, RMSE={utama['RMSE']:.4f}")
    pembanding = metrik[metrik["Skema"] == "LORO"].sort_values("RMSE")
    print("Peringkat model (LORO, RMSE kecil = baik):")
    print(pembanding[["Label", "R2", "MAE", "RMSE"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

