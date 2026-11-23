# -*- coding: utf-8 -*-
"""Fase 4-5: interpretasi model dan penyimpanan model akhir.

Isi modul:
- koefisien MLR beserta koefisien yang dikembalikan ke variabel asli
  (loading x koefisien komponen),
- uji asumsi klasik pada data penuh 80 baris (Shapiro-Wilk, Breusch-Pagan,
  Durbin-Watson, VIF fitur mentah vs VIF komponen, galat baku robust
  berkelompok per wilayah),
- pelatihan model akhir, penyimpanan artefak ke ``models/``, dan tabel
  estimasi P0 per wilayah-tahun ke ``outputs/estimasi.csv``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from src import config as cfg
from src import model as mod_model
from src import plots as mod_plot


def vif_tabel(bingkai: pd.DataFrame) -> pd.DataFrame:
    """VIF tiap kolom pada matriks fitur mentah (tanpa intersep)."""
    x = bingkai.to_numpy(dtype=float)
    nilai = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    return (pd.DataFrame({"Fitur": bingkai.columns, "VIF": np.round(nilai, 3)})
            .sort_values("VIF", ascending=False).reset_index(drop=True))


def desain_final(model: mod_model.EstimatorP0, data: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Matriks desain model akhir (intersep + komponen + lag) dan targetnya."""
    z = model.skor_pca(data)
    lag = (model.lag_baris_uji(data) if model.pakai_lag
           else np.zeros(len(data)))
    nama = [f"PC{i + 1}" for i in range(model.m)] if model.pakai_pca \
        else list(model.kolom_fitur)
    kolom = ["intersep"] + nama + (["SpatialLag"] if model.pakai_lag else [])
    desain = pd.DataFrame(np.column_stack([np.ones(len(data)), z] + (
        [lag] if model.pakai_lag else [])), columns=kolom)
    return desain, data[cfg.TARGET].astype(float)


def koefisien_statsmodels(desain: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Tabel koefisien OLS lengkap (koefisien, galat baku, t, p)."""
    hasil = sm.OLS(y, desain).fit()
    tabel = pd.DataFrame({
        "Koefisien": hasil.params.round(6),
        "Galat_baku": hasil.bse.round(6),
        "t": hasil.tvalues.round(4),
        "p_value": hasil.pvalues.round(4),
    })
    tabel["Signifikan_5persen"] = tabel["p_value"] < 0.05
    return tabel


def koefisien_ke_variabel_asli(model: mod_model.EstimatorP0) -> pd.DataFrame:
    """Kembalikan koefisien komponen ke variabel asli (loading x koefisien)."""
    if not model.pakai_pca:
        raise ValueError("Model tanpa PCA tidak perlu dikembalikan")
    loading = model.tabel_loadings()
    koef_pc = np.array([model.koef[1 + i] for i in range(model.m)])  # lewati intersep
    kontribusi = loading.to_numpy() @ koef_pc
    tabel = pd.DataFrame({
        "Variabel": model.kolom_fitur,
        "Label": [cfg.LABEL_FITUR.get(v, v) for v in model.kolom_fitur],
        "Koefisien_efektif": np.round(kontribusi, 6),
    })
    tabel["Arah"] = np.where(tabel["Koefisien_efektif"] > 0, "positif terhadap P0",
                             "negatif terhadap P0")
    return tabel.sort_values("Koefisien_efektif").reset_index(drop=True)


def uji_asumsi(model: mod_model.EstimatorP0, data: pd.DataFrame,
               desain: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Uji asumsi klasik pada model akhir (data penuh 80 baris)."""
    hasil = sm.OLS(y, desain).fit()
    residual = hasil.resid
    shapiro = stats.shapiro(residual)
    bp = het_breuschpagan(residual, desain)
    dw = durbin_watson(residual)

    baris = [
        {"Uji": "Shapiro-Wilk (normalitas residual)",
         "Statistik": round(float(shapiro.statistic), 4),
         "p_value": round(float(shapiro.pvalue), 4),
         "Kesimpulan": ("residual normal (alpha 5%)" if shapiro.pvalue > 0.05
                        else "residual TIDAK normal (alpha 5%)")},
        {"Uji": "Breusch-Pagan (heteroskedastisitas)",
         "Statistik": round(float(bp[0]), 4),
         "p_value": round(float(bp[1]), 4),
         "Kesimpulan": ("ragam galat homogen (alpha 5%)" if bp[1] > 0.05
                        else "indikasi heteroskedastisitas (alpha 5%)")},
        {"Uji": "Durbin-Watson (autokorelasi)",
         "Statistik": round(float(dw), 4),
         "p_value": np.nan,
         "Kesimpulan": ("mendekati 2 sehingga tidak ada autokorelasi kuat"
                        if 1.5 <= dw <= 2.5 else "perlu diperhatikan")},
        {"Uji": "Uji F keseluruhan",
         "Statistik": round(float(hasil.fvalue), 4),
         "p_value": round(float(hasil.f_pvalue), 6),
         "Kesimpulan": ("model signifikan (alpha 5%)" if hasil.f_pvalue < 0.05
                        else "model tidak signifikan (alpha 5%)")},
    ]
    return pd.DataFrame(baris)


def galat_baku_robust(model: mod_model.EstimatorP0, data: pd.DataFrame,
                      desain: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Galat baku robust yang dikelompokkan per wilayah (panel tidak independen)."""
    hasil = sm.OLS(y, desain).fit(cov_type="cluster",
                                  cov_kwds={"groups": data[cfg.KOLOM_WILAYAH]})
    return pd.DataFrame({
        "Koefisien": hasil.params.round(6),
        "Galat_baku_robust": hasil.bse.round(6),
        "t_robust": hasil.tvalues.round(4),
        "p_value_robust": hasil.pvalues.round(4),
    })


def interpretasi_komponen(loadings: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Ringkasan verbal: indikator dengan loading terbesar tiap komponen."""
    baris = []
    for pc in loadings.columns:
        urut = loadings[pc].abs().sort_values(ascending=False)
        teratas = urut.index[:n]
        baris.append({
            "Komponen": pc,
            "Loading_dominan": "; ".join(
                f"{cfg.LABEL_PENDEK.get(v, v)} ({loadings.loc[v, pc]:+.2f})"
                for v in teratas),
            "Varians_dijelaskan": np.nan,
        })
    return pd.DataFrame(baris)


def latih_model_akhir(data: pd.DataFrame, jarak: pd.DataFrame,
                      kolom_fitur: list[str] | None = None
                      ) -> tuple[mod_model.EstimatorP0, int, pd.DataFrame]:
    """Latih model utama pada seluruh data; k dipilih dengan validasi dalam.

    Returns
    -------
    (model, k_terpilih, tabel_kandidat_k)
    """
    kolom_fitur = list(kolom_fitur or cfg.FITUR)
    k_terbaik, tabel_k = mod_model.pilih_k_inner(data, jarak,
                                                 kolom_fitur=kolom_fitur)
    model = mod_model.EstimatorP0(k=k_terbaik, pakai_pca=True, pakai_lag=True,
                                  kolom_fitur=kolom_fitur).fit(data, jarak)
    return model, k_terbaik, tabel_k


def tabel_estimasi(model: mod_model.EstimatorP0, data: pd.DataFrame,
                   skema: str = "dalam sampel") -> pd.DataFrame:
    """Tabel estimasi P0 per wilayah-tahun beserta galatnya."""
    estimasi = model.predict(data)
    tabel = pd.DataFrame({
        "Wilayah": data[cfg.KOLOM_WILAYAH].values,
        "Nama_Tampilan": [cfg.NAMA_TAMPILAN.get(w, w)
                          for w in data[cfg.KOLOM_WILAYAH]],
        "Tahun": data[cfg.KOLOM_TAHUN].astype(int).values,
        "P0_Aktual": np.round(data[cfg.TARGET].astype(float).values, 4),
        "P0_Estimasi": np.round(estimasi, 4),
    })
    tabel["Galat"] = np.round(tabel["P0_Aktual"] - tabel["P0_Estimasi"], 4)
    tabel["Galat_Absolut"] = np.round(tabel["Galat"].abs(), 4)
    tabel["Skema"] = skema
    return tabel


def simpan_artefak_model(model: mod_model.EstimatorP0, k: int,
                         tabel_k: pd.DataFrame, data: pd.DataFrame,
                         jarak: pd.DataFrame) -> dict:
    """Simpan model, koefisien, loadings, dan centroid ke folder ``models/``."""
    cfg.FOLDER_MODEL.mkdir(parents=True, exist_ok=True)
    mod_model.simpan_model(model, cfg.FOLDER_MODEL / "model_akhir.joblib")

    model.koefisien_tabel().to_csv(cfg.FOLDER_MODEL / "koefisien_mlr.csv",
                                   index=False)
    model.tabel_loadings().round(6).to_csv(cfg.FOLDER_MODEL / "loadings_pca.csv")
    koefisien_ke_variabel_asli(model).to_csv(
        cfg.FOLDER_MODEL / "koefisien_variabel_asli.csv", index=False)

    centroid = pd.read_csv(cfg.FOLDER_OUTPUT / "centroid_wilayah.csv") \
        if (cfg.FOLDER_OUTPUT / "centroid_wilayah.csv").exists() else None
    if centroid is not None:
        centroid.to_csv(cfg.FOLDER_MODEL / "centroid_wilayah.csv", index=False)

    jarak.to_csv(cfg.FOLDER_MODEL / "jarak_antar_wilayah.csv")
    pd.DataFrame({"wilayah": model.wilayah_latih,
                  "tetangga": [", ".join(model.tetangga_untuk(w))
                               for w in model.wilayah_latih]}
                 ).to_csv(cfg.FOLDER_MODEL / "daftar_tetangga.csv", index=False)

    metadata = {
        "judul": "Model estimasi P0 kabupaten/kota Kalimantan Timur",
        "periode_latih": [int(min(cfg.TAHUN_MODEL)), int(max(cfg.TAHUN_MODEL))],
        "jumlah_baris_latih": int(len(data)),
        "jumlah_wilayah": int(data[cfg.KOLOM_WILAYAH].nunique()),
        "fitur": list(model.kolom_fitur),
        "kolom_tidak_dipakai_sebagai_fitur": list(cfg.KOLOM_TERLARANG_SEBAGAI_FITUR),
        "m_komponen": int(model.m),
        "varians_kumulatif_komponen": [
            round(float(v), 6) for v in model.varians_kumulatif],
        "ambang_variansi": cfg.PCA_VAR_THRESHOLD,
        "k_tetangga_terpilih": int(k),
        "kandidat_k_inner_loro": tabel_k.to_dict(orient="records"),
        "koefisien": {n: round(float(v), 6)
                      for n, v in zip(model.nama_koef, model.koef)},
        "random_state": cfg.RANDOM_STATE,
        "catatan": ("Spatial lag dihitung dari P0 tetangga data latih, wilayah "
                    "tidak pernah menjadi tetangga dirinya sendiri."),
    }
    with open(cfg.FOLDER_MODEL / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def simpan_grafik_interpretasi(model: mod_model.EstimatorP0,
                               data: pd.DataFrame) -> None:
    """Buat scree plot, heatmap loading, dan grafik residual model akhir."""
    varians = model.pca.explained_variance_ratio_
    kumulatif = np.cumsum(varians)
    mod_plot.scree_plot(varians, kumulatif, model.m)
    mod_plot.heatmap_loading(model.tabel_loadings())

    desain, y = desain_final(model, data)
    hasil = sm.OLS(y, desain).fit()
    mod_plot.plot_residual(hasil.fittedvalues, hasil.resid,
                           "Residual vs Nilai Cocok - Model Akhir "
                           "(data penuh 80 baris)",
                           nama="plot_residual_model_akhir.png")

