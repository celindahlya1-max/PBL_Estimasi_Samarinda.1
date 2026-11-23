# -*- coding: utf-8 -*-
"""KNN sebagai pembanding: fitur spasial, regresi, dan klasifikasi kategori P0.

Modul ini menjawab tiga kebutuhan pelaporan:

1. **Transparansi fitur spasial.** :func:`fitur_spasial_detail` menampilkan
   tetangga setiap wilayah, jarak (km), nilai P0 tetangga, dan nilai lag yang
   dipakai model, sehingga cara fitur itu dihitung dapat diperiksa ulang.
2. **Pembanding regresi KNN.** Dua varian diuji dengan skema yang sama seperti
   model utama (leave-one-region-out dan pembagian waktu):
   - ``knn_spasial`` : estimasi = rata-rata P0 k tetangga terdekat (tanpa MLR),
   - ``knn_pca``     : KNeighborsRegressor pada skor PCA 12 fitur.
3. **Klasifikasi kategori kemiskinan.** Tiga kelas (Rendah/Menengah/Tinggi)
   dibentuk dari kuantil 1/3 dan 2/3 P0 *data latih lipatan itu*, lalu
   diklasifikasikan dengan KNN (fitur PCA dan fitur geografis). Hasilnya
   dilaporkan sebagai confusion matrix, akurasi, akurasi seimbang, presisi,
   recall, dan F1 per kelas, beserta pembanding kelas mayoritas.

Seluruh prediksi bersifat out-of-fold: scaler, PCA, ambang kelas, dan jumlah
Tetangga k hanya diambil dari data latih lipatan tersebut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src import model as mod_model
from src import plots as mod_plot

#: Nama varian (juga menjadi nilai kolom ``Model`` pada tabel perbandingan).
NAMA_REG_SPASIAL = "knn_spasial"
NAMA_REG_PCA = "knn_pca"
NAMA_KLAS_SPASIAL = "knn_klas_spasial"
NAMA_KLAS_PCA = "knn_klas_pca"
NAMA_KLAS_BASELINE = "knn_klas_baseline"

LABEL_VARIAN_KLAS = {
    NAMA_KLAS_PCA: "KNN pada skor PCA (3 kelas)",
    NAMA_KLAS_SPASIAL: "KNN geografis (kelas k tetangga terdekat)",
    NAMA_KLAS_BASELINE: "Baseline kelas mayoritas data latih",
}


# ---------------------------------------------------------------------------
# Kategori kemiskinan
# ---------------------------------------------------------------------------
def ambang_kelas(p0_latih) -> tuple[float, float]:
    """Batas kelas dari kuantil 1/3 dan 2/3 P0 data latih (poin persen)."""
    nilai = np.asarray(p0_latih, dtype=float)
    bawah, atas = np.quantile(nilai, cfg.KELAS_KUANTIL)
    return float(bawah), float(atas)


def kelas_kemiskinan(nilai, ambang: tuple[float, float]) -> str:
    """Kelas satu nilai P0 menurut ambang (bawah, atas)."""
    if nilai is None or (isinstance(nilai, float) and np.isnan(nilai)):
        return ""
    bawah, atas = ambang
    if float(nilai) <= bawah:
        return cfg.LABEL_KELAS_P0[0]
    if float(nilai) <= atas:
        return cfg.LABEL_KELAS_P0[1]
    return cfg.LABEL_KELAS_P0[2]


def kategori_kelas(nilai_p0, ambang: tuple[float, float]) -> list[str]:
    """Daftar kelas untuk sederet nilai P0."""
    return [kelas_kemiskinan(v, ambang) for v in np.asarray(nilai_p0, dtype=float)]

# ---------------------------------------------------------------------------
# Fitur spasial: transparansi tetangga dan lag
# ---------------------------------------------------------------------------
def fitur_spasial_detail(data: pd.DataFrame, jarak: pd.DataFrame,
                         k: int) -> pd.DataFrame:
    """Tabel panjang: tetangga, jarak, P0 tetangga, dan lag per wilayah-tahun.

    Satu baris = satu tetangga. Nilai ``P0_lag`` berulang pada semua baris
    wilayah-tahun yang sama karena lag adalah rata-rata seluruh tetangga.
    """
    ambang = ambang_kelas(data[cfg.TARGET])
    p0 = {(str(w), int(t)): float(v)
          for w, t, v in data[[cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN,
                               cfg.TARGET]].itertuples(index=False)}
    baris: list[dict] = []
    for wilayah in cfg.WILAYAH:
        lain = [w for w in cfg.WILAYAH if w != wilayah]
        urut = jarak.loc[wilayah, lain].sort_values(kind="mergesort")
        tetangga = list(urut.index[:k])
        for tahun in sorted(int(t) for t in pd.unique(data[cfg.KOLOM_TAHUN])):
            nilai = [p0.get((t, tahun), np.nan) for t in tetangga]
            tersedia = [v for v in nilai if not np.isnan(v)]
            lag = float(np.mean(tersedia)) if tersedia else float("nan")
            aktual = p0.get((wilayah, tahun), float("nan"))
            for peringkat, (tetangga_w, jarak_km, nilai_t) in enumerate(
                    zip(tetangga, urut.to_numpy()[:k], nilai), start=1):
                baris.append({
                    "Wilayah": wilayah,
                    "Tahun": tahun,
                    "k": int(k),
                    "Peringkat_tetangga": peringkat,
                    "Wilayah_tetangga": tetangga_w,
                    "Jarak_km": round(float(jarak_km), 2),
                    "P0_tetangga": (None if np.isnan(nilai_t)
                                    else round(float(nilai_t), 4)),
                    "P0_lag": None if np.isnan(lag) else round(lag, 4),
                    "P0_aktual": None if np.isnan(aktual) else round(float(aktual), 4),
                    "Kelas_aktual": kelas_kemiskinan(aktual, ambang),
                    "Kelas_lag": kelas_kemiskinan(lag, ambang),
                    "Ambang_bawah": round(ambang[0], 4),
                    "Ambang_atas": round(ambang[1], 4),
                })
    return pd.DataFrame(baris)


# ---------------------------------------------------------------------------
# Pembanding regresi KNN
# ---------------------------------------------------------------------------
def _prediksi_lag(latih: pd.DataFrame, uji: pd.DataFrame, jarak: pd.DataFrame,
                  k: int, tahun_lag: int | None = None) -> np.ndarray:
    """Estimasi KNN geospasial: rata-rata P0 k tetangga terdekat."""
    model = mod_model.EstimatorP0(k=k, pakai_pca=False, pakai_lag=True)
    model.fit(latih, jarak)
    return model.lag_baris_uji(uji, tahun_lag=tahun_lag)


def _ruang_pca(latih: pd.DataFrame, uji: pd.DataFrame
               ) -> tuple[np.ndarray, np.ndarray, int]:
    """Skor PCA latih dan uji; scaler/PCA hanya di-fit pada data latih.

    Aturan jumlah komponen m identik dengan :class:`src.model.EstimatorP0`
    (variansi kumulatif sampai ambang 85%) agar pembanding KNN memakai
    ruang fitur yang sama dengan model utama.
    """
    x_latih = latih[cfg.FITUR].to_numpy(dtype=float)
    x_uji = uji[cfg.FITUR].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_latih)
    z_std = scaler.transform(x_latih)
    pca = PCA(svd_solver="full", random_state=cfg.RANDOM_STATE).fit(z_std)
    kumulatif = np.cumsum(pca.explained_variance_ratio_)
    m = int(np.searchsorted(kumulatif, cfg.PCA_VAR_THRESHOLD - 1e-12) + 1)
    m = max(1, min(m, z_std.shape[1]))
    komponen = pca.components_[:m]
    return z_std @ komponen.T, scaler.transform(x_uji) @ komponen.T, m


def _prediksi_knn_pca(latih: pd.DataFrame, uji: pd.DataFrame,
                      jarak: pd.DataFrame, k: int) -> np.ndarray:
    """Estimasi KNeighborsRegressor pada skor PCA data latih lipatan."""
    # jarak tidak dipakai di varian ini: KNN bekerja pada ruang skor PCA.
    z_latih, z_uji, _ = _ruang_pca(latih, uji)
    knn = KNeighborsRegressor(n_neighbors=k, metric=cfg.KNN_METRIK,
                              weights=cfg.KNN_BOBOT)
    knn.fit(z_latih, latih[cfg.TARGET].to_numpy(dtype=float))
    return knn.predict(z_uji)


def _estimasi_regresi(latih, uji, jarak, jenis: str, k: int,
                      tahun_lag: int | None = None) -> np.ndarray:
    """Arahkan ke varian KNN yang diminta."""
    if jenis == NAMA_REG_SPASIAL:
        return _prediksi_lag(latih, uji, jarak, k, tahun_lag=tahun_lag)
    if jenis == NAMA_REG_PCA:
        return _prediksi_knn_pca(latih, uji, jarak, k)
    raise ValueError(f"Jenis KNN tidak dikenal: {jenis}")


def pilih_k_regresi(latih: pd.DataFrame, jarak: pd.DataFrame, jenis: str,
                    tahun_lag: int | None = None,
                    kandidat: tuple[int, ...] | None = None
                    ) -> tuple[int, pd.DataFrame]:
    """Pilih k lewat leave-one-region-out di dalam data latih saja."""
    kandidat = tuple(kandidat or cfg.KNN_K_KANDIDAT)
    wilayah = list(pd.unique(latih[cfg.KOLOM_WILAYAH]))
    baris = []
    for k in kandidat:
        aktual, estimasi = [], []
        for uji_w in wilayah:
            dalam = latih[latih[cfg.KOLOM_WILAYAH] != uji_w]
            bagian = latih[latih[cfg.KOLOM_WILAYAH] == uji_w]
            pred = _estimasi_regresi(dalam, bagian, jarak, jenis, k,
                                     tahun_lag=tahun_lag)
            aktual.extend(bagian[cfg.TARGET].tolist())
            estimasi.extend(np.asarray(pred, dtype=float).tolist())
        m = mod_model.metrik_regresi(aktual, estimasi)
        baris.append({"Jenis": jenis, "k": int(k),
                      "RMSE_inner": round(m["RMSE"], 6),
                      "MAE_inner": round(m["MAE"], 6),
                      "R2_inner": round(m["R2"], 6)})
    tabel = pd.DataFrame(baris).sort_values(["RMSE_inner", "k"]).reset_index(drop=True)
    return int(tabel.loc[0, "k"]), tabel

def _baris_hasil(skema: str, lipatan: str, subset: pd.DataFrame,
                 model: str, estimasi) -> list[dict]:
    """Susun baris hasil dengan format sama seperti ``src.evaluate``."""
    baris = []
    for (_, r), nilai in zip(subset.iterrows(), np.asarray(estimasi, dtype=float)):
        baris.append({
            "Skema": skema,
            "Lipatan": lipatan,
            "Wilayah": r[cfg.KOLOM_WILAYAH],
            "Tahun": int(r[cfg.KOLOM_TAHUN]),
            "Aktual": round(float(r[cfg.TARGET]), 4),
            "Model": model,
            "Estimasi": round(float(nilai), 4),
            "Galat": round(float(r[cfg.TARGET]) - float(nilai), 4),
        })
    return baris


def validasi_regresi(data: pd.DataFrame, jarak: pd.DataFrame,
                     skema: str = "LORO"
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validasi dua varian KNN regresi (LORO atau pembagian waktu).

    Returns
    -------
    (hasil, tuning)
        ``hasil`` memakai format yang sama dengan hasil model utama, sehingga
        dapat digabungkan pada tabel perbandingan model; ``tuning`` memuat
        RMSE dalam untuk setiap k pada setiap lipatan.
    """
    baris: list[dict] = []
    tuning: list[pd.DataFrame] = []

    if skema == "LORO":
        for uji_w in cfg.WILAYAH:
            latih = data[data[cfg.KOLOM_WILAYAH] != uji_w]
            uji = data[data[cfg.KOLOM_WILAYAH] == uji_w]
            for jenis in (NAMA_REG_SPASIAL, NAMA_REG_PCA):
                k, tabel = pilih_k_regresi(latih, jarak, jenis)
                tuning.append(tabel.assign(Skema=skema, Lipatan=uji_w))
                pred = _estimasi_regresi(latih, uji, jarak, jenis, k)
                baris.extend(_baris_hasil(skema, uji_w, uji, jenis, pred))
    elif skema == "Waktu":
        latih = data[data[cfg.KOLOM_TAHUN].isin(cfg.TAHUN_LATIH_WAKTU)]
        uji = data[data[cfg.KOLOM_TAHUN].isin(cfg.TAHUN_UJI_WAKTU)]
        for jenis in (NAMA_REG_SPASIAL, NAMA_REG_PCA):
            # pemilihan k memakai tahun yang sama di dalam periode latih;
            # prediksi baris uji memakai lag tahun acuan (2022) agar tidak
            # memakai label periode uji.
            k, tabel = pilih_k_regresi(latih, jarak, jenis)
            tuning.append(tabel.assign(Skema=skema, Lipatan="2017-2022"))
            pred = _estimasi_regresi(latih, uji, jarak, jenis, k,
                                     tahun_lag=cfg.TAHUN_ACUAN_LAG_WAKTU)
            baris.extend(_baris_hasil(skema, "2017-2022", uji, jenis, pred))
    else:
        raise ValueError(f"Skema tidak dikenal: {skema}")

    tabel_tuning = (pd.concat(tuning, ignore_index=True) if tuning
                    else pd.DataFrame())
    return pd.DataFrame(baris), tabel_tuning

# ---------------------------------------------------------------------------
# Klasifikasi kategori kemiskinan dengan KNN
# ---------------------------------------------------------------------------
def _prediksi_klas_pca(latih: pd.DataFrame, uji: pd.DataFrame,
                       jarak: pd.DataFrame, k: int,
                       ambang: tuple[float, float]) -> list[str]:
    """Klasifikasi KNN pada skor PCA (scaler/PCA hanya dari data latih)."""
    # jarak tidak dipakai di varian ini: KNN bekerja pada ruang skor PCA.
    z_latih, z_uji, _ = _ruang_pca(latih, uji)
    y_latih = kategori_kelas(latih[cfg.TARGET].to_numpy(), ambang)
    knn = KNeighborsClassifier(n_neighbors=k, metric=cfg.KNN_METRIK,
                               weights=cfg.KNN_BOBOT)
    knn.fit(z_latih, y_latih)
    return list(knn.predict(z_uji))


def _prediksi_klas_spasial(latih: pd.DataFrame, uji: pd.DataFrame,
                           jarak: pd.DataFrame, k: int,
                           ambang: tuple[float, float]) -> list[str]:
    """Kelas dari suara terbanyak k wilayah tetangga terdekat pada tahun sama.

    Bila suara teratas seri, dipakai kelas tetangga yang paling dekat.
    """
    wilayah_latih = list(pd.unique(latih[cfg.KOLOM_WILAYAH]))
    kategori = kategori_kelas(latih[cfg.TARGET].to_numpy(), ambang)
    tabel = {(str(w), int(t)): kelas for w, t, kelas in zip(
        latih[cfg.KOLOM_WILAYAH], latih[cfg.KOLOM_TAHUN], kategori)}
    mayoritas = pd.Series(kategori).mode().iloc[0]

    hasil: list[str] = []
    for _, r in uji.iterrows():
        wilayah = str(r[cfg.KOLOM_WILAYAH])
        tahun = int(r[cfg.KOLOM_TAHUN])
        lain = [w for w in wilayah_latih if w != wilayah]
        if not lain:
            hasil.append(mayoritas)
            continue
        urut = jarak.loc[wilayah, lain].sort_values(kind="mergesort")
        kelas_tetangga = [tabel[(t, tahun)] for t in urut.index
                          if (t, tahun) in tabel][:k]
        if not kelas_tetangga:
            hasil.append(mayoritas)
            continue
        hitung = pd.Series(kelas_tetangga).value_counts()
        teratas = set(hitung[hitung == hitung.max()].index)
        hasil.append(next(x for x in kelas_tetangga if x in teratas))
    return hasil


def _estimasi_klasifikasi(latih, uji, jarak, varian: str, k: int,
                          ambang: tuple[float, float]) -> list[str]:
    """Arahkan ke varian klasifikasi yang diminta."""
    if varian == NAMA_KLAS_PCA:
        return _prediksi_klas_pca(latih, uji, jarak, k, ambang)
    if varian == NAMA_KLAS_SPASIAL:
        return _prediksi_klas_spasial(latih, uji, jarak, k, ambang)
    raise ValueError(f"Varian klasifikasi tidak dikenal: {varian}")


def pilih_k_klasifikasi(latih: pd.DataFrame, jarak: pd.DataFrame, varian: str,
                        ambang: tuple[float, float],
                        kandidat: tuple[int, ...] | None = None
                        ) -> tuple[int, pd.DataFrame]:
    """Pilih k yang memaksimalkan akurasi leave-one-region-out di data latih."""
    kandidat = tuple(kandidat or cfg.KNN_K_KANDIDAT)
    wilayah = list(pd.unique(latih[cfg.KOLOM_WILAYAH]))
    baris = []
    for k in kandidat:
        benar = total = 0
        for uji_w in wilayah:
            dalam = latih[latih[cfg.KOLOM_WILAYAH] != uji_w]
            bagian = latih[latih[cfg.KOLOM_WILAYAH] == uji_w]
            if dalam[cfg.KOLOM_WILAYAH].nunique() < 2:
                continue
            pred = _estimasi_klasifikasi(dalam, bagian, jarak, varian, k, ambang)
            aktual = kategori_kelas(bagian[cfg.TARGET].to_numpy(), ambang)
            benar += int(sum(a == p for a, p in zip(aktual, pred)))
            total += len(aktual)
        akurasi = benar / total if total else float("nan")
        baris.append({"Varian": varian, "k": int(k),
                      "Akurasi_inner": round(akurasi, 4), "n_inner": total})
    tabel = (pd.DataFrame(baris)
             .sort_values(["Akurasi_inner", "k"], ascending=[False, True])
             .reset_index(drop=True))
    return int(tabel.loc[0, "k"]), tabel

def _baris_klas(skema: str, lipatan: str, varian: str, subset: pd.DataFrame,
                aktual: list[str], prediksi: list[str],
                ambang: tuple[float, float]) -> list[dict]:
    """Susun baris detail klasifikasi (satu baris per wilayah-tahun)."""
    baris = []
    for (_, r), a, p in zip(subset.iterrows(), aktual, prediksi):
        baris.append({
            "Skema": skema,
            "Lipatan": lipatan,
            "Varian": varian,
            "Wilayah": r[cfg.KOLOM_WILAYAH],
            "Tahun": int(r[cfg.KOLOM_TAHUN]),
            "P0_aktual": round(float(r[cfg.TARGET]), 4),
            "Kelas_aktual": a,
            "Kelas_prediksi": p,
            "Benar": bool(a == p),
            "Ambang_bawah": round(ambang[0], 4),
            "Ambang_atas": round(ambang[1], 4),
        })
    return baris


def validasi_klasifikasi(data: pd.DataFrame, jarak: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Klasifikasi out-of-fold: LORO (dua varian) dan LOOCV baris (varian PCA).

    LOOCV baris tidak diterapkan pada varian geografis karena membuang satu
    baris tidak menghilangkan wilayah tersebut dari data latih, sehingga label
    tahun lain di wilayah yang sama masih terbaca model (bocor informasi).
    """
    detail: list[dict] = []
    tuning: list[pd.DataFrame] = []
    k_terpilih: dict[str, list[int]] = {NAMA_KLAS_PCA: [], NAMA_KLAS_SPASIAL: []}

    # ------------------------------------------------------- LORO (10 lipatan)
    for uji_w in cfg.WILAYAH:
        latih = data[data[cfg.KOLOM_WILAYAH] != uji_w]
        uji = data[data[cfg.KOLOM_WILAYAH] == uji_w]
        ambang = ambang_kelas(latih[cfg.TARGET])
        aktual = kategori_kelas(uji[cfg.TARGET].to_numpy(), ambang)
        mayor = pd.Series(kategori_kelas(latih[cfg.TARGET].to_numpy(),
                                         ambang)).mode().iloc[0]
        detail.extend(_baris_klas("LORO", uji_w, NAMA_KLAS_BASELINE, uji,
                                  aktual, [mayor] * len(uji), ambang))
        for varian in (NAMA_KLAS_PCA, NAMA_KLAS_SPASIAL):
            k, tabel = pilih_k_klasifikasi(latih, jarak, varian, ambang)
            k_terpilih[varian].append(k)
            tuning.append(tabel.assign(Skema="LORO", Lipatan=uji_w))
            pred = _estimasi_klasifikasi(latih, uji, jarak, varian, k, ambang)
            detail.extend(_baris_klas("LORO", uji_w, varian, uji, aktual, pred, ambang))

    # ------------------------------------- LOOCV baris (hanya varian PCA)
    k_loocv = int(pd.Series(k_terpilih[NAMA_KLAS_PCA]).mode().iloc[0])
    for i in range(len(data)):
        latih = data.drop(index=data.index[i])
        uji = data.iloc[[i]]
        ambang = ambang_kelas(latih[cfg.TARGET])
        aktual = kategori_kelas(uji[cfg.TARGET].to_numpy(), ambang)
        mayor = pd.Series(kategori_kelas(latih[cfg.TARGET].to_numpy(),
                                         ambang)).mode().iloc[0]
        detail.extend(_baris_klas("LOOCV_baris", "baris", NAMA_KLAS_BASELINE, uji,
                                  aktual, [mayor], ambang))
        pred = _estimasi_klasifikasi(latih, uji, jarak, NAMA_KLAS_PCA, k_loocv, ambang)
        detail.extend(_baris_klas("LOOCV_baris", "baris", NAMA_KLAS_PCA, uji,
                                  aktual, pred, ambang))

    tabel_tuning = (pd.concat(tuning, ignore_index=True) if tuning
                    else pd.DataFrame())
    return pd.DataFrame(detail), tabel_tuning


# ---------------------------------------------------------------------------
# Confusion matrix dan metrik klasifikasi
# ---------------------------------------------------------------------------
def tabel_confusion(detail: pd.DataFrame,
                    label: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Confusion matrix (baris = aktual, kolom = prediksi) berurutan kelas."""
    label = tuple(label or cfg.LABEL_KELAS_P0)
    tabel = pd.crosstab(detail["Kelas_aktual"], detail["Kelas_prediksi"])
    tabel = tabel.reindex(index=label, columns=label, fill_value=0).astype(int)
    tabel.index.name = "Aktual"
    tabel.columns.name = "Prediksi"
    return tabel


def metrik_per_kelas(conf: pd.DataFrame) -> pd.DataFrame:
    """Presisi, recall, F1, dan support per kelas + rata-rata makro."""
    baris = []
    for kelas in conf.index:
        tp = int(conf.loc[kelas, kelas])
        support = int(conf.loc[kelas].sum())
        pred_pos = int(conf[kelas].sum())
        presisi = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * presisi * recall / (presisi + recall)) if (presisi + recall) else 0.0
        baris.append({"Kelas": kelas, "Presisi": round(presisi, 4),
                      "Recall": round(recall, 4), "F1": round(f1, 4),
                      "Support": support})
    tabel = pd.DataFrame(baris)
    tabel.loc[len(tabel)] = {
        "Kelas": "macro rata-rata",
        "Presisi": round(float(tabel["Presisi"].mean()), 4),
        "Recall": round(float(tabel["Recall"].mean()), 4),
        "F1": round(float(tabel["F1"].mean()), 4),
        "Support": int(tabel["Support"].sum()),
    }
    return tabel


def ringkasan_metrik(detail: pd.DataFrame) -> dict:
    """Akurasi, akurasi seimbang, dan macro F1 dari tabel detail klasifikasi."""
    conf = tabel_confusion(detail)
    total = int(conf.to_numpy().sum())
    akurasi = float(np.trace(conf.to_numpy()) / total) if total else float("nan")
    per_kelas = metrik_per_kelas(conf)
    return {
        "n": total,
        "Akurasi": round(akurasi, 4),
        "Akurasi_seimbang": round(float(per_kelas["Recall"].iloc[:-1].mean()), 4),
        "Macro_F1": float(per_kelas["F1"].iloc[-1]),
        "Macro_presisi": float(per_kelas["Presisi"].iloc[-1]),
        "Macro_recall": float(per_kelas["Recall"].iloc[-1]),
    }

# ---------------------------------------------------------------------------
# Orkestrasi: jalankan semua analisis KNN dan simpan hasilnya
# ---------------------------------------------------------------------------
def _k_terpilih(tuning: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    """k terbaik per (skema, lipatan, jenis) berdasarkan RMSE dalam."""
    hasil = {}
    for (skema, lipatan, jenis), grup in tuning.groupby(["Skema", "Lipatan", "Jenis"]):
        urut = grup.sort_values(["RMSE_inner", "k"]).iloc[0]
        hasil[(skema, lipatan, jenis)] = int(urut["k"])
    return hasil


def jalankan_knn(data: pd.DataFrame, jarak: pd.DataFrame) -> dict:
    """Jalankan pembanding KNN regresi + klasifikasi, simpan tabel dan grafik.

    Returns
    -------
    dict
        ``regresi`` (baris siap digabung ke tabel perbandingan model),
        ``regresi_metrik``, ``tuning_regresi``, ``fitur_spasial``,
        ``klasifikasi`` (detail), ``klasifikasi_metrik``, ``metrik_per_kelas``.
    """
    out = cfg.FOLDER_OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 78)
    print("FASE 3c - PEMBANDING KNN DAN KLASIFIKASI KATEGORI KEMISKINAN")
    print("-" * 78)

    # --------------------------------------------------------- regresi KNN
    hasil_loro, tuning_loro = validasi_regresi(data, jarak, "LORO")
    hasil_waktu, tuning_waktu = validasi_regresi(data, jarak, "Waktu")
    hasil = pd.concat([hasil_loro, hasil_waktu], ignore_index=True)
    tuning = pd.concat([tuning_loro, tuning_waktu], ignore_index=True)
    tuning.round(6).to_csv(out / "knn_tuning_k_regresi.csv", index=False)
    hasil.to_csv(out / "knn_regresi_detail.csv", index=False)

    baris = []
    for (skema, model), grup in hasil.groupby(["Skema", "Model"]):
        m = mod_model.metrik_regresi(grup["Aktual"], grup["Estimasi"])
        baris.append({"Skema": skema, "Model": model,
                      "Label": cfg.MODEL.get(model, model), "n": m["n"],
                      "R2": round(m["R2"], 4), "MAE": round(m["MAE"], 4),
                      "RMSE": round(m["RMSE"], 4),
                      "MAPE_persen": round(m["MAPE"], 2)})
    metrik = pd.DataFrame(baris).sort_values(["Skema", "RMSE"]).reset_index(drop=True)
    metrik.to_csv(out / "knn_regresi_metrik.csv", index=False)
    print("\nMetrik KNN regresi (out-of-fold):")
    print(metrik.to_string(index=False))

    k_lipatan = _k_terpilih(tuning)
    k_spasial = int(pd.Series([v for (s, _, j), v in k_lipatan.items()
                               if s == "LORO" and j == NAMA_REG_SPASIAL]).mode().iloc[0])
    fitur = fitur_spasial_detail(data, jarak, k_spasial)
    fitur.to_csv(out / "knn_fitur_spasial.csv", index=False)
    print(f"\nFitur spasial (k={k_spasial}) tersimpan: knn_fitur_spasial.csv "
          f"({len(fitur)} baris tetangga)")
    contoh = fitur[(fitur["Tahun"] == 2024) & (fitur["Wilayah"] == "Samarinda")]
    print("Contoh perhitungan lag Samarinda 2024:")
    print(contoh[["Wilayah_tetangga", "Jarak_km", "P0_tetangga", "P0_lag",
                  "P0_aktual", "Kelas_aktual", "Kelas_lag"]].to_string(index=False))

    # ----------------------------------------------------- klasifikasi KNN
    detail_klas, tuning_klas = validasi_klasifikasi(data, jarak)
    tuning_klas.round(4).to_csv(out / "knn_tuning_k_klasifikasi.csv", index=False)
    detail_klas.to_csv(out / "knn_klasifikasi_detail.csv", index=False)

    ringkas, per_kelas_semua = [], []
    for (skema, varian), grup in detail_klas.groupby(["Skema", "Varian"]):
        conf = tabel_confusion(grup)
        akhiran = varian.replace("knn_klas_", "")
        conf.to_csv(out / f"knn_confusion_{skema.lower()}_{akhiran}.csv")
        label_varian = LABEL_VARIAN_KLAS.get(varian, varian)
        mod_plot.plot_confusion(
            conf, f"Confusion matrix - {label_varian} ({skema}, n={len(grup)})",
            f"knn_confusion_{skema.lower()}_{akhiran}.png")
        r = ringkasan_metrik(grup)
        ringkas.append({"Skema": skema, "Varian": varian, "Label": label_varian, **r})
        per = metrik_per_kelas(conf)
        per.insert(0, "Varian", varian)
        per.insert(0, "Skema", skema)
        per_kelas_semua.append(per)

    tabel_ringkas = pd.DataFrame(ringkas)
    dasar = (tabel_ringkas[tabel_ringkas["Varian"] == NAMA_KLAS_BASELINE]
             .set_index("Skema")["Akurasi"])
    tabel_ringkas["Akurasi_baseline_mayoritas"] = tabel_ringkas["Skema"].map(dasar)
    tabel_ringkas["Selisih_vs_baseline"] = (tabel_ringkas["Akurasi"]
                                            - tabel_ringkas["Akurasi_baseline_mayoritas"]).round(4)
    tabel_ringkas.to_csv(out / "knn_klasifikasi_metrik.csv", index=False)
    pd.concat(per_kelas_semua, ignore_index=True).to_csv(
        out / "knn_metrik_per_kelas.csv", index=False)

    print("\nMetrik klasifikasi kategori kemiskinan:")
    print(tabel_ringkas[["Skema", "Label", "n", "Akurasi", "Akurasi_seimbang",
                         "Macro_F1", "Akurasi_baseline_mayoritas"]].to_string(index=False))
    for (skema, varian), grup in detail_klas.groupby(["Skema", "Varian"]):
        if varian == NAMA_KLAS_BASELINE:
            continue
        print(f"\nConfusion matrix {LABEL_VARIAN_KLAS.get(varian, varian)} ({skema}):")
        print(tabel_confusion(grup).to_string())

    return {"regresi": hasil, "regresi_metrik": metrik, "tuning_regresi": tuning,
            "fitur_spasial": fitur, "k_fitur_spasial": k_spasial,
            "klasifikasi": detail_klas, "klasifikasi_metrik": tabel_ringkas,
            "metrik_per_kelas": pd.concat(per_kelas_semua, ignore_index=True)}
