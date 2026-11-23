# -*- coding: utf-8 -*-
"""Pipeline estimasi P0: StandardScaler -> PCA -> spatial lag -> MLR.

Aturan anti-kebocoran data yang dipegang kelas ini:

1. ``StandardScaler`` dan ``PCA`` hanya di-*fit* pada baris data latih.
2. Spatial lag sebuah baris (i, t) = rata-rata P0 ``k`` tetangga terdekat
   wilayah i **pada tahun t**, dengan syarat:
   - tetangga dipilih hanya dari wilayah yang ada di data latih,
   - wilayah i sendiri tidak pernah menjadi tetangganya,
   sehingga label P0 wilayah uji tidak pernah masuk sebagai fitur.
3. Untuk skema pembagian waktu, lag baris uji boleh memakai P0 tetangga pada
   tahun latih terakhir (``tahun_lag``) - didokumentasikan di README.
4. Jumlah komponen m dipilih dari variansi kumulatif >= ambang (85%) pada
   data latih lipatan itu, bukan pada seluruh data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src import geo as mod_geo


@dataclass
class EstimatorP0:
    """Estimator P0 dengan tiga saklar: PCA, spatial lag, dan jumlah tetangga.

    Attributes
    ----------
    k:
        Jumlah tetangga terdekat untuk spatial lag (diabaikan bila
        ``pakai_lag=False``).
    pakai_pca:
        Bila False, regresi memakai 12 fitur terstandardisasi langsung
        (pembanding "MLR tanpa PCA").
    pakai_lag:
        Bila False, spatial lag tidak dipakai (pembanding "PCA tanpa lag").
    ambang_variansi:
        Ambang variansi kumulatif untuk memilih m komponen.
    kolom_fitur:
        Daftar fitur; baku 12 fitur (atau 11 pada analisis sensitivitas).
    """

    k: int = 3
    pakai_pca: bool = True
    pakai_lag: bool = True
    ambang_variansi: float = cfg.PCA_VAR_THRESHOLD
    kolom_fitur: list[str] | None = None

    #: Atribut hasil fit (diisi oleh :meth:`fit`).
    scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    pca: PCA | None = field(default=None, init=False, repr=False)
    m: int | None = field(default=None, init=False)
    varians_ratio: np.ndarray | None = field(default=None, init=False)
    varians_kumulatif: np.ndarray | None = field(default=None, init=False)
    loadings: np.ndarray | None = field(default=None, init=False)
    koef: np.ndarray | None = field(default=None, init=False)
    nama_koef: list[str] = field(default_factory=list, init=False)
    wilayah_latih: list[str] = field(default_factory=list, init=False)
    jarak: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _y_latih: dict = field(default_factory=dict, init=False, repr=False)
    lag_latih: np.ndarray | None = field(default=None, init=False)

    # ------------------------------------------------------------------ fit
    def fit(self, data: pd.DataFrame, jarak: pd.DataFrame) -> "EstimatorP0":
        """Latih scaler, PCA, tabel lag, dan koefisien MLR pada data latih.

        Parameters
        ----------
        data:
            Baris latih (kolom wilayah, tahun, fitur, dan target).
        jarak:
            Matriks jarak haversine antar seluruh 10 wilayah (km).
        """
        if self.kolom_fitur is None:
            self.kolom_fitur = list(cfg.FITUR)
        data = data.reset_index(drop=True)
        if data.empty:
            raise ValueError("Data latih kosong")

        self.wilayah_latih = list(pd.unique(data[cfg.KOLOM_WILAYAH]))
        self.jarak = jarak
        if self.pakai_lag and len(self.wilayah_latih) <= self.k:
            raise ValueError(
                f"Jumlah wilayah latih ({len(self.wilayah_latih)}) harus lebih "
                f"besar dari k={self.k}")

        # 1) standardisasi (fit hanya pada data latih)
        x = data[self.kolom_fitur].to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(x)
        x_std = self.scaler.transform(x)

        # 2) PCA (fit hanya pada data latih); m dari variansi kumulatif
        if self.pakai_pca:
            self.pca = PCA(svd_solver="full", random_state=cfg.RANDOM_STATE).fit(x_std)
            kumulatif = np.cumsum(self.pca.explained_variance_ratio_)
            self.m = int(np.searchsorted(kumulatif, self.ambang_variansi - 1e-12) + 1)
            self.m = max(1, min(self.m, x_std.shape[1]))
            self.varians_ratio = self.pca.explained_variance_ratio_[: self.m]
            self.varians_kumulatif = kumulatif[: self.m]
            komponen = self.pca.components_[: self.m]
            self.loadings = komponen.T * np.sqrt(
                self.pca.explained_variance_[: self.m])
            z = x_std @ komponen.T
            nama_komponen = [f"PC{i + 1}" for i in range(self.m)]
        else:
            self.pca = None
            self.m = None
            self.varians_ratio = None
            self.varians_kumulatif = None
            self.loadings = None
            z = x_std
            nama_komponen = list(self.kolom_fitur)

        # 3) tabel P0 latih per (wilayah, tahun) untuk menghitung lag
        self._y_latih = {
            (baris[cfg.KOLOM_WILAYAH], int(baris[cfg.KOLOM_TAHUN])): float(baris[cfg.TARGET])
            for _, baris in data.iterrows()
        }
        lag = np.array([
            self._lag_dari_latih(baris[cfg.KOLOM_WILAYAH], int(baris[cfg.KOLOM_TAHUN]))
            for _, baris in data.iterrows()
        ], dtype=float)
        self.lag_latih = lag

        # 4) MLR (OLS) memakai kuadrat terkecil
        desain = self._susun_desain(z, lag)
        koef, *_ = np.linalg.lstsq(desain, data[cfg.TARGET].to_numpy(dtype=float),
                                   rcond=None)
        self.koef = koef
        self.nama_koef = (["intersep"] + nama_komponen
                          + (["SpatialLag"] if self.pakai_lag else []))
        return self

    # ------------------------------------------------------------- tetangga
    def tetangga_untuk(self, wilayah: str) -> list[str]:
        """k tetangga terdekat wilayah tertentu dari wilayah data latih saja.

        Wilayah uji tidak pernah menjadi tetangganya sendiri, dan wilayah uji
        tidak pernah menjadi tetangga wilayah lain (karena bukan data latih).
        """
        if self.jarak is None:
            raise RuntimeError("Model belum di-fit")
        lain = [w for w in self.wilayah_latih if w != wilayah]
        if len(lain) < self.k:
            raise ValueError(
                f"Hanya tersedia {len(lain)} kandidat tetangga untuk {wilayah} "
                f"(k={self.k})")
        urut = self.jarak.loc[wilayah, lain].sort_values(kind="mergesort")
        return list(urut.index[: self.k])

    def _lag_dari_latih(self, wilayah: str, tahun: int) -> float:
        """Spatial lag dari nilai P0 data latih pada tahun tertentu."""
        tetangga = self.tetangga_untuk(wilayah)
        nilai = []
        for t in tetangga:
            kunci = (t, int(tahun))
            if kunci not in self._y_latih:
                raise ValueError(
                    f"P0 latih untuk {t} tahun {tahun} tidak tersedia; "
                    "gunakan parameter tahun_lag pada skema pembagian waktu")
            nilai.append(self._y_latih[kunci])
        return float(np.mean(nilai))

    def lag_baris_uji(self, data: pd.DataFrame,
                      tahun_lag: int | None = None) -> np.ndarray:
        """Spatial lag baris uji (memakai P0 tetangga dari data latih)."""
        if self.jarak is None:
            raise RuntimeError("Model belum di-fit")
        if not self.pakai_lag:
            return np.zeros(len(data), dtype=float)
        nilai = []
        for _, baris in data.iterrows():
            tahun = int(tahun_lag if tahun_lag is not None else baris[cfg.KOLOM_TAHUN])
            nilai.append(self._lag_dari_latih(baris[cfg.KOLOM_WILAYAH], tahun))
        return np.array(nilai, dtype=float)

    # ---------------------------------------------------------------- desain
    def _susun_desain(self, z: np.ndarray, lag: np.ndarray) -> np.ndarray:
        """Susun matriks desain [intersep, komponen/fitur, (lag)]."""
        bagian = [np.ones((z.shape[0], 1)), z]
        if self.pakai_lag:
            bagian.append(lag.reshape(-1, 1))
        return np.hstack(bagian)

    def skor_pca(self, data: pd.DataFrame) -> np.ndarray:
        """Skor komponen (atau fitur terstandardisasi bila tanpa PCA)."""
        if self.scaler is None:
            raise RuntimeError("Model belum di-fit")
        x = data[self.kolom_fitur].to_numpy(dtype=float)
        x_std = self.scaler.transform(x)
        if self.pakai_pca:
            return x_std @ self.pca.components_[: self.m].T
        return x_std

    # --------------------------------------------------------------- prediksi
    def predict(self, data: pd.DataFrame, tahun_lag: int | None = None) -> np.ndarray:
        """Estimasi P0 untuk baris baru.

        Parameters
        ----------
        data:
            Baris yang akan diestimasi (fitur sudah tersedia).
        tahun_lag:
            Bila diisi, spatial lag dihitung dari P0 tetangga pada tahun itu
            (dipakai pada skema pembagian waktu).
        """
        z = self.skor_pca(data)
        lag = (self.lag_baris_uji(data, tahun_lag) if self.pakai_lag
               else np.zeros(len(data)))
        return self._susun_desain(z, lag) @ self.koef

    def prediksi_dengan_lag(self, data: pd.DataFrame, lag: float) -> np.ndarray:
        """Estimasi P0 dengan spatial lag yang ditahan tetap (untuk simulasi)."""
        z = self.skor_pca(data)
        lag_vec = np.full(len(data), float(lag))
        return self._susun_desain(z, lag_vec) @ self.koef

    # ------------------------------------------------------------ ringkasan
    def ringkasan(self) -> dict:
        """Ringkasan konfigurasi dan hasil fit (untuk log/tabel/laporan)."""
        return {
            "k": self.k,
            "pakai_pca": self.pakai_pca,
            "pakai_lag": self.pakai_lag,
            "jumlah_fitur": len(self.kolom_fitur or []),
            "m_komponen": self.m,
            "varians_kumulatif": (float(self.varians_kumulatif[-1])
                                  if self.varians_kumulatif is not None else None),
            "wilayah_latih": len(self.wilayah_latih),
            "koefisien": dict(zip(self.nama_koef,
                                  np.round(self.koef, 6).tolist()
                                  if self.koef is not None else [])),
        }

    def koefisien_tabel(self) -> pd.DataFrame:
        """Tabel koefisien MLR (nama -> nilai)."""
        return pd.DataFrame({"Koefisien": self.nama_koef,
                             "Nilai": np.round(self.koef, 6)})

    def tabel_loadings(self) -> pd.DataFrame:
        """Tabel loading komponen (indeks = fitur, kolom = PC)."""
        if self.loadings is None:
            raise ValueError("Model tidak memakai PCA")
        return pd.DataFrame(self.loadings, index=self.kolom_fitur,
                            columns=[f"PC{i + 1}" for i in range(self.m)])



def metrik_regresi(aktual, estimasi) -> dict:
    """R2, MAE, RMSE, dan MAPE untuk satu himpunan estimasi."""
    aktual = np.asarray(aktual, dtype=float)
    estimasi = np.asarray(estimasi, dtype=float)
    galat = aktual - estimasi
    ss_res = float(np.sum(galat ** 2))
    ss_tot = float(np.sum((aktual - aktual.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(galat)))
    rmse = float(np.sqrt(np.mean(galat ** 2)))
    dengan_nilai = np.abs(aktual) > 1e-9
    mape = (float(np.mean(np.abs(galat[dengan_nilai] / aktual[dengan_nilai])) * 100)
            if dengan_nilai.any() else float("nan"))
    return {"n": int(len(aktual)), "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def pilih_k_inner(data_latih: pd.DataFrame, jarak: pd.DataFrame,
                  kandidat_k: tuple[int, ...] | None = None,
                  pakai_pca: bool = True,
                  kolom_fitur: list[str] | None = None) -> tuple[int, pd.DataFrame]:
    """Pilih k dengan validasi dalam (leave-one-region-out pada data latih).

    Seluruh proses hanya memakai data latih, sehingga pemilihan k tidak
    memandang label wilayah uji. Hasilnya tetap dapat optimistis karena
    dilakukan berulang di setiap lipatan (dicatat di README).

    Returns
    -------
    (k_terbaik, tabel_metrik)
    """
    kandidat_k = tuple(kandidat_k or cfg.K_CANDIDATES)
    wilayah = list(pd.unique(data_latih[cfg.KOLOM_WILAYAH]))
    hasil = []
    for k in kandidat_k:
        aktual, estimasi = [], []
        for uji in wilayah:
            latih = data_latih[data_latih[cfg.KOLOM_WILAYAH] != uji]
            bagian_uji = data_latih[data_latih[cfg.KOLOM_WILAYAH] == uji]
            model = EstimatorP0(k=k, pakai_pca=pakai_pca, pakai_lag=True,
                                kolom_fitur=kolom_fitur).fit(latih, jarak)
            aktual.extend(bagian_uji[cfg.TARGET].tolist())
            estimasi.extend(model.predict(bagian_uji).tolist())
        m = metrik_regresi(aktual, estimasi)
        hasil.append({"k": k, "RMSE_inner": round(m["RMSE"], 6),
                      "MAE_inner": round(m["MAE"], 6),
                      "R2_inner": round(m["R2"], 6)})
    tabel = pd.DataFrame(hasil).sort_values(["RMSE_inner", "k"]).reset_index(drop=True)
    return int(tabel.loc[0, "k"]), tabel


def muat_model(berkas=None) -> "EstimatorP0":
    """Muat model akhir (EstimatorP0) dari folder models/."""
    import joblib

    path = Path(berkas) if berkas is not None else cfg.FOLDER_MODEL / "model_akhir.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Berkas model tidak ditemukan: {path}\n"
            "Jalankan lebih dahulu:  python -m src.evaluate")
    return joblib.load(path)


def simpan_model(model: "EstimatorP0", berkas=None):
    """Simpan model akhir (scaler + PCA + koefisien) ke folder models/."""
    import joblib

    path = Path(berkas) if berkas is not None else cfg.FOLDER_MODEL / "model_akhir.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

