# -*- coding: utf-8 -*-
"""Pembacaan dan validasi panel data kemiskinan Kalimantan Timur.

Modul ini hanya membaca berkas ``data/dataset_panel_kaltim_2016_2024.xlsx``
(atau berkas lain yang diberikan eksplisit) dan memeriksa mutunya. Tidak ada
nilai yang diubah kecuali satu hal: bila sel ``P0_Persen_Miskin`` kosong,
nilainya dihitung ulang dengan rumus jumlah penduduk miskin / jumlah penduduk
x 100 - sesuai definisi kolom tersebut pada berkas sumber.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg

#: Varian nama wilayah (huruf kecil) -> nama baku.
_VARIAN_NAMA: dict[str, str] = {}
for _w in cfg.WILAYAH:
    _VARIAN_NAMA[_w.lower()] = _w
    _VARIAN_NAMA[("kota " + _w).lower()] = _w
    _VARIAN_NAMA[("kabupaten " + _w).lower()] = _w


def normalisasi_nama_wilayah(nama) -> str:
    """Ubah variasi nama wilayah ('Kota Balikpapan') menjadi nama baku."""
    if nama is None:
        return ""
    bersih = str(nama).replace("\xa0", " ").strip().strip('"')
    return _VARIAN_NAMA.get(bersih.lower(), bersih)


def muat_panel(berkas: Path | str | None = None) -> pd.DataFrame:
    """Muat panel dari berkas xlsx.

    Parameters
    ----------
    berkas:
        Lokasi berkas xlsx. Bila None dipakai berkas bawaan proyek di ``data/``.

    Raises
    ------
    FileNotFoundError
        Bila berkas tidak ada, disertai petunjuk cara membangkitkannya.
    ValueError
        Bila kolom wajib tidak tersedia.
    """
    path = Path(berkas) if berkas is not None else cfg.BERKAS_PANEL
    if not path.exists():
        raise FileNotFoundError(
            f"Berkas panel tidak ditemukan: {path}\n"
            "Jalankan lebih dahulu:  python tools\\bangun_dataset.py"
        )

    panel = pd.read_excel(path, sheet_name=cfg.SHEET_PANEL)
    panel.columns = [str(k).strip() for k in panel.columns]

    wajib = [cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN, cfg.TARGET] + cfg.FITUR
    hilang = [k for k in wajib if k not in panel.columns]
    if hilang:
        raise ValueError(
            "Kolom wajib tidak ditemukan pada berkas panel: " + ", ".join(hilang))
    if cfg.KOLOM_BANTU not in panel.columns:
        panel[cfg.KOLOM_BANTU] = np.nan

    panel[cfg.KOLOM_WILAYAH] = panel[cfg.KOLOM_WILAYAH].map(normalisasi_nama_wilayah)
    panel[cfg.KOLOM_TAHUN] = pd.to_numeric(
        panel[cfg.KOLOM_TAHUN], errors="coerce").astype("Int64")

    kolom_angka = [c for c in panel.columns
                   if c not in (cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN)]
    for kolom in kolom_angka:
        panel[kolom] = pd.to_numeric(panel[kolom], errors="coerce")

    # Hitung ulang P0 yang kosong (tidak ada imputasi lain apa pun)
    kosong_p0 = panel[cfg.TARGET].isna()
    bisa_hitung = (kosong_p0
                   & panel[cfg.KOLOM_BANTU].notna()
                   & panel[cfg.KOLOM_PENDUDUK].notna())
    panel.loc[bisa_hitung, cfg.TARGET] = (
        panel.loc[bisa_hitung, cfg.KOLOM_BANTU]
        / panel.loc[bisa_hitung, cfg.KOLOM_PENDUDUK] * 100.0).round(4)
    panel.attrs["p0_dihitung_ulang"] = int(bisa_hitung.sum())
    panel.attrs["p0_tetap_kosong"] = int(panel[cfg.TARGET].isna().sum())

    # Urutkan mengikuti urutan wilayah baku lalu tahun
    urutan = {w: i for i, w in enumerate(cfg.WILAYAH)}
    panel["_urut"] = panel[cfg.KOLOM_WILAYAH].map(urutan).fillna(99).astype(int)
    panel = (panel.sort_values(["_urut", cfg.KOLOM_TAHUN])
                  .drop(columns="_urut").reset_index(drop=True))
    return panel


def validasi_panel(panel: pd.DataFrame) -> dict:
    """Periksa struktur panel; hasilkan laporan (dict) untuk dicetak/diarsipkan."""
    laporan: dict = {}
    laporan["n_baris"] = int(len(panel))
    laporan["n_wilayah"] = int(panel[cfg.KOLOM_WILAYAH].nunique())
    laporan["wilayah_tidak_dikenal"] = sorted(
        set(panel[cfg.KOLOM_WILAYAH]) - set(cfg.WILAYAH))
    laporan["tahun"] = sorted(int(t) for t in panel[cfg.KOLOM_TAHUN].dropna().unique())
    laporan["duplikat_kunci"] = int(
        panel.duplicated([cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN]).sum())

    kosong = panel.isna().sum()
    laporan["sel_kosong_per_kolom"] = {
        str(k): int(v) for k, v in kosong.items() if int(v) > 0}

    model = panel[panel[cfg.KOLOM_TAHUN].isin(cfg.TAHUN_MODEL)]
    lengkap = model.dropna(subset=cfg.FITUR + [cfg.TARGET])
    laporan["baris_periode_model"] = int(len(model))
    laporan["baris_lengkap_periode_model"] = int(len(lengkap))
    laporan["p0_dihitung_ulang"] = int(panel.attrs.get("p0_dihitung_ulang", 0))

    pesan: list[str] = []
    if laporan["n_wilayah"] != len(cfg.WILAYAH):
        pesan.append(
            f"Jumlah wilayah {laporan['n_wilayah']}, seharusnya {len(cfg.WILAYAH)}")
    if laporan["duplikat_kunci"]:
        pesan.append("Terdapat duplikat pasangan (wilayah, tahun)")
    if laporan["baris_lengkap_periode_model"] != 80:
        pesan.append("Baris lengkap periode model "
                     f"{laporan['baris_lengkap_periode_model']}, seharusnya 80")
    laporan["pesan"] = pesan
    laporan["lengkap_sesuai_rencana"] = not pesan
    return laporan


def siapkan_data_model(panel: pd.DataFrame | None = None,
                       tahun: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Filter panel ke periode pemodelan dan hanya baris tanpa sel kosong.

    Periode baku 2017-2024; baris dengan fitur/target kosong dibuang (bukan
    diimputasi). Hasil yang diharapkan: 10 wilayah x 8 tahun = 80 baris.
    """
    panel = muat_panel() if panel is None else panel
    tahun = tuple(tahun) if tahun is not None else cfg.TAHUN_MODEL
    data = panel[panel[cfg.KOLOM_TAHUN].isin(tahun)].copy()
    data = data.dropna(subset=cfg.FITUR + [cfg.TARGET]).reset_index(drop=True)

    jumlah_wilayah = data[cfg.KOLOM_WILAYAH].nunique()
    if jumlah_wilayah == 0 or len(data) != jumlah_wilayah * len(tahun):
        raise ValueError(
            "Panel periode model tidak lengkap: "
            f"{len(data)} baris untuk {jumlah_wilayah} wilayah x {len(tahun)} tahun")
    return data


def muat_catatan_mutu() -> pd.DataFrame:
    """Muat catatan mutu data (koreksi & pencilan yang dipertahankan)."""
    if not cfg.BERKAS_CATATAN_MUTU.exists():
        return pd.DataFrame(columns=["Kabupaten_Kota", "Tahun", "Kolom",
                                     "Nilai_Sumber", "Nilai_Pembanding",
                                     "Jenis", "Tindakan", "Catatan"])
    return pd.read_csv(cfg.BERKAS_CATATAN_MUTU)


def tabel_lebar(data: pd.DataFrame, kolom: str) -> pd.DataFrame:
    """Susun tabel lebar wilayah x tahun untuk satu kolom (mis. P0)."""
    return (data.pivot_table(index=cfg.KOLOM_WILAYAH, columns=cfg.KOLOM_TAHUN,
                             values=kolom, aggfunc="mean")
                 .reindex(cfg.WILAYAH))


def ringkasan_data_lengkap(panel: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan kelengkapan sel per tahun (untuk README/dashboard)."""
    baris = []
    for tahun, grup in panel.groupby(cfg.KOLOM_TAHUN, dropna=True):
        sel = grup.drop(columns=[cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN])
        baris.append({
            "Tahun": int(tahun),
            "Jumlah_wilayah": int(grup[cfg.KOLOM_WILAYAH].nunique()),
            "Sel_terisi": int(sel.notna().sum().sum()),
            "Sel_kosong": int(sel.isna().sum().sum()),
            "Persen_terisi": round(sel.notna().sum().sum() / sel.size * 100, 2),
        })
    return pd.DataFrame(baris)



# ---------------------------------------------------------------------------
# Audit pembentukan target P0
# ---------------------------------------------------------------------------
#: Penjelasan baku mengapa P0 hasil hitung dapat berbeda dari angka resmi BPS.
CATATAN_SELISIH_P0 = (
    "Pembilang = jumlah penduduk miskin (ribu jiwa) hasil Susenas periode "
    "September; penyebut = jumlah penduduk dari proyeksi/SP2020; pembulatan "
    "pembilang 2 desimal dan hasil 4 desimal; P0 resmi BPS memakai penimbang "
    "Susenas dan pembulatan 2 desimal sehingga dapat berbeda tipis dari rasio "
    "agregat ini."
)


def berkas_sumber(awalan: str, tahun: int) -> str:
    """Nama berkas CSV sumber untuk sebuah awalan judul tabel dan tahun."""
    kandidat = sorted(cfg.FOLDER_SUMBER_CSV.glob(f"{awalan}*{tahun}*.csv"))
    return kandidat[0].name if kandidat else ""


def audit_target_p0(panel: pd.DataFrame) -> pd.DataFrame:
    """Audit traceability target: pembilang, penyebut, hasil hitung, selisih.

    Dua kolom terakhir sengaja dikosongkan agar dapat diisi manual dengan angka
    P0 resmi BPS dari publikasi (mis. Berita Resmi Statistik / Statistik
    Kesejahteraan Rakyat) sehingga perbedaan dapat ditunjukkan terukur.
    """
    baris = []
    for _, r in panel.iterrows():
        tahun = int(r[cfg.KOLOM_TAHUN])
        pembilang = r.get(cfg.KOLOM_BANTU)
        penyebut = r.get(cfg.KOLOM_PENDUDUK)
        hitung = (float(pembilang) / float(penyebut) * 100.0
                  if pd.notna(pembilang) and pd.notna(penyebut) and float(penyebut) != 0
                  else float("nan"))
        di_panel = r.get(cfg.TARGET)
        baris.append({
            "Kabupaten_Kota": r[cfg.KOLOM_WILAYAH],
            "Tahun": tahun,
            f"Pembilang_{cfg.KOLOM_BANTU}": (None if pd.isna(pembilang)
                                             else round(float(pembilang), 4)),
            f"Penyebut_{cfg.KOLOM_PENDUDUK}": (None if pd.isna(penyebut)
                                               else round(float(penyebut), 4)),
            "P0_hitung_ulang": None if np.isnan(hitung) else round(hitung, 4),
            "P0_di_panel": None if pd.isna(di_panel) else round(float(di_panel), 4),
            "Selisih_poin": (None if (np.isnan(hitung) or pd.isna(di_panel))
                             else round(float(di_panel) - hitung, 6)),
            "Sumber_pembilang": berkas_sumber(
                "Jumlah Penduduk Miskin menurut Kabupaten_Kota,", tahun),
            "Sumber_penyebut": berkas_sumber(
                "Penduduk, Laju Pertumbuhan Penduduk,", tahun),
            "P0_resmi_BPS": "",
            "Selisih_vs_P0_resmi": "",
            "Catatan_kemungkinan_selisih": CATATAN_SELISIH_P0,
        })
    return pd.DataFrame(baris)