# -*- coding: utf-8 -*-
"""Operasi geospasial: pemetaan nama, centroid, jarak haversine, tetangga.

Centroid dihitung pada CRS terproyeksi (UTM zona 50S / EPSG:32750) agar satuan
perhitungannya meter, lalu dikembalikan ke lintang-bujur (EPSG:4326) untuk
ditampilkan dan dipakai pada rumus haversine.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from src import config as cfg
from src.data import normalisasi_nama_wilayah

#: Jari-jari bumi rata-rata (km) untuk rumus haversine.
JARI_JARI_BUMI_KM = 6371.0088


def muat_geometri(berkas: Path | str | None = None) -> gpd.GeoDataFrame:
    """Muat GeoJSON batas kabupaten/kota dan bersihkan poligon artefak.

    Fitur yang namanya tidak dikenali (mis. poligon tanpa ``WADMKK``) dibuang.
    Kolom hasil: ``Wilayah`` (nama baku dataset), ``Nama_GeoJSON``, ``geometry``.
    """
    path = Path(berkas) if berkas is not None else cfg.BERKAS_GEOJSON
    if not path.exists():
        raise FileNotFoundError(f"Berkas GeoJSON tidak ditemukan: {path}")

    mentah = gpd.read_file(path)
    if "WADMKK" not in mentah.columns:
        raise ValueError("GeoJSON tidak memiliki properti 'WADMKK'")

    mentah["Nama_GeoJSON"] = mentah["WADMKK"].fillna("").astype(str).str.strip()
    mentah["Wilayah"] = [
        normalisasi_nama_wilayah(n) if n else "" for n in mentah["Nama_GeoJSON"]
    ]
    bersih = mentah[mentah["Wilayah"].isin(cfg.WILAYAH)].copy()
    if len(bersih) != len(cfg.WILAYAH):
        hilang = sorted(set(cfg.WILAYAH) - set(bersih["Wilayah"]))
        raise ValueError(
            f"GeoJSON hanya memuat {len(bersih)} dari {len(cfg.WILAYAH)} wilayah. "
            f"Tidak ditemukan: {hilang}")
    if bersih.crs is None:
        bersih = bersih.set_crs(cfg.CRS_GEOGRAFIS)
    return bersih[["Wilayah", "Nama_GeoJSON", "geometry"]].reset_index(drop=True)


def tabel_pemetaan(geometri: gpd.GeoDataFrame) -> pd.DataFrame:
    """Tabel pemetaan nama wilayah GeoJSON <-> nama baku dataset."""
    return pd.DataFrame({
        "Nama_GeoJSON": geometri["Nama_GeoJSON"].values,
        "Wilayah_Dataset": geometri["Wilayah"].values,
        "Sama_Persis": [a == b for a, b in
                        zip(geometri["Nama_GeoJSON"], geometri["Wilayah"])],
    })


def hitung_centroid(geometri: gpd.GeoDataFrame) -> pd.DataFrame:
    """Hitung centroid tiap wilayah di CRS terproyeksi lalu ubah ke lon/lat.

    Returns
    -------
    pandas.DataFrame
        Kolom: Wilayah, X_proyeksi, Y_proyeksi (meter), Longitude, Latitude,
        Luas_km2.
    """
    proyeksi = geometri.to_crs(cfg.CRS_PROYEKSI)
    titik = proyeksi.geometry.centroid
    geografis = gpd.GeoSeries(titik, crs=cfg.CRS_PROYEKSI).to_crs(cfg.CRS_GEOGRAFIS)
    tabel = pd.DataFrame({
        "Wilayah": geometri["Wilayah"].values,
        "X_proyeksi": titik.x.values.round(2),
        "Y_proyeksi": titik.y.values.round(2),
        "Longitude": geografis.x.values.round(6),
        "Latitude": geografis.y.values.round(6),
        "Luas_km2": (proyeksi.geometry.area / 1e6).values.round(1),
    })
    urutan = {w: i for i, w in enumerate(cfg.WILAYAH)}
    return (tabel.sort_values("Wilayah", key=lambda s: s.map(urutan))
                 .reset_index(drop=True))


def matriks_jarak_haversine(centroid: pd.DataFrame) -> pd.DataFrame:
    """Matriks jarak haversine (km) antar centroid wilayah."""
    lat = np.radians(centroid["Latitude"].to_numpy(dtype=float))
    lon = np.radians(centroid["Longitude"].to_numpy(dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2)
    jarak = 2.0 * JARI_JARI_BUMI_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return pd.DataFrame(jarak, index=centroid["Wilayah"], columns=centroid["Wilayah"])


def tetangga_terdekat(jarak: pd.DataFrame, k: int,
                      kandidat: list[str] | None = None) -> dict[str, list[str]]:
    """k tetangga terdekat (bukan diri sendiri) untuk setiap wilayah kandidat.

    Parameters
    ----------
    jarak:
        Matriks jarak simetris antar seluruh wilayah terpantau.
    k:
        Jumlah tetangga.
    kandidat:
        Daftar wilayah yang BOLEH menjadi tetangga. Pada validasi diisi wilayah
        data latih saja, supaya label wilayah uji tidak pernah menjadi fitur.
    """
    kandidat = list(kandidat) if kandidat is not None else list(jarak.index)
    hasil: dict[str, list[str]] = {}
    for wilayah in kandidat:
        lain = [w for w in kandidat if w != wilayah]
        if len(lain) < k:
            raise ValueError(
                f"Wilayah {wilayah} hanya memiliki {len(lain)} kandidat tetangga, "
                f"tidak cukup untuk k={k}")
        urut = jarak.loc[wilayah, lain].sort_values(kind="mergesort")
        hasil[wilayah] = list(urut.index[:k])
    return hasil


def daftar_tetangga_umum(centroid: pd.DataFrame, k: int) -> dict[str, list[str]]:
    """Daftar tetangga memakai seluruh 10 wilayah (model akhir/dashboard)."""
    return tetangga_terdekat(matriks_jarak_haversine(centroid), k, cfg.WILAYAH)


def simpan_centroid(centroid: pd.DataFrame,
                    berkas: Path | str | None = None) -> Path:
    """Simpan tabel centroid ke CSV (path relatif, tanpa path absolut)."""
    path = (Path(berkas) if berkas is not None
            else cfg.FOLDER_OUTPUT / "centroid_wilayah.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    centroid.to_csv(path, index=False)
    return path


def muat_centroid(berkas: Path | str | None = None) -> pd.DataFrame:
    """Muat tabel centroid; hitung ulang dari GeoJSON bila berkas belum ada."""
    path = (Path(berkas) if berkas is not None
            else cfg.FOLDER_OUTPUT / "centroid_wilayah.csv")
    if path.exists():
        return pd.read_csv(path)
    return hitung_centroid(muat_geometri())


def geometri_geojson_folium(geometri: gpd.GeoDataFrame | None = None) -> dict:
    """Ubah geometri menjadi dict GeoJSON siap pakai untuk peta Folium."""
    geometri = muat_geometri() if geometri is None else geometri
    fitur = []
    for _, baris in geometri.iterrows():
        fitur.append({
            "type": "Feature",
            "properties": {"Wilayah": baris["Wilayah"],
                           "Nama_GeoJSON": baris["Nama_GeoJSON"]},
            "geometry": shape(baris["geometry"]).__geo_interface__,
        })
    return {"type": "FeatureCollection", "features": fitur}

