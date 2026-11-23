# -*- coding: utf-8 -*-
"""Konfigurasi global proyek.

Seluruh modul lain mengambil konstanta dari sini agar tidak ada pengulangan
dan tidak ada path absolut: semua path dihitung relatif terhadap akar proyek.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Lokasi berkas (relatif, dihitung dari letak berkas ini)
# ---------------------------------------------------------------------------
AKAR_PROYEK: Path = Path(__file__).resolve().parent.parent
FOLDER_DATA: Path = AKAR_PROYEK / "data"
FOLDER_OUTPUT: Path = AKAR_PROYEK / "outputs"
FOLDER_MODEL: Path = AKAR_PROYEK / "models"
FOLDER_SUMBER_CSV: Path = FOLDER_DATA / "sumber_csv"

BERKAS_PANEL: Path = FOLDER_DATA / "dataset_panel_kaltim_2016_2024.xlsx"
BERKAS_GEOJSON: Path = FOLDER_DATA / "Kaltim_Kabupaten.geojson"
BERKAS_CATATAN_MUTU: Path = FOLDER_DATA / "catatan_mutu_data.csv"
SHEET_PANEL: str = "Data_Panel"

# ---------------------------------------------------------------------------
# Wilayah dan tahun
# ---------------------------------------------------------------------------
#: Urutan 10 kabupaten/kota Provinsi Kalimantan Timur (7 kabupaten + 3 kota).
#: Nama "baku" ini dipakai di seluruh dataset; nama pada GeoJSON bisa memuat
#: awalan "Kota " sehingga dipetakan di :mod:`src.geo`.
WILAYAH: list[str] = [
    "Paser",
    "Kutai Barat",
    "Kutai Kartanegara",
    "Kutai Timur",
    "Berau",
    "Penajam Paser Utara",
    "Mahakam Ulu",
    "Balikpapan",
    "Samarinda",
    "Bontang",
]

#: Nama tampilan (untuk dashboard).
NAMA_TAMPILAN: dict[str, str] = {
    "Paser": "Kabupaten Paser",
    "Kutai Barat": "Kabupaten Kutai Barat",
    "Kutai Kartanegara": "Kabupaten Kutai Kartanegara",
    "Kutai Timur": "Kabupaten Kutai Timur",
    "Berau": "Kabupaten Berau",
    "Penajam Paser Utara": "Kabupaten Penajam Paser Utara",
    "Mahakam Ulu": "Kabupaten Mahakam Ulu",
    "Balikpapan": "Kota Balikpapan",
    "Samarinda": "Kota Samarinda",
    "Bontang": "Kota Bontang",
}

TAHUN_TERSEDIA: tuple[int, ...] = tuple(range(2016, 2025))
#: Tahun yang dipakai untuk pemodelan: 2016 tidak lengkap
#: (HLS, IDG, TPAK, TPT kosong di tingkat kabupaten/kota).
TAHUN_MODEL: tuple[int, ...] = tuple(range(2017, 2025))

# ---------------------------------------------------------------------------
# Kolom dataset
# ---------------------------------------------------------------------------
KOLOM_WILAYAH = "Kabupaten_Kota"
KOLOM_TAHUN = "Tahun"
TARGET = "P0_Persen_Miskin"
KOLOM_BANTU = "Jumlah_Penduduk_Miskin_Ribu"
KOLOM_PENDUDUK = "Jumlah_Penduduk_Ribu"

#: 12 fitur prediktor sesuai usulan penelitian.
FITUR: list[str] = [
    "Peng_NonMakanan_Rp",
    "Peng_Disesuaikan_RibuRp",
    "RLS",
    "HLS_Perempuan",
    "UHH",
    "IDG",
    "TPAK",
    "TPT",
    "Jumlah_Penduduk_Ribu",
    "Laju_Pertumbuhan_Penduduk",
    "Kepadatan_per_km2",
    "Rasio_Jenis_Kelamin",
]

#: Fitur tanpa kolom penduduk, dipakai pada analisis sensitivitas
#: (Jumlah_Penduduk_Ribu adalah penyebut rumus P0).
FITUR_TANPA_PENDUDUK: list[str] = [f for f in FITUR if f != KOLOM_PENDUDUK]

#: Kolom yang tidak boleh dipakai sebagai fitur (data leakage).
KOLOM_TERLARANG_SEBAGAI_FITUR: tuple[str, ...] = (TARGET, KOLOM_BANTU)

#: Label ramah untuk tabel dan grafik.
LABEL_FITUR: dict[str, str] = {
    "Peng_NonMakanan_Rp": "Pengeluaran bukan makanan (Rp/kapita/bulan)",
    "Peng_Disesuaikan_RibuRp": "Pengeluaran disesuaikan (ribu Rp/kapita/tahun)",
    "RLS": "Rata-rata lama sekolah (tahun)",
    "HLS_Perempuan": "Harapan lama sekolah perempuan (tahun)",
    "UHH": "Umur harapan hidup (tahun)",
    "IDG": "Indeks pemberdayaan gender",
    "TPAK": "Tingkat partisipasi angkatan kerja (%)",
    "TPT": "Tingkat pengangguran terbuka (%)",
    "Jumlah_Penduduk_Ribu": "Jumlah penduduk (ribu jiwa)",
    "Laju_Pertumbuhan_Penduduk": "Laju pertumbuhan penduduk (%/tahun)",
    "Kepadatan_per_km2": "Kepadatan penduduk (jiwa/km2)",
    "Rasio_Jenis_Kelamin": "Rasio jenis kelamin (per 100 perempuan)",
    TARGET: "Tingkat kemiskinan P0 (%)",
    KOLOM_BANTU: "Jumlah penduduk miskin (ribu jiwa)",
}

LABEL_PENDEK: dict[str, str] = {
    "Peng_NonMakanan_Rp": "Peng. non-makanan",
    "Peng_Disesuaikan_RibuRp": "Peng. disesuaikan",
    "RLS": "RLS",
    "HLS_Perempuan": "HLS perempuan",
    "UHH": "UHH",
    "IDG": "IDG",
    "TPAK": "TPAK",
    "TPT": "TPT",
    "Jumlah_Penduduk_Ribu": "Jumlah penduduk",
    "Laju_Pertumbuhan_Penduduk": "Laju pertumbuhan",
    "Kepadatan_per_km2": "Kepadatan",
    "Rasio_Jenis_Kelamin": "Rasio jenis kelamin",
    TARGET: "P0",
}

# ---------------------------------------------------------------------------
# Parameter pemodelan
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
#: Ambang variansi kumulatif untuk memilih jumlah komponen utama m.
PCA_VAR_THRESHOLD: float = 0.85
#: Kandidat jumlah tetangga terdekat pada KNN geospasial.
K_CANDIDATES: tuple[int, ...] = (2, 3, 4)
#: CRS terproyeksi untuk perhitungan centroid (UTM zona 50S, sesuai Kaltim).
CRS_PROYEKSI: str = "EPSG:32750"
CRS_GEOGRAFIS: str = "EPSG:4326"

#: Pembagian waktu untuk validasi tambahan.
TAHUN_LATIH_WAKTU: tuple[int, ...] = tuple(range(2017, 2023))   # 2017-2022
TAHUN_UJI_WAKTU: tuple[int, ...] = (2023, 2024)
#: Tahun acuan spatial lag pada skema pembagian waktu (tahun latih terakhir).
TAHUN_ACUAN_LAG_WAKTU: int = 2022

#: Nama model yang dibandingkan (dipakai di tabel & dashboard).
MODEL = {
    "baseline": "Baseline (rata-rata P0 latih)",
    "mlr_tanpa_pca": "MLR tanpa PCA (12 fitur + spatial lag)",
    "pca_tanpa_lag": "PCA + MLR (tanpa spatial lag)",
    "utama": "PCA + spatial lag + MLR (model utama)",
    "knn_spasial": "KNN geospasial (rata-rata P0 k tetangga, tanpa MLR)",
    "knn_pca": "KNN regresi pada skor PCA (pembanding)",
}

# ---------------------------------------------------------------------------
# KNN pembanding dan klasifikasi kategori kemiskinan
# ---------------------------------------------------------------------------
#: Kandidat jumlah tetangga untuk pembanding KNN (dipilih lewat LORO dalam).
KNN_K_KANDIDAT: tuple[int, ...] = (1, 2, 3, 4, 5)
#: Metrik jarak KNN pada ruang fitur (skor PCA terstandardisasi).
KNN_METRIK: str = "euclidean"
#: Bobot KNN: "uniform" (rata-rata sederhana) atau "distance".
KNN_BOBOT: str = "uniform"

#: Batas kelas kategori kemiskinan dihitung dari kuantil P0 data latih tiap
#: lipatan. Bukan ambang resmi BPS: semata klasifikasi analitis untuk
#: membandingkan kinerja KNN sebagai klasifikator.
KELAS_KUANTIL: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
LABEL_KELAS_P0: tuple[str, str, str] = ("Rendah", "Menengah", "Tinggi")