# PBL — Estimasi dan Pemetaan Tingkat Kemiskinan Kabupaten/Kota di Provinsi Kalimantan Timur, 2017–2024

**Politeknik Negeri Samarinda** — Proyek Berbasis Pembelajaran (PBL).

Proyek ini mengestimasi **tingkat kemiskinan (P0)** 10 kabupaten/kota Provinsi
Kalimantan Timur dengan alur: **Standardisasi → PCA (reduksi dimensi) →
KNN geospasial (pembangun fitur *spatial lag*) → Regresi Linear Multivariabel
(MLR)**, lalu menyajikannya dalam **dashboard Streamlit 7 tab** (bahasa Indonesia).

> **Peran KNN diperjelas.** Istilah "KNN geospasial" di sini berarti *k*
> tetangga terdekat dipakai untuk menghitung fitur **spatial lag** (rata-rata P0
> tetangga), bukan algoritma klasifikasi. Agar tetap sebanding dengan rancangan
> awal, KNN juga diuji sebagai **pembanding regresi** dan sebagai **klasifikator
> tiga kelas** (Rendah/Menengah/Tinggi) lengkap dengan *confusion matrix* dan
> metrik per kelas - lihat §6.6.

> **Istilah penting.** Seluruh nilai keluaran model disebut **estimasi**.
> P0 dihitung sendiri dari data resmi: `jumlah penduduk miskin / jumlah penduduk × 100`.
> Angka P0 di sini adalah **hasil hitung**, bukan angka final publikasi BPS,
> dan bukan ramalan kebijakan.

---

## 1. Ringkasan hasil (jujur, apa adanya)

Validasi utama: **leave-one-region-out (LORO)** — setiap wilayah pernah menjadi
data uji, jadi 80 nilai estimasi bersifat *out-of-fold* (tidak memakai label
wilayah itu sendiri).

| Skema | Model | R² | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|
| LORO (80 obs) | **PCA + MLR tanpa spatial lag** | **0,4934** | **1,4975** | **1,7759** | 29,38 |
| LORO (80 obs) | PCA + spatial lag + MLR (model utama) | 0,4245 | 1,5220 | 1,8929 | 27,09 |
| LORO (80 obs) | Baseline (rata-rata P0 latih) | −0,2183 | 2,4242 | 2,7541 | 49,01 |
| LORO (80 obs) | MLR tanpa PCA (12 fitur + lag) | −0,9092 | 2,7158 | 3,4477 | 53,79 |
| LORO (80 obs) | KNN regresi pada skor PCA (pembanding) | 0,2389 | 1,8797 | 2,1768 | 36,14 |
| LORO (80 obs) | KNN geospasial murni (rata-rata P0 tetangga) | 0,0121 | 2,1950 | 2,4801 | 38,82 |
| Waktu (20 obs) | **PCA + spatial lag + MLR (model utama)** | **0,6427** | **1,0044** | **1,3834** | 20,24 |
| Waktu (20 obs) | MLR tanpa PCA (12 fitur + lag) | 0,5906 | 1,2673 | 1,4809 | 26,18 |
| Waktu (20 obs) | PCA + MLR tanpa spatial lag | 0,5066 | 1,3706 | 1,6257 | 28,86 |
| Waktu (20 obs) | Baseline (rata-rata P0 latih) | −0,1020 | 2,1379 | 2,4297 | 53,51 |
| Waktu (20 obs) | KNN regresi pada skor PCA (pembanding) | 0,5586 | 1,2150 | 1,5377 | 25,19 |
| Waktu (20 obs) | KNN geospasial murni (rata-rata P0 tetangga) | −0,1349 | 2,0555 | 2,4658 | 50,44 |

Bacaan singkat:

1. **PCA jelas membantu.** Tanpa PCA (12 fitur mentah, VIF sampai 36,8), RMSE
   LORO 3,4477 — lebih buruk daripada baseline. Dengan PCA, RMSE turun ke 1,77–1,89.
2. **Model utama mengalahkan baseline dan MLR tanpa PCA pada kedua skema**, dan
   menjadi **yang terbaik pada pembagian waktu** (RMSE 1,3834 vs 1,6257).
3. **Pada LORO, model utama kalah tipis** dari PCA+MLR tanpa lag (RMSE 1,8929
   vs 1,7759). Ini dilaporkan apa adanya: menambahkan tetangga tidak otomatis
   memperbaiki estimasi wilayah yang belum pernah dilihat model.
4. Sensitivitas: membuang `Jumlah_Penduduk_Ribu` (penyebut rumus P0) **sedikit
   memperbaiki** hasil (RMSE 1,8513; R² 0,4495) — tanda variabel itu lemah
   sebagai prediktor dan sebaiknya tidak diandalkan.
5. **KNN tidak menggantikan MLR.** KNN pada skor PCA (RMSE LORO 2,1768) dan KNN
   geospasial murni (2,4801) sama-sama kalah dari PCA + MLR (1,8929), bahkan dari
   PCA + MLR tanpa lag (1,7759). Nilai tambah ada pada **PCA + regresi**;
   kedekatan wilayah hanya berguna sebagai satu fitur tambahan.

Model akhir (latih 80 baris): m = **5 komponen** (variansi kumulatif 89,35 %),
k = **2 tetangga**, koefisien lag −0,3745, R² dalam sampel 0,7475.

---

## 2. Struktur proyek

```
pbl_estimasi/
├── app.py                     # dashboard Streamlit 7 tab
├── requirements.txt           # daftar pustaka (versi teruji)
├── README.md
├── docs/                      # tanggapan atas catatan reviewer
├── data/
│   ├── dataset_panel_kaltim_2016_2024.xlsx   # 90 baris x 16 kolom (3 sheet)
│   ├── catatan_mutu_data.csv                 # 4 catatan mutu data
│   ├── Kaltim_Kabupaten.geojson              # batas 10 kabupaten/kota
│   └── sumber_csv/                           # 90 CSV BPS yang dihimpun
├── src/
│   ├── config.py       # path relatif, daftar wilayah/fitur, parameter model
│   ├── data.py         # muat & validasi panel
│   ├── geo.py          # pemetaan nama, centroid UTM 50S, jarak haversine, KNN
│   ├── plots.py        # semua grafik matplotlib
│   ├── eda.py          # Fase 1: EDA + geospasial (python -m src.eda)
│   ├── model.py        # kelas EstimatorP0 + pemilihan k + metrik
│   ├── interpret.py    # Fase 4-5: uji asumsi, koefisien, artefak model
│   ├── knn.py          # pembanding KNN + klasifikasi 3 kelas
│   └── evaluate.py     # Fase 3-5: validasi, sensitivitas, model akhir
├── tools/
│   ├── bangun_dataset.py   # 90 CSV → xlsx (agregasi, tanpa membuat data baru)
│   └── uji_dashboard.py    # uji dashboard tanpa browser (AppTest)
├── models/                 # model_akhir.joblib, koefisien, loadings, metadata.json
└── outputs/                # tabel CSV + grafik PNG hasil EDA & evaluasi
```

Semua path dihitung **relatif** terhadap akar proyek (`src/config.py`), tidak ada
path absolut, sehingga folder dapat dipindah atau diletakkan di Git.

---

## 3. Cara menjalankan

Prasyarat: **Python 3.13** (di mesin pengembangan dipakai 3.13 dari
`python.org`; Python 3.15 belum mendukung sebagian pustaka geospasial).

```powershell
# 1) masuk folder proyek
cd pbl_estimasi

# 2) buat virtual environment dan pasang pustaka
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 3) bangun dataset dari 90 CSV BPS (sekali saja)
.\venv\Scripts\python.exe tools\bangun_dataset.py

# 4) Fase 1: validasi data, EDA, centroid, jarak, tetangga
.\venv\Scripts\python.exe -m src.eda

# 5) Fase 3-5: validasi LORO + pembagian waktu, uji asumsi, model akhir
.\venv\Scripts\python.exe -m src.evaluate

# 6) dashboard
.\venv\Scripts\python.exe -m streamlit run app.py
```

Uji cepat dashboard tanpa membuka browser:

```powershell
.\venv\Scripts\python.exe tools\uji_dashboard.py
```

Keluaran yang diharapkan dari `tools/uji_dashboard.py`:
`Tidak ada exception.` dengan **7 komponen tab, 21 tabel, 14 kartu metrik, 12 slider** (angka terakhir diperbarui 2026).

---

## 4. Sumber data dan cara dataset dibangun

### 4.1 Sumber
Seluruh data berasal dari publikasi **BPS Provinsi Kalimantan Timur** yang
tersedia sebagai 90 berkas CSV di `data/sumber_csv/`:

| Kelompok data | Berkas per tahun | Kolom panel |
|---|---|---|
| Jumlah Penduduk Miskin | 9 | `Jumlah_Penduduk_Miskin_Ribu` |
| Rata-rata pengeluaran bukan makanan per kapita sebulan | 9 | `Peng_NonMakanan_Rp` |
| Pengeluaran per kapita disesuaikan (Metode Baru) | 9 | `Peng_Disesuaikan_RibuRp` |
| Rata-rata lama sekolah (RLS, Metode Baru) | 9 | `RLS` |
| Harapan lama sekolah (HLS) perempuan | 9 | `HLS_Perempuan` |
| Umur harapan hidup (UHH, Metode Baru) | 9 | `UHH` |
| Indeks Pemberdayaan Gender (IDG) | 9 | `IDG` |
| Tingkat partisipasi angkatan kerja (TPAK) | 9 | `TPAK` |
| Tingkat pengangguran terbuka (TPT) | 9 | `TPT` |
| Penduduk, laju pertumbuhan, distribusi, kepadatan, rasio jenis kelamin | 9 | 5 kolom sekaligus |

Jumlah: **10 kelompok × 9 tahun (2016–2024) = 90 berkas**.

### 4.2 Pembangunan berkas panel
`tools/bangun_dataset.py` membaca 90 CSV tersebut dan menulis
`data/dataset_panel_kaltim_2016_2024.xlsx` dengan **90 baris × 16 kolom**:

`Kabupaten_Kota, Tahun, P0_Persen_Miskin, Peng_NonMakanan_Rp,
Peng_Disesuaikan_RibuRp, RLS, HLS_Perempuan, UHH, IDG, TPAK, TPT,
Jumlah_Penduduk_Ribu, Laju_Pertumbuhan_Penduduk, Kepadatan_per_km2,
Rasio_Jenis_Kelamin, Jumlah_Penduduk_Miskin_Ribu`

Sheet yang ditulis: `Data_Panel` (data), `Catatan_Mutu_Data` (4 catatan),
`Metadata` (dokumentasi sumber & aturan).

**Rumus target:** `P0_Persen_Miskin = Jumlah_Penduduk_Miskin_Ribu / Jumlah_Penduduk_Ribu × 100`
(dibulatkan 4 desimal). `Jumlah_Penduduk_Miskin_Ribu` adalah **turunan target**
sehingga **tidak pernah** dipakai sebagai fitur.

Verifikasi kelengkapan (hasil nyata):

| Tahun | Sel terisi | Sel kosong | Keterangan |
|---|---|---|---|
| 2016 | 120 / 160 | 40 | HLS, IDG, TPAK, TPT kosong untuk 10 wilayah |
| 2017–2024 | 160 / 160 setiap tahun | 0 | lengkap |

Periode pemodelan **80 baris** (10 wilayah × 8 tahun, 2017–2024). Angka kunci
yang sama dengan sumber: Paser 2024 P0 = **8,8084 %**, Samarinda 2021 = **5,1552 %**,
Penajam Paser Utara 2024 = **4,0269 %**.

### 4.3 Empat catatan mutu data (tidak ada imputasi)

| Wilayah | Tahun | Kolom | Jenis | Tindakan |
|---|---|---|---|---|
| Samarinda | 2021 | `Jumlah_Penduduk_Ribu` | salah ketik sumber (83,1 vs 831,3 ribu) | **dikoreksi → 831,0** |
| Penajam Paser Utara | 2024 | `Laju_Pertumbuhan_Penduduk` | pencilan 11,38 % (diduga terkait IKN) | dipertahankan, ditandai |
| Semua wilayah | 2019 | `Laju_Pertumbuhan_Penduduk` | definisi berbeda: akumulasi 2010–2019 (5,07–46,00 %) | dipertahankan, ditandai |
| Semua wilayah | 2020 | `Jumlah_Penduduk_Ribu` | basis Sensus Penduduk 2020 (tidak nyambung dengan 2019) | dipertahankan, ditandai |

Sel kosong **tidak diimputasi**; baris dengan sel kosong hanya dibuang bila
masuk periode pemodelan (tahun 2016 memang di luar periode model).


---

## 5. Metodologi

### 5.1 Standardisasi (Z-score)
`z = (x − μ_latih) / σ_latih`. μ dan σ **hanya dari data latih lipatan itu**.
Wajib dilakukan karena satuan fitur sangat berbeda (pengeluaran ~10⁵–10⁶ Rupiah
vs IDG ~puluhan vs rasio jenis kelamin ~110).

### 5.2 PCA (Principal Component Analysis)
PCA di-*fit* pada matriks Z data latih. Jumlah komponen **m** dipilih otomatis
sebagai jumlah komponen pertama yang membuat variansi kumulatif ≥ **85 %**
(`cfg.PCA_VAR_THRESHOLD`), dihitung **per lipatan**. Loading dilaporkan sebagai
`eigenvector × √eigenvalue`, yaitu korelasi antara indikator asli dan komponen.

- Model akhir (80 baris): m = **5**, variansi kumulatif **0,8935**
  (PC1 0,5126; PC2 0,1346; PC3 0,1064; PC4 0,0849; PC5 0,0550).
- Lima lipatan LORO lain memilih m = 4, sisanya m = 5 → m stabil di 4–5.
- Alasan PCA memang dibutuhkan: **VIF fitur mentah** RLS 36,768;
  Peng_Disesuaikan 26,521; Kepadatan 11,881; dan 7 pasangan fitur dengan
  |r| ≥ 0,80 (tertinggi: Peng_Disesuaikan–RLS = 0,911).

### 5.3 KNN geospasial → fitur *spatial lag*
1. **Centroid** setiap kabupaten/kota dihitung pada CRS terproyeksi
   **EPSG:32750 (UTM zona 50S)** supaya satuannya meter, lalu dikembalikan ke
   EPSG:4326 untuk ditampilkan. Contoh luas hasil hitung: Balikpapan 509,3 km²;
   Bontang 160,3 km²; Kutai Timur 31.511,5 km².
2. **Jarak antar centroid** memakai rumus **haversine** (R = 6.371,0088 km).
   Terdekat: Penajam Paser Utara–Balikpapan 31,3 km; terjauh: Paser–Berau 429,3 km.
3. **Spatial lag** sebuah baris (i, t):
   `Lag(i,t) = (1/k) · Σ_{j ∈ N_k(i)} P0(j, t)`
   dengan `N_k(i)` = k wilayah terdekat i.
4. **Aturan anti-kebocoran:** `N_k(i)` dipilih **hanya dari wilayah data latih**
   dan **tidak pernah memuat i sendiri**; nilai P0 yang dirata-ratakan juga
   hanya dari data latih. Untuk skema pembagian waktu, lag baris uji memakai
   P0 tetangga pada tahun latih terakhir (**2022**), karena P0 2023–2024 tidak
   boleh dilihat model.
5. **Pemilihan k** (`k ∈ {2, 3, 4}`) memakai *validasi dalam*: leave-one-region-out
   pada data latih saja. Pada model akhir terpilih **k = 2**
   (RMSE dalam 1,9135 vs k = 4 1,9236 dan k = 3 2,1872).

### 5.4 Regresi Linear Multivariabel (MLR)
`P0 = β₀ + β₁·PC1 + … + β_m·PCm + γ·SpatialLag + ε`

Empat varian yang dibandingkan (`cfg.MODEL`):

| Nama | Isi |
|---|---|
| `baseline` | rata-rata P0 data latih (tanpa fitur) |
| `mlr_tanpa_pca` | 12 fitur terstandardisasi + spatial lag |
| `pca_tanpa_lag` | m komponen PCA saja |
| `utama` | m komponen PCA + spatial lag |

### 5.5 Daftar periksa anti-kebocoran data (*data leakage*)
1. `P0_Persen_Miskin` dan `Jumlah_Penduduk_Miskin_Ribu` **tidak pernah** menjadi fitur.
2. Scaler dan PCA di-*fit* pada data latih lipatan saja.
3. Tetangga spatial lag dipilih dari data latih saja; wilayah uji tidak menjadi
   tetangganya sendiri (tetangga Samarinda saat uji: Bontang, Balikpapan, PPU —
   tanpa Samarinda).
4. `k` dipilih lewat validasi dalam pada data latih, bukan dengan melihat uji.
5. Skema pembagian waktu memakai lag tahun 2022 untuk baris 2023–2024.
6. `Jumlah_Penduduk_Ribu` diuji ulang pada analisis sensitivitas karena
   merupakan penyebut rumus P0.

### 5.6 Skema validasi
1. **LORO (utama).** 10 lipatan; tiap lipatan satu wilayah (8 baris) menjadi uji,
   9 wilayah lain (72 baris) menjadi latih. Semua 80 baris memperoleh estimasi
   *out-of-fold*.
2. **Pembagian waktu (tambahan).** Latih 2017–2022 (60 baris), uji 2023–2024
   (20 baris), lag dari P0 tetangga tahun 2022.
3. **Catatan metrik.** R² per wilayah bisa sangat negatif (mis. Berau −102,6)
   karena ragam P0 di dalam satu wilayah kecil, sehingga penyebut `SS_tot` kecil.
   Karena itu angka utama yang dilaporkan adalah **metrik gabungan 80 observasi**
   (`outputs/perbandingan_model.csv`), dan R² per wilayah disediakan sebagai
   diagnostik, bukan klaim performa.


---

## 6. Hasil

### 6.1 Detail pemilihan m dan k per lipatan LORO
| Lipatan (wilayah uji) | k utama | k MLR | m | Variansi kumulatif | RMSE dalam (k=2 / k=3 / k=4) |
|---|---|---|---|---|---|
| Paser | 2 | 4 | 5 | 0,8989 | 2,2599 / 2,4369 / 2,4990 |
| Kutai Barat | 4 | 4 | 5 | 0,8941 | 3,0377 / 2,7807 / 2,3356 |
| Kutai Kartanegara | 3 | 4 | 5 | 0,8998 | 3,4050 / 2,0182 / 2,0800 |
| Kutai Timur | 4 | 4 | 4 | 0,8512 | 1,9312 / 1,9033 / 1,7887 |
| Berau | 3 | 4 | 4 | 0,8654 | 1,7904 / 1,7418 / 1,7699 |
| Penajam Paser Utara | 4 | 4 | 4 | 0,8661 | 1,8333 / 1,8216 / 1,7454 |
| Mahakam Ulu | 4 | 4 | 5 | 0,8934 | 2,6224 / 2,6929 / 2,0945 |
| Balikpapan | 3 | 3 | 5 | 0,8849 | 2,0032 / 1,6771 / 1,7148 |
| Samarinda | 2 | 4 | 5 | 0,8823 | 2,0314 / 2,9334 / 2,2204 |
| Bontang | 4 | 2 | 5 | 0,8988 | 2,3654 / 2,4170 / 2,1272 |

(k = nilai terpilih pada tiap lipatan; angka RMSE dalam berasal dari validasi
dalam 9 lipatan.)

### 6.2 Metrik pooled per tahun (model utama)
| Skema | Tahun | R² | MAE | RMSE |
|---|---|---|---|---|
| LORO | 2017 | 0,4192 | 1,5154 | 1,9216 |
| LORO | 2018 | 0,4809 | 1,4518 | 1,9905 |
| LORO | 2019 | 0,3356 | 1,8948 | 2,2830 |
| LORO | 2020 | 0,5711 | 1,2939 | 1,5438 |
| LORO | 2021 | 0,5185 | 1,4097 | 1,6054 |
| LORO | 2022 | 0,3985 | 1,6062 | 1,7961 |
| LORO | 2023 | 0,4322 | 1,4879 | 1,7248 |
| LORO | 2024 | 0,1334 | 1,5166 | 2,1536 |
| Waktu | 2023 | 0,7587 | 0,9031 | 1,1244 |
| Waktu | 2024 | 0,5210 | 1,1058 | 1,6011 |

Tahun 2019 dan 2024 paling sulit diestimasi — konsisten dengan dua catatan mutu
data (definisi laju pertumbuhan 2019, pencilan PPU 2024).

### 6.3 PCA: variansi dan makna komponen
| Komponen | Eigenvalue | Variansi | Kumulatif | Loading dominan | Tafsiran |
|---|---|---|---|---|---|
| PC1 | 6,2290 | 0,5126 | 0,5126 | RLS +0,97; Kepadatan +0,92; Peng. disesuaikan +0,91 | poros **kesejahteraan/infrastruktur** (kota vs pedalaman) |
| PC2 | 1,6357 | 0,1346 | 0,6472 | IDG +0,77; Jumlah penduduk +0,51; HLS perempuan +0,44 | **pemberdayaan gender & skala penduduk** |
| PC3 | 1,2930 | 0,1064 | 0,7536 | TPAK +0,70; IDG +0,44; Laju pertumbuhan −0,44 | **partisipasi kerja** |
| PC4 | 1,0313 | 0,0849 | 0,8385 | Laju pertumbuhan +0,79; IDG +0,37; TPAK +0,31 | **dinamika demografi** (memuat kasus PPU 2024) |
| PC5 | 0,6683 | 0,0550 | 0,8935 | Rasio jenis kelamin +0,57; Laju pertumbuhan −0,35 | **struktur jenis kelamin** |

### 6.4 Koefisien model akhir (80 baris)
| Koefisien | Nilai | Galat baku | t | p | Galat baku robust | p robust |
|---|---|---|---|---|---|---|
| intersep | 9,009872 | 0,8143 | 11,0651 | <0,0001 | 1,6587 | <0,0001 |
| PC1 | −0,938101 | 0,0701 | −13,3731 | <0,0001 | 0,1641 | <0,0001 |
| PC2 | 0,199484 | 0,1198 | 1,6655 | 0,1001 | 0,2753 | 0,4687 |
| PC3 | 0,298471 | 0,1416 | 2,1074 | 0,0385 | 0,2918 | 0,3064 |
| PC4 | 0,081772 | 0,1456 | 0,5618 | 0,5760 | 0,2255 | 0,7169 |
| PC5 | 0,249577 | 0,1807 | 1,3813 | 0,1714 | 0,5748 | 0,6642 |
| SpatialLag | −0,374460 | 0,1349 | −2,7768 | 0,0070 | 0,2898 | 0,1963 |

Tanpa galat baku robust, PC1, PC3, dan spatial lag "signifikan". Setelah galat
baku **dikelompokkan per wilayah** (karena observasi panel tidak independen),
hanya **PC1** yang tetap signifikan. Kesimpulan yang jujur: **bukti statistik
untuk spatial lag lemah** pada data sekecil ini.

Koefisien dikembalikan ke variabel asli (loading × koefisien komponen), arah
negatif berarti indikator naik → P0 turun:

| Indikator | Koefisien efektif | Arah |
|---|---|---|
| Pengeluaran disesuaikan | −0,9622 | menurunkan P0 |
| RLS | −0,8994 | menurunkan P0 |
| Kepadatan penduduk | −0,8470 | menurunkan P0 |
| UHH | −0,7121 | menurunkan P0 |
| Pengeluaran bukan makanan | −0,7065 | menurunkan P0 |
| TPT | −0,6790 | menurunkan P0 |
| Jumlah penduduk | −0,6070 | menurunkan P0 |
| HLS perempuan | −0,6008 | menurunkan P0 |
| Laju pertumbuhan penduduk | −0,1627 | menurunkan P0 |
| IDG | +0,4184 | menaikkan P0 |
| Rasio jenis kelamin | +0,6340 | menaikkan P0 |
| TPAK | +0,6769 | menaikkan P0 |
| Spatial lag | −0,3745 | menurunkan P0 |

Arah TPAK dan IDG yang positif **tidak** boleh dibaca sebagai "pengangguran
menurunkan kemiskinan": itu efek setelah mengontrol PC lain (yang sudah memuat
RLS, pengeluaran, kepadatan), dan pada data agregat 10 wilayah arah seperti ini
lazim muncul sebagai efek komposisi. Ini keterbatasan model, bukan temuan kausal.

### 6.5 Uji asumsi klasik (model akhir)
| Uji | Statistik | p | Kesimpulan |
|---|---|---|---|
| Shapiro-Wilk (normalitas residual) | 0,9671 | 0,0373 | residual **tidak** normal pada α 5 % |
| Breusch-Pagan (heteroskedastisitas) | 4,2046 | 0,6490 | ragam galat homogen |
| Durbin-Watson (autokorelasi) | 0,6642 | – | ada autokorelasi positif kuat |
| Uji F keseluruhan | 36,0232 | <0,0001 | model signifikan |

Konsekuensi yang diterapkan: galat baku **robust berkelompok wilayah** disediakan
(`outputs/galat_baku_robust.csv`), dan kesimpulan tidak bersandar pada asumsi
normalitas/independensi. Autokorelasi positif wajar untuk panel wilayah-waktu
dan merupakan alasan tambahan untuk memakai metrik validasi LORO, bukan R²
dalam sampel.


### 6.6 Pembanding KNN dan klasifikasi kategori kemiskinan

**a) KNN sebagai regresi** (`outputs/knn_regresi_metrik.csv`; k dipilih lewat
LORO dalam pada data latih, kandidat 1-5).

| Skema | Varian KNN | k terpilih | R² | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|---|
| LORO | KNN pada skor PCA | 3-5 (modus 5) | 0,2389 | 1,8797 | 2,1768 | 36,14 |
| LORO | KNN geospasial murni (rata-rata P0 tetangga) | 1-5 (modus 2) | 0,0121 | 2,1950 | 2,4801 | 38,82 |
| Waktu | KNN pada skor PCA | 5 | 0,5586 | 1,2150 | 1,5377 | 25,19 |
| Waktu | KNN geospasial murni | 4 | −0,1349 | 2,0555 | 2,4658 | 50,44 |

Pembandingnya model utama PCA + lag + MLR: RMSE LORO **1,8929**, RMSE waktu
**1,3834**. Karena itu KNN **tidak** dipakai sebagai model akhir.

**b) Fitur spasial secara operasional** (`outputs/knn_fitur_spasial.csv`):
- **Definisi tetangga:** k wilayah dengan jarak **haversine** terkecil antar
  centroid (centroid dihitung pada EPSG:32750, jarak dihitung pada lintang-bujur).
- **Nilai k:** 2 (modus pilihan dalam LORO) untuk tabel fitur spasial; kandidat
  1-5 diuji pada setiap lipatan.
- **Contoh terverifikasi (Samarinda 2024):** tetangga **Bontang 73,01 km**
  (P0 3,7759) dan **Balikpapan 80,47 km** (P0 2,0259) → lag =
  (3,7759 + 2,0259) / 2 = **2,9009**. P0 Samarinda 2024 (4,4051) tidak masuk
  perhitungan ini karena wilayah tidak boleh menjadi tetangga dirinya sendiri.
- **Peran pada model:** satu kolom `SpatialLag` pada MLR dengan koefisien
  −0,3745; setelah galat baku robust berkelompok wilayah, p = 0,1963
  (tidak signifikan).

**c) KNN sebagai klasifikasi tiga kelas.** Ambang kelas = kuantil 1/3 dan 2/3 P0
**data latih tiap lipatan** (bukan ambang resmi BPS), sehingga kelas pada data
penuh seimbang: 27/26/27.

| Skema | Varian | Akurasi | Akurasi seimbang | Macro F1 | Baseline kelas mayoritas |
|---|---|---|---|---|---|
| LORO | KNN pada skor PCA (k modus 3) | 0,6500 | 0,6167 | 0,6170 | 0,1250 |
| LORO | KNN geografis (kelas k tetangga, k modus 1) | 0,4500 | 0,4444 | 0,4423 | 0,1250 |
| LOOCV baris | KNN pada skor PCA | 0,7375 | 0,7369 | 0,7389 | 0,3375 |

Confusion matrix LORO, KNN pada skor PCA (`outputs/knn_confusion_loro_pca.csv`):

| Aktual / Prediksi | Rendah | Menengah | Tinggi |
|---|---|---|---|
| **Rendah** | 22 | 6 | 2 |
| **Menengah** | 0 | 7 | 13 |
| **Tinggi** | 4 | 3 | 23 |

Metrik per kelas (LORO): presisi/recall/F1 = **Rendah** 0,846/0,733/0,786;
**Menengah** 0,438/0,350/0,389; **Tinggi** 0,605/0,767/0,677.

Tafsiran yang dipakai:
1. **KNN geografis 45 % setara dengan laporan awal (50 %)**: kelemahannya bukan
   implementasi, melainkan informasi - kelas wilayah bertetangga tidak cukup mirip.
2. KNN pada skor PCA lebih baik (65 % LORO; 74 % LOOCV baris), tetapi tetap di
   bawah model regresi untuk tujuan **estimasi nilai** (RMSE).
3. Baseline "kelas mayoritas" pada LORO hanya 12,5 % karena ambang kelas relatif
   per lipatan: satu wilayah umumnya jatuh pada satu kelas, sehingga baseline itu
   tidak sebanding dengan baseline 33,75 % pada LOOCV baris. Akurasi harus
   dibandingkan dengan baseline pada skema yang sama.
4. Kelas **Menengah** paling sulit (F1 0,389); kesalahan bergeser ke kelas
   tetangga, bukan acak.
5. Dengan 10 wilayah, satu kesalahan klasifikasi setara 8 observasi. Angka ini
   **indikasi awal**, bukan dasar penentuan kebijakan.

### 6.7 Audit target P0 (rumus, pembilang, penyebut)

| Butir | Isi |
|---|---|
| Rumus | `P0 = (Jumlah_Penduduk_Miskin_Ribu / Jumlah_Penduduk_Ribu) × 100`, dibulatkan 4 desimal |
| Pembilang | tabel **Jumlah Penduduk Miskin menurut Kabupaten/Kota** (Susenas, ribu jiwa, 2 desimal) |
| Penyebut | tabel **Penduduk, Laju Pertumbuhan Penduduk, ...** (proyeksi/SP2020, ribu jiwa) |
| Cakupan | 2016-2024 = 90 sel; pemodelan 2017-2024 = 80 sel |
| Konsistensi | selisih maksimum panel vs hitung ulang **0,00005 poin** → murni pembulatan |
| Contoh | Paser 2024: 25,28 / 287,0 × 100 = **8,8084**; Samarinda 2021: 42,84 / 831,0 × 100 = **5,1552** |
| Berkas | `outputs/audit_target_p0.csv` (memuat nama berkas sumber pembilang dan penyebut) |

Kolom `P0_resmi_BPS` pada CSV itu sengaja **dikosongkan** agar dapat diisi angka
publikasi BPS; dengan begitu selisih terhadap angka resmi dapat **ditunjukkan
terukur**, bukan diklaim. Empat sebab kemungkinan selisih (juga tertulis pada
kolom terakhir CSV):
1. pembilang = penduduk miskin hasil Susenas periode September, penyebut =
   penduduk tahunan (proyeksi/SP2020);
2. pembulatan pembilang 2 desimal (ribu jiwa) dan hasil 4 desimal;
3. P0 resmi memakai penimbang Susenas, sedangkan di sini rasio agregat;
4. revisi/backcasting angka historis BPS antar publikasi.

---

## 7. Pembahasan jujur: apa yang berhasil dan apa yang tidak

**Berhasil**
1. PCA menyelesaikan masalah multikolinearitas: VIF PC turun ke 1,00–1,67
   (dari 36,77 pada RLS), dan RMSE LORO membaik dari 3,4477 (tanpa PCA)
   menjadi 1,7759–1,8929.
2. Model utama mengalahkan baseline pada kedua skema (RMSE 1,8929 vs 2,7541
   pada LORO; 1,3834 vs 2,4297 pada uji waktu).
3. Pada uji waktu 2023–2024, model utama **terbaik di antara semua varian**
   (R² 0,6427). Spatial lag memang paling relevan ketika model harus
   mengekstrapolasi ke tahun yang belum dilihat.
4. Loading PC1 sangat mudah ditafsirkan (RLS, kepadatan, pengeluaran) dan
   koefisiennya sangat stabil (p < 0,0001 pada OLS biasa maupun robust).

**Belum berhasil / perlu kejujuran**
1. Pada LORO, spatial lag **tidak** memperbaiki RMSE (1,8929 vs 1,7759 tanpa lag).
   Dugaan penyebab: hanya 10 wilayah sehingga `N_k(i)` sangat sedikit, dan
   P0 antar wilayah bertetangga di Kaltim tidak seragam (kota pesisir rendah,
   pedalaman tinggi) sehingga rata-rata tetangga menjadi bising.
2. Koefisien spatial lag **negatif** dan **tidak signifikan** setelah galat baku
   robust (`p = 0,1963`). Artinya klaim "kemiskinan menyebar ke tetangga"
   **tidak didukung kuat** oleh data ini.
3. Residual tidak normal (Shapiro-Wilk p = 0,0373) dan berautokorelasi
   (DW = 0,6642).
4. Membuang `Jumlah_Penduduk_Ribu` sedikit memperbaiki hasil (RMSE 1,8513),
   menandakan variabel itu lemah sebagai prediktor.
5. Akurasi absolut masih sedang: MAPE LORO 27,09 %; galat absolut rata-rata
   ±1,5 poin persentase. Estimasi model **bukan pengganti** survei resmi.
6. **KNN bukan model terbaik.** RMSE LORO 2,1768 (KNN pada skor PCA) dan 2,4801
   (KNN geografis) vs 1,8929 model utama. Sebagai klasifikator, KNN geografis
   hanya 45 % - sehingga tidak layak disebut "model siap pakai".

Karena itu dashboard selalu memberi label *estimasi*, menampilkan galat, dan
menyediakan perbandingan antar model — bukan hanya satu angka tunggal.

---

## 8. Keputusan desain (design decisions)

1. **Dataset dibangun ulang dari 90 CSV BPS** (`tools/bangun_dataset.py`) karena
   berkas `dataset_panel_kaltim_2016_2024.xlsx` tidak tersedia. Proses ini
   bersifat transparan dan hanya menghimpun sel yang sudah ada — tanpa
   membuat, menginterpolasi, atau menambahkan data baru. Berkas lama
   `Dataset_Terpadu_Kaltim.xlsx` (hanya 2023) tidak dipakai.
2. **P0 dihitung sendiri** = penduduk miskin / penduduk × 100 dan dibulatkan
   4 desimal; `P0_Persen_Miskin` dan `Jumlah_Penduduk_Miskin_Ribu` dilarang
   menjadi fitur untuk mencegah kebocoran data.
3. **Periode model 2017–2024 (80 baris)**, tahun 2016 dikeluarkan karena HLS,
   IDG, TPAK, dan TPT kosong untuk seluruh 10 wilayah (hanya tersedia tingkat provinsi).
4. **Satu koreksi data:** Samarinda 2021 `Jumlah_Penduduk_Ribu` 83,1 → 831,0 ribu
   jiwa. Dasar: kolom distribusi penduduk provinsi (21,83 % × 3.808,2 ribu =
   831,3 ribu). Koreksi ini dibatasi pada satu sel dan dicatat di
   `data/catatan_mutu_data.csv`.
5. **Pencilan dan diskontinuitas dipertahankan** (PPU 2024 11,38 %; definisi
   laju pertumbuhan 2019; basis penduduk SP2020) — hanya ditandai, tidak diubah,
   karena koreksi lebih lanjut tidak dapat dipertanggungjawabkan tanpa data mentah.
6. **Pemilihan m per lipatan** memakai ambang variansi kumulatif ≥ 85 % pada
   data latih lipatan itu, bukan angka tetap dari seluruh data.
7. **Pemilihan k (2, 3, 4) memakai validasi dalam LORO pada data latih.** Proses
   ini berulang di tiap lipatan sehingga tetap dapat agak optimistis; karena itu
   angka hasilnya dilaporkan sebagai batas atas kejujuran, dan model akhir
   memilih k = 2.
8. **Spatial lag memakai P0 tetangga pada tahun yang sama** (cross-section
   spasial), sementara untuk baris uji 2023–2024 dipakai P0 2022 — satu-satunya
   pilihan yang tidak melanggar aturan tidak melihat label masa depan.
9. **Centroid dihitung pada EPSG:32750 lalu dikembalikan ke EPSG:4326**, agar
   jarak dihitung dengan satuan meter pada proyeksi yang tepat untuk Kalimantan
   Timur, tetapi tetap disajikan dalam lintang-bujur.
10. **Jarak memakai haversine** (bukan jarak Euclidean antar-centroid dalam
    derajat) supaya skala lintang-bujur tidak mendistorsi urutan tetangga.
11. **Poligon tanpa nama pada GeoJSON dibuang** (11 fitur → 10 wilayah dikenal);
    pemetaan nama `Kota Samarinda`, `Kota Balikpapan`, `Kota Bontang` → nama baku
    dataset di `src/geo.py`, dengan tabel bukti di `outputs/pemetaan_nama_wilayah.csv`.
12. **`random_state = 42`** pada semua komponen yang memakai keacakan, dan seluruh
    path bersifat **relatif** terhadap akar proyek, sehingga hasil dapat direproduksi
    di mesin lain.


---

## 9. Batasan penelitian

1. **Jumlah wilayah sangat kecil (10).** Setiap lipatan LORO hanya menyisakan
   9 wilayah latih; lag dibangun dari ≤ 9 tetangga. Kesimpulan statistik
   (terutama signifikansi) harus dibaca konservatif.
2. **R² per wilayah tidak stabil** karena ragam P0 dalam satu wilayah kecil.
3. **Observasi panel tidak independen** (antar-tahun berkorelasi; DW = 0,6642),
   sehingga galat baku robust berkelompok wilayah lebih layak dipakai.
4. **Tidak ada klaim kausal.** Koefisien MLR hanyalah asosiasi pada periode
   2017–2024; simulasi di dashboard menahan spatial lag tetap dan tidak
   memodelkan umpan balik antar-wilayah.
5. **Data agregat level kabupaten/kota**, bukan rumah tangga; tidak dapat
   menjelaskan kemiskinan individual.
6. **Definisi variabel berubah antar tahun** (laju pertumbuhan 2019, basis
   penduduk SP2020), memengaruhi ketepatan estimasi tahun tersebut.
7. **Rentang slider simulasi dibatasi** pada min–maks data 2017–2024 untuk
   menghindari ekstrapolasi yang tidak dapat dipertanggungjawabkan.
8. **Agnostik terhadap kebijakan**: model tidak memuat variabel program
   pemerintah (bansos, infrastruktur spesifik), sehingga tidak tepat dipakai
   untuk mengklaim efek suatu intervensi.

---

## 10. Dashboard (7 tab)

| Tab | Isi |
|---|---|
| **1 · Peta** | Choropleth P0 per tahun pada batas kabupaten/kota; pilihan nilai **aktual**, **estimasi LORO (out-of-fold)**, atau **estimasi dalam sampel**; warna 5 kelas kuantil; tooltip memuat nilai aktual dan selisihnya. |
| **1b · Audit P0** (di dalam tab 1) | Tabel penelusuran target: pembilang, penyebut, hasil hitung, selisih terhadap panel, nama berkas sumber, dan kolom kosong `P0_resmi_BPS` untuk diisi angka resmi. |
| **2 · Estimasi vs Aktual** | Grafik garis aktual vs 4 model (bisa per wilayah atau rata-rata 10 wilayah), tabel galat, dan MAE/galat maksimum per model. |
| **3 · Hasil PCA** | Scree plot, variansi kumulatif, heatmap loading, tabel explained variance, dan tafsiran tiap komponen; menampilkan m dan ambang 85 %. |
| **4 · Model & Evaluasi** | Tabel perbandingan model, koefisien MLR, galat baku robust, koefisien dikembalikan ke variabel asli, uji asumsi, metrik per wilayah, dan sensitivitas tanpa fitur penduduk. |
| **5 · Prioritas Wilayah** | Peringkat wilayah menurut estimasi P0 tahun terpilih, lengkap dengan diagram batang dan nilai aktual sebagai pembanding. |
| **6 · Simulasi** | Slider 12 indikator untuk satu wilayah-tahun; menghitung ulang skor PCA memakai scaler/PCA tersimpan dengan **lag ditahan tetap**, lalu menampilkan perubahan estimasi dan kontribusi tiap komponen (PC). |
| **7 · KNN & Klasifikasi** | Tabel fitur spasial (tetangga, jarak, P0 tetangga, lag), perbandingan model KNN, confusion matrix interaktif, akurasi/akurasi seimbang/macro F1, metrik per kelas, dan detail prediksi per wilayah-tahun. |

Dashboard menampilkan pemberitahuan bila berkas keluaran belum ada, beserta
tiga perintah yang harus dijalankan berurutan.

---

## 11. Daftar berkas keluaran

**Tabel (`outputs/`)**

| Berkas | Isi |
|---|---|
| `kelengkapan_data.csv` | kelengkapan sel per tahun |
| `eda_deskriptif.csv` | statistik deskriptif 12 fitur + P0 |
| `eda_korelasi_heatmap.csv` | matriks korelasi |
| `eda_vif_12fitur.csv` | VIF fitur mentah |
| `eda_p0_lebar.csv` | P0 per wilayah × tahun |
| `pemetaan_nama_wilayah.csv` | pemetaan nama GeoJSON ↔ dataset |
| `centroid_wilayah.csv`, `jarak_antar_wilayah.csv` | centroid & jarak (km) |
| `detail_lipatan.csv` | m dan k terpilih per lipatan LORO |
| `perbandingan_model.csv` | metrik semua model (LORO & waktu) |
| `metrik_per_wilayah.csv`, `metrik_per_tahun.csv` | diagnostik per wilayah/tahun |
| `estimasi_lipatan.csv` | estimasi out-of-fold (lebar, per model) |
| `estimasi_lipatan_panjang.csv` | hasil estimasi format panjang |
| `sensitivitas_tanpa_penduduk.csv` | uji tanpa `Jumlah_Penduduk_Ribu` |
| `pca_varians_model_akhir.csv`, `pca_loadings_model_akhir.csv`, `pca_interpretasi_komponen.csv` | hasil PCA |
| `koefisien_model_akhir.csv`, `galat_baku_robust.csv` | koefisien & robust SE |
| `koefisien_variabel_asli.csv` | loading × koefisien (per indikator) |
| `vif_fitur_mentah.csv`, `vif_komponen.csv` | perbandingan multikolinearitas |
| `uji_asumsi.csv` | Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, uji F |
| `estimasi.csv` | estimasi P0 per wilayah-tahun + galat |
| `audit_target_p0.csv` | audit target: pembilang, penyebut, P0 hitung, berkas sumber, kolom kosong untuk angka resmi |
| `knn_regresi_detail.csv`, `knn_regresi_metrik.csv`, `knn_tuning_k_regresi.csv` | pembanding KNN regresi + pemilihan k |
| `knn_fitur_spasial.csv` | tetangga, jarak, P0 tetangga, dan lag per wilayah-tahun |
| `knn_klasifikasi_detail.csv`, `knn_klasifikasi_metrik.csv`, `knn_metrik_per_kelas.csv`, `knn_tuning_k_klasifikasi.csv` | klasifikasi 3 kelas + metrik per kelas |
| `knn_confusion_loro_pca.csv`, `knn_confusion_loro_spasial.csv`, `knn_confusion_loocv_baris_pca.csv`, `knn_confusion_loro_baseline.csv` | confusion matrix per skema/varian |

**Grafik (`outputs/`)**: `eda_korelasi_heatmap.png`, `eda_tren_p0.png`,
`eda_boxplot_pencilan.png`, `pca_scree_plot.png`, `pca_loading_heatmap.png`,
`plot_aktual_vs_estimasi_loro.png`, `plot_aktual_vs_estimasi_waktu.png`,
`plot_residual_model_akhir.png`, `perbandingan_model_rmse_loro.png`,
`perbandingan_model_rmse_waktu.png`, `knn_confusion_loro_pca.png`,
`knn_confusion_loro_spasial.png`, `knn_confusion_loocv_baris_pca.png`,
`knn_confusion_loro_baseline.png`, `knn_confusion_loocv_baris_baseline.png`.

**Model (`models/`)**: `model_akhir.joblib` (scaler + PCA + koefisien),
`koefisien_mlr.csv`, `loadings_pca.csv`, `koefisien_variabel_asli.csv`,
`centroid_wilayah.csv`, `jarak_antar_wilayah.csv`, `daftar_tetangga.csv`,
`metadata.json`.

---

## 12. Istilah dan referensi

- **P0** — persentase penduduk miskin di bawah garis kemiskinan.
- **RLS/HLS** — rata-rata lama sekolah / harapan lama sekolah (tahun).
- **UHH** — umur harapan hidup (tahun). **IDG** — indeks pemberdayaan gender.
- **TPAK/TPT** — tingkat partisipasi angkatan kerja / tingkat pengangguran terbuka (%).
- **Spatial lag** — rata-rata nilai P0 wilayah tetangga, dipakai sebagai fitur.
- **LORO** — *leave-one-region-out*, skema validasi silang pada level wilayah.
- **Out-of-fold** — estimasi untuk baris yang tidak dipakai saat melatih model itu.

Tambahan istilah untuk bagian KNN:
- **KNN** - *k-nearest neighbors*; dipakai sebagai pembangun fitur *spatial lag*,
  pembanding regresi, dan klasifikator.
- **Confusion matrix** - tabel silang kelas aktual x kelas prediksi.
- **Akurasi seimbang** - rata-rata recall tiap kelas (adil untuk kelas timpang).
- **Macro F1** - rata-rata F1 tiap kelas tanpa pembobotan jumlah observasi.

### 12.1 Sumber data
Tabel publikasi **BPS Provinsi Kalimantan Timur** (https://kaltim.bps.go.id),
90 berkas CSV di `data/sumber_csv/`; nama berkas mengikuti judul tabel aslinya
sehingga dapat dicari ulang. Tahun data 2016-2024; pemodelan 2017-2024.

### 12.2 Referensi metode (dapat diverifikasi)
1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Dordrecht: Kluwer.
2. Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Cham: Springer.
3. Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (1996). Geographically weighted regression: a method for exploring spatial nonstationarity. *Geographical Analysis*, 28(4), 281-298.
4. Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.
5. Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Berlin: Springer.
6. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). New York: Springer.
7. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.
8. LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. Boca Raton: CRC Press.
9. Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17-23.
10. Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B*, 36(2), 111-147.

Catatan: referensi di atas adalah rujukan metode yang sudah mapan. Untuk laporan,
tambahkan 5-10 artikel terbaru (2019-2024) tentang pemetaan kemiskinan, KNN, dan
analisis panel dari jurnal Indonesia (Sinta) atau internasional, dan **hapus**
referensi yang tidak dipakai - termasuk entri "Sepeda (Bike Sharing)" yang tidak
terkait topik.

### 12.3 Pustaka perangkat lunak
pandas, NumPy, scikit-learn, statsmodels, GeoPandas, Shapely, pyproj, Folium,
Matplotlib, Plotly, Streamlit (versi teruji pada `requirements.txt`).

---

## 13. Tanggapan atas catatan reviewer

Rincian per butir (kutipan komentar, tindakan, lokasi bab, dan angka pendukung)
ada di **`docs/TANGGAPAN_REVIEWER.md`**. Ringkasnya:

| Catatan | Status di repositori ini |
|---|---|
| Judul/abstrak menyebut MLR dan KNN, hasil hanya KNN | Hasil MLR lengkap: R², MAE, RMSE, MAPE, uji asumsi, galat baku robust (§1, §6.4, §6.5); KNN diuji sebagai pembanding (§6.6) |
| Peran KNN geospasial tidak jelas | Definisi tetangga, jarak, nilai k, dan contoh hitung lag (§5.3, §6.6b) |
| Target P0 tidak dijelaskan | Audit pembilang/penyebut + selisih pembulatan (§6.7, `outputs/audit_target_p0.csv`) |
| Akurasi KNN 50 % ditafsirkan berlebihan | Confusion matrix, metrik per kelas, dan baseline kelas mayoritas (§6.6c) |
| GUI masih berupa rancangan | Dashboard 7 tab berjalan dan diuji otomatis (`tools/uji_dashboard.py`) |
| Periode 2017-2024 vs tabel 2023 | Pemodelan 80 observasi 2017-2024; 2016 dikeluarkan dengan alasan terdokumentasi (§4.2, §8) |
| PCA ditafsirkan kausal | Loading disebut eksploratif; tidak ada klaim kausal (§6.3, §7, §9) |
| Pustaka minim, ada entri tidak relevan | Daftar rujukan metode + instruksi penambahan/penghapusan (§12.2) |

*Dokumen ini dan seluruh angka di dalamnya dihasilkan dari eksekusi nyata
`tools/bangun_dataset.py`, `python -m src.eda`, dan `python -m src.evaluate`
pada mesin pengembangan (Python 3.13, Windows). Bila data atau parameter
diubah, jalankan ulang ketiga perintah tersebut agar README dan dashboard tetap
konsisten.*

