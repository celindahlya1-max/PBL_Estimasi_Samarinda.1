# Tanggapan atas Catatan Reviewer — PBL Estimasi Kemiskinan Kalimantan Timur

Dokumen ini memetakan **setiap catatan reviewer** ke tindakan yang sudah
dilakukan, bukti berupa berkas dan angka, serta teks yang dapat langsung
disalin ke naskah laporan. Seluruh angka berasal dari eksekusi nyata
di folder `pbl_estimasi/` (bukan perkiraan).

Perintah yang menghasilkan seluruh angka:

```powershell
python tools\bangun_dataset.py
python -m src.eda
python -m src.evaluate
python tools\uji_dashboard.py
```

---

## 0. Ringkasan status

| Kelompok catatan | Skor reviewer | Status |
|---|---|---|
| Konsistensi judul, abstrak, dan hasil | 72 | **Selesai di repositori**, perlu disalin ke naskah (bagian 2, 3, 4) |
| Orisinalitas dan kontribusi | 68 | Perlu paragraf pembanding penelitian terdahulu (bagian 2.B2) |
| Metodologi | 55 | **Selesai**: rumus P0, spatial lag, PCA eksploratif, KNN (bagian 4) |
| Hasil dan pembahasan | 60 | **Selesai**: R², MAE, RMSE, MAPE, uji asumsi, confusion matrix (bagian 5) |
| Kesimpulan dan keterbatasan | 58 | Perlu paragraf keterbatasan yang jujur (bagian 6) |
| Sistematika, pustaka, tata tulis | 48 | Sebagian: daftar rujukan tersedia; penyuntingan naskah manual (bagian 7, 8) |

Tiga keputusan yang harus diambil **sebelum** menyalin angka:

1. **KNN dipertahankan atau dihapus dari judul?** Repositori ini mendukung dua
   pilihan: KNN sebagai pembanding (angka tersedia) atau KNN dihapus dari judul
   dan hanya muncul di bagian pembahasan. Rekomendasi: **pertahankan sebagai
   pembanding** dengan judul yang menyebut "perbandingan", karena reviewer
   mempertanyakan hilangnya KNN.
2. **Periode:** gunakan **2017–2024 (80 observasi)**. Tahun 2016 tidak lengkap
   (HLS, IDG, TPAK, TPT kosong pada 10 wilayah), sehingga tidak dapat dipakai
   tanpa imputasi. Judul dan tabel wajib memakai periode yang sama.
3. **Deliverable:** revisi naskah + dokumen tanggapan ini (dapat dilampirkan
   sebagai lampiran "response to reviewer").

---

## 1. Peta perubahan angka (wajib diselaraskan)

Naskah lama dan implementasi ini **tidak** memakai pipeline yang sama. Angka lama
harus diganti, bukan dicampur.

| Klaim pada naskah lama | Angka pada implementasi ini | Bukti |
|---|---|---|
| 3 komponen utama menjelaskan 95,30 % variansi | **5 komponen, 89,35 %** (ambang 85 %, 12 fitur, 80 observasi) | `outputs/pca_varians_model_akhir.csv`, `models/metadata.json` |
| Akurasi KNN 50 % (10 wilayah) | **KNN geografis 45,00 %** (LORO); KNN pada skor PCA **65,00 %** (LORO) / **73,75 %** (LOOCV baris) | `outputs/knn_klasifikasi_metrik.csv` |
| Tabel hasil hanya tahun 2023 | Panel **2017–2024, 80 observasi**, 10 wilayah | `outputs/eda_p0_lebar.csv` |
| GUI masih berupa rancangan (wireframe) | **Dashboard 7 tab berjalan**, 0 exception | `tools/uji_dashboard.py` |
| Hasil MLR tidak dilaporkan | R² **0,4245**; MAE **1,5220**; RMSE **1,8929** (LORO) | `outputs/perbandingan_model.csv` |

Catatan penting: bila angka 95,30 % berasal dari fitur/periode yang berbeda,
jangan dipertahankan. Jelaskan di naskah bahwa PCA dijalankan ulang pada 12 fitur
dan 80 observasi, lalu laporkan angka baru beserta ambang pemilihan komponen.

---

## 2. Tanggapan per butir

### A. Catatan umum: konsistensi dan kelengkapan

**A1. Judul/abstrak menjanjikan MLR dan KNN, tetapi hasil yang disajikan hanya KNN.**

- Tindakan: hasil MLR dilaporkan lengkap (koefisien, galat baku, uji asumsi,
  metrik validasi), dan KNN dilaporkan sebagai pembanding pada skema validasi
  yang sama.
- Bukti: `outputs/koefisien_model_akhir.csv`, `outputs/uji_asumsi.csv`,
  `outputs/galat_baku_robust.csv`, `outputs/perbandingan_model.csv`,
  `outputs/knn_regresi_metrik.csv`.
- Teks siap pakai: lihat bagian 4.6 dan tabel pada bagian 5.

**A2. Periode penelitian tidak konsisten (judul 2017–2024, tabel hanya 2023).**

- Tindakan: seluruh analisis memakai **2017–2024 = 80 observasi** (10 wilayah x
  8 tahun). Tahun 2016 dikeluarkan dan alasannya didokumentasikan.
- Bukti: `data/Metadata` pada xlsx (butir "Cakupan"), `src/config.py`
  (`TAHUN_MODEL`), `outputs/kelengkapan_data.csv` (2016: 100 dari 140 sel terisi).
- Teks siap pakai: "Penelitian memakai data panel 10 kabupaten/kota Provinsi
  Kalimantan Timur periode 2017–2024 (80 observasi). Tahun 2016 tidak digunakan
  karena empat indikator (HLS perempuan, IDG, TPAK, TPT) tidak tersedia pada
  tingkat kabupaten/kota; tidak dilakukan imputasi."

**A3. Jumlah observasi dan dataset panel harus disajikan lengkap.**

- Tindakan: dataset panel 90 baris x 16 kolom dibangun dari 90 berkas CSV BPS dan
  diverifikasi (0 duplikat, 0 sel kosong pada periode model).
- Bukti: `tools/bangun_dataset.py`, `outputs/kelengkapan_data.csv`,
  `outputs/eda_deskriptif.csv`.

### B. Enam aspek penilaian

**B1. Kontribusi sistem/dashboard dinilai belum optimal; GUI masih berupa rancangan (wireframe).**

- Tindakan: dashboard **sudah diimplementasikan** dan dijalankan otomatis. Isinya
  7 tab: Peta (choropleth + audit P0), Estimasi vs Aktual, Hasil PCA,
  Model & Evaluasi, Prioritas Wilayah, Simulasi, serta KNN & Klasifikasi.
- Bukti eksekusi: `python tools\uji_dashboard.py` menghasilkan
  "**Tidak ada exception**", **7 komponen tab, 21 tabel, 14 kartu metrik,
  12 slider**, durasi 15,9 detik. Dashboard juga memuat grafik PNG hasil model.
- Yang harus dilakukan: **ambil tangkapan layar 7 tab** dari
  `streamlit run app.py`, beri nomor Gambar, dan ganti seluruh gambar wireframe
  pada naskah. Sertakan satu paragraf pengujian: "Dashboard diuji dengan
  Streamlit AppTest; seluruh tab dieksekusi tanpa exception."
- Jangan menulis "GUI masih dalam rancangan". Tulis: "Sistem diimplementasikan
  sebagai dashboard Streamlit yang telah diuji fungsionalitasnya."

**B2. Orisinalitas dan posisi terhadap penelitian terdahulu belum jelas.**

- Tindakan: kontribusi dirumuskan menjadi tiga hal yang dapat diverifikasi:
  1. panel 10 kabupaten/kota 2017–2024 yang dibangun ulang dari 90 tabel BPS
     dengan audit target (pembilang/penyebut) eksplisit;
  2. penggunaan **spatial lag dari k tetangga terdekat** sebagai fitur, dengan
     aturan anti-kebocoran yang dinyatakan (tetangga hanya dari data latih);
  3. penyajian hasil sebagai dashboard yang menampilkan galat, bukan satu angka.
- Teks siap pakai (letakkan di akhir pendahuluan):
  "Berbeda dari studi pemetaan kemiskinan yang umumnya berhenti pada
  pengelompokan wilayah atau satu model regresi, penelitian ini (i) menyusun
  panel 10 kabupaten/kota 2017–2024 dari tabel publikasi BPS dan mengaudit
  pembentukan target P0, (ii) membandingkan empat varian model (baseline, MLR
  tanpa PCA, PCA tanpa lag, PCA + spatial lag) pada skema validasi yang sama,
  serta (iii) menyediakan dashboard yang melaporkan galat estimasi."
- Wajib ditambahkan: 5–8 penelitian terdahulu yang benar-benar dibaca, dengan
  kalimat pembanding seperti "Studi X memakai KNN untuk klasifikasi tingkat
  kemiskinan dan melaporkan akurasi ...; penelitian ini memakai KNN sebagai
  pembanding dan menunjukkan bahwa regresi PCA menghasilkan galat lebih kecil."

**B3. Hasil regresi linear multivariabel tidak disajikan.**

- Tindakan: hasil MLR dilaporkan lengkap, termasuk uji asumsi dan galat baku
  robust berkelompok wilayah.
- Angka kunci (LORO, 80 observasi): R² **0,4245**; MAE **1,5220**; RMSE **1,8929**;
  MAPE **27,09 %**. Dalam sampel: R² **0,7475**; RMSE **1,2537**.
- Koefisien: intersep 9,0099; PC1 −0,9381 (p < 0,0001); PC2 0,1995 (p = 0,1001);
  PC3 0,2985 (p = 0,0385); PC4 0,0818 (p = 0,5760); PC5 0,2496 (p = 0,1714);
  SpatialLag −0,3745 (p = 0,0070; robust p = **0,1963**).
- Uji asumsi: Shapiro-Wilk p = **0,0373** (residual tidak normal);
  Breusch-Pagan p = 0,6490 (homoskedastik); Durbin-Watson **0,6642**
  (autokorelasi positif); uji F p < 0,0001.
- Bukti: `outputs/koefisien_model_akhir.csv`, `outputs/galat_baku_robust.csv`,
  `outputs/uji_asumsi.csv`, `outputs/perbandingan_model.csv`.

**B4/B6. Sistematika penulisan, penomoran, dan salah ketik.**

- Tindakan: daftar perbaikan teknis tersedia pada bagian 7 dokumen ini.
- Contoh yang harus diperbaiki: "TINJUAN PUSTAKA" -> "TINJAUAN PUSTAKA";
  "Graphicel" -> "Graphical"; "Perincipal Component Abalysis" ->
  "Principal Component Analysis"; identitas kelompok pada halaman judul harus
  sama dengan daftar anggota; penomoran bab/subbab dan rujukan "Tabel 4.1"
  harus cocok dengan isi.

**B5. Pustaka minim dan ada entri yang tidak relevan.**

- Tindakan: daftar rujukan metode yang dapat diverifikasi disediakan (bagian 8),
  dan entri **"Sepeda (Bike Sharing)" dihapus** karena tidak dipakai.
- Aturan: setiap rujukan pada daftar pustaka wajib disitasi di batang tubuh, dan
  setiap sitasi wajib ada di daftar pustaka (cek dengan fitur "Find" pada Word).

---

## 3. Catatan substantif (C) dan jawabannya

### C1. Target P0: rumus, sumber, dan perbedaan dengan angka resmi BPS

- Rumus: **P0 = (jumlah penduduk miskin / jumlah penduduk) x 100**, 4 desimal.
- Pembilang: tabel *Jumlah Penduduk Miskin menurut Kabupaten/Kota* (Susenas,
  ribu jiwa). Penyebut: tabel *Penduduk, Laju Pertumbuhan Penduduk, ...*
  (proyeksi/SP2020, ribu jiwa). Nama berkas sumber tercatat per baris pada
  `outputs/audit_target_p0.csv`.
- Konsistensi: selisih maksimum panel terhadap hitung ulang **0,00005 poin**
  (murni pembulatan). Contoh: Paser 2024 = 25,28 / 287,0 x 100 = **8,8084**;
  Samarinda 2021 = 42,84 / 831,0 x 100 = **5,1552**; PPU 2024 = 10,78 / 267,7 x 100
  = **4,0269**.
- Perbedaan dengan angka resmi BPS dijelaskan oleh empat hal: (i) pembilang
  periode September vs penyebut penduduk tahunan; (ii) pembulatan 2 desimal ribu
  jiwa; (iii) angka resmi memakai penimbang Susenas; (iv) revisi angka historis.
- Kolom `P0_resmi_BPS` disediakan **kosong** untuk diisi angka publikasi, agar
  selisih dapat ditunjukkan, bukan diklaim.
- Teks siap pakai: "Target penelitian bukan angka resmi BPS, melainkan hasil
  hitung dari komponen yang dipublikasikan. Karena itu seluruh keluaran disebut
  *estimasi*, dan perbedaan terhadap angka resmi dilaporkan pada Tabel audit."

### C2. Peran KNN geospasial harus dijelaskan secara operasional

- Definisi: "KNN geospasial" pada penelitian ini = **pembangun fitur spasial**.
  Untuk setiap baris (wilayah i, tahun t):

  ```
  Lag(i,t) = (1/k) x SUM_{j in N_k(i)} P0(j,t)
  ```

  dengan `N_k(i)` = k wilayah dengan **jarak haversine** terkecil ke i.
- Jarak: haversine antar centroid (centroid dihitung pada EPSG:32750, jarak
  dihitung pada lintang-bujur; radius bumi 6.371,0088 km). Terdekat:
  PPU–Balikpapan 31,3 km; terjauh: Paser–Berau 429,3 km.
- Nilai k: kandidat 1–5 diuji pada setiap lipatan; untuk tabel fitur spasial
  dipakai **k = 2** (modus). Matriks bobot yang dipakai adalah bobot seragam
  1/k (bukan bobot jarak).
- Aturan anti-kebocoran: tetangga **hanya** dari wilayah data latih; wilayah i
  tidak pernah menjadi tetangga dirinya sendiri (N_k(Samarinda) tidak memuat
  Samarinda); nilai P0 yang dirata-ratakan juga dari data latih.
- Contoh terverifikasi (Samarinda 2024): N_2 = Bontang (73,01 km; P0 3,7759) dan
  Balikpapan (80,47 km; P0 2,0259) -> lag = 2,9009. P0 Samarinda sendiri
  (4,4051) tidak masuk perhitungan.
- Kontribusi fitur ini pada model: satu kolom `SpatialLag` dengan koefisien
  **−0,3745**. Setelah galat baku robust berkelompok wilayah, p = **0,1963**,
  jadi **belum signifikan** - ini harus ditulis apa adanya.
- Tabel bukti lengkap: `outputs/knn_fitur_spasial.csv` (160 baris tetangga untuk
  10 wilayah x 8 tahun x k = 2).

### C3. Akurasi KNN 50 % pada 10 wilayah tidak boleh diklaim kuat

- Tindakan: KNN diuji sebagai **klasifikator tiga kelas** (Rendah/Menengah/Tinggi)
  dengan ambang kelas = kuantil 1/3 dan 2/3 P0 data latih tiap lipatan, dan
  dilaporkan bersama *confusion matrix*, metrik per kelas, serta baseline.

| Skema | Varian | Akurasi | Akurasi seimbang | Macro F1 | Baseline kelas mayoritas |
|---|---|---|---|---|---|
| LORO | KNN pada skor PCA | 0,6500 | 0,6167 | 0,6170 | 0,1250 |
| LORO | KNN geografis (kelas k tetangga) | 0,4500 | 0,4444 | 0,4423 | 0,1250 |
| LOOCV baris | KNN pada skor PCA | 0,7375 | 0,7369 | 0,7389 | 0,3375 |

Confusion matrix LORO (KNN pada skor PCA):

| Aktual / Prediksi | Rendah | Menengah | Tinggi |
|---|---|---|---|
| Rendah | 22 | 6 | 2 |
| Menengah | 0 | 7 | 13 |
| Tinggi | 4 | 3 | 23 |

Confusion matrix LOOCV baris (KNN pada skor PCA):

| Aktual / Prediksi | Rendah | Menengah | Tinggi |
|---|---|---|---|
| Rendah | 22 | 3 | 2 |
| Menengah | 2 | 18 | 6 |
| Tinggi | 1 | 7 | 19 |

Metrik per kelas (LORO): Rendah 0,846 / 0,733 / 0,786; Menengah 0,438 / 0,350 /
0,389; Tinggi 0,605 / 0,767 / 0,677 (presisi / recall / F1).

- Teks siap pakai: "Akurasi KNN geografis sebesar 45,00 % hanya sedikit di bawah
  angka pada versi awal (50 %). Rendahnya akurasi ini bukan semata masalah
  implementasi, melainkan menunjukkan bahwa kelas kemiskinan wilayah bertetangga
  tidak cukup mirip. Ketika KNN dijalankan pada skor PCA (ruang indikator
  sosial-ekonomi), akurasi naik menjadi 65,00 % (LORO) dan 73,75 % (LOOCV
  baris), tetap dengan catatan bahwa jumlah wilayah hanya 10 sehingga satu
  kesalahan setara delapan observasi. Karena itu KNN dilaporkan sebagai
  pembanding, bukan sebagai model utama, dan estimasi nilai P0 tetap memakai
  regresi."
- Bukti: `outputs/knn_klasifikasi_metrik.csv`, `outputs/knn_metrik_per_kelas.csv`,
  `outputs/knn_confusion_loro_pca.csv`, `outputs/knn_confusion_loro_spasial.csv`,
  `outputs/knn_confusion_loocv_baris_pca.csv`, dan gambar PNG yang sesuai.

### C4. PCA tidak boleh ditafsirkan kausal

- Tindakan: seluruh tafsiran PCA dinyatakan **eksploratif**. Loading = kontribusi
  variansi indikator pada komponen, **bukan** hubungan sebab-akibat.
- Angka: m = 5 komponen (kumulatif 89,35 %); PC1 memuat RLS (+0,97),
  kepadatan (+0,92), pengeluaran disesuaikan (+0,91); PC2 IDG (+0,77); PC3 TPAK
  (+0,70); PC4 laju pertumbuhan (+0,79); PC5 rasio jenis kelamin (+0,57).
- Teks siap pakai: "PCA dipakai untuk mereduksi 12 indikator yang berkorelasi
  tinggi (VIF RLS mencapai 36,77) menjadi 5 komponen. Loading menunjukkan
  kontribusi indikator terhadap variansi, bukan pengaruh sebab-akibat. Penamaan
  komponen (mis. 'poros kesejahteraan') adalah label deskriptif sementara."
- Bukti: `outputs/pca_loadings_model_akhir.csv`,
  `outputs/pca_interpretasi_komponen.csv`, `outputs/pca_varians_model_akhir.csv`.

### C5. Kelengkapan hasil: uji asumsi, stabilitas, dan pembanding

- Sudah dilaporkan: uji asumsi klasik, galat baku robust berkelompok wilayah,
  VIF sebelum/sesudah PCA, analisis sensitivitas, metrik per tahun, dan
  perbandingan empat varian model + dua varian KNN.
- Temuan yang **tidak boleh disembunyikan**:
  1. pada LORO, model utama (RMSE 1,8929) **kalah tipis** dari PCA + MLR tanpa
     lag (1,7759);
  2. koefisien spatial lag negatif dan tidak signifikan (p robust = 0,1963);
  3. residual tidak normal (p = 0,0373) dan berautokorelasi (DW = 0,6642);
  4. membuang `Jumlah_Penduduk_Ribu` justru memperbaiki hasil (RMSE 1,8513;
     R² 0,4495);
  5. KNN regresi juga kalah dari model utama (RMSE 2,1768 dan 2,4801).
- Bukti: `outputs/perbandingan_model.csv`, `outputs/sensitivitas_tanpa_penduduk.csv`,
  `outputs/uji_asumsi.csv`, `outputs/metrik_per_tahun.csv`.

### C6. Klaim kebijakan dan penamaan keluaran

- Tindakan: seluruh keluaran model disebut **estimasi**; P0 dihitung sendiri dan
  **bukan** angka resmi publikasi BPS; tidak ada klaim kausal; dashboard
  memuat peringatan serupa.
- Teks siap pakai: "Nilai yang dilaporkan adalah estimasi statistik dengan galat
  absolut rata-rata sekitar 1,5 poin persentase, sehingga tidak dapat menggantikan
  survei resmi maupun dipakai sebagai dasar tunggal penetapan sasaran program."

---

## 4. Bahan siap pakai untuk naskah

### 4.1 Opsi judul (pilih satu, lalu selaraskan abstrak dan metode)

1. **Estimasi dan Pemetaan Tingkat Kemiskinan Kabupaten/Kota di Provinsi
   Kalimantan Timur, 2017–2024: Pendekatan PCA, Spatial Lag, dan Regresi Linear
   Multivariabel** - dipakai bila KNN hanya muncul di pembahasan.
2. **Estimasi dan Pemetaan Tingkat Kemiskinan Kabupaten/Kota di Provinsi
   Kalimantan Timur, 2017–2024: Perbandingan K-Nearest Neighbors dan Regresi
   Linear Multivariabel Berbasis PCA** - **direkomendasikan** bila KNN tetap di
   judul, karena kata "perbandingan" sesuai dengan isi.
3. **Pemanfaatan K-Nearest Neighbors Geospasial dan PCA untuk Estimasi
   Kemiskinan di Kalimantan Timur Berbasis Dashboard Interaktif** - hanya bila
   dashboard dijadikan kontribusi utama.

Hindari judul yang menjanjikan "prediksi" atau "penentuan kebijakan": model ini
menghasilkan estimasi dengan galat yang masih cukup besar.

### 4.2 Draf abstrak (sekitar 200 kata)

"Penelitian ini mengestimasi dan memetakan tingkat kemiskinan (P0) pada sepuluh
kabupaten/kota Provinsi Kalimantan Timur periode 2017–2024. Data disusun sebagai
panel 80 observasi dari 90 tabel publikasi BPS, dengan target P0 dihitung dari
rasio jumlah penduduk miskin terhadap jumlah penduduk dan diaudit
ketertelusurannya. Dua belas indikator sosial-ekonomi direduksi dengan principal
component analysis (lima komponen, 89,35 % variansi kumulatif) setelah
standardisasi, lalu dilengkapi fitur spatial lag berupa rata-rata P0 dua wilayah
tetangga terdekat (jarak haversine) dan dimodelkan dengan regresi linear
multivariabel. Validasi leave-one-region-out menghasilkan R² 0,4245, MAE 1,5220,
dan RMSE 1,8929 (MAPE 27,09 %), lebih baik daripada baseline rata-rata (RMSE
2,7541) dan regresi tanpa PCA (3,4477), tetapi sedikit lebih rendah daripada
varian tanpa spatial lag (1,7759). Pada pembagian waktu 2023–2024 model utama
menjadi yang terbaik (RMSE 1,3834). K-nearest neighbors diuji sebagai pembanding:
sebagai regresi, RMSE-nya 2,1768; sebagai klasifikasi tiga kelas, akurasinya
65,00 % (leave-one-region-out). Koefisien spatial lag negatif dan tidak
signifikan setelah galat baku robust (p = 0,1963), sehingga bukti penularan
spasial belum kuat. Hasil disajikan dalam dashboard tujuh tab yang diuji otomatis
(21 tabel, 12 slider, tanpa exception)."

### 4.3 Teks metodologi: target P0

"Target penelitian adalah persentase penduduk miskin (P0) yang dihitung dengan
rumus P0 = (Jumlah_Penduduk_Miskin / Jumlah_Penduduk) x 100. Pembilang diambil
dari tabel *Jumlah Penduduk Miskin menurut Kabupaten/Kota* (satuan ribu jiwa,
hasil Susenas), penyebut dari tabel *Penduduk, Laju Pertumbuhan Penduduk,
Distribusi Persentase Penduduk, Kepadatan Penduduk, Rasio Jenis Kelamin* (satuan
ribu jiwa, proyeksi/SP2020). Nilai dibulatkan empat desimal. Karena dibentuk dari
komponen publikasi, nilai ini **bukan** angka resmi BPS; perbedaan yang mungkin
muncul dijelaskan oleh perbedaan periode pencacahan (September), pembulatan,
dan penggunaan penimbang Susenas pada angka resmi. Audit pembilang, penyebut,
dan selisihnya disajikan pada lampiran."

### 4.4 Teks metodologi: spatial lag

"Fitur spasial dihitung dari centroid kabupaten/kota pada sistem koordinat
terproyeksi EPSG:32750 (UTM zona 50S) yang dikembalikan ke EPSG:4326. Jarak
antar wilayah dihitung dengan rumus haversine (R = 6.371,0088 km). Untuk setiap
wilayah i dan tahun t, spatial lag didefinisikan sebagai
Lag(i,t) = (1/k) x jumlah P0(j,t) untuk j anggota N_k(i), dengan N_k(i) adalah k
wilayah berjarak terdekat dan bobot seragam 1/k. Nilai k dipilih dari kandidat
{1,2,3,4,5} melalui validasi leave-one-region-out di dalam data latih; pada model
akhir terpilih k = 2. Untuk mencegah kebocoran informasi, N_k(i) hanya memuat
wilayah data latih dan tidak pernah memuat i sendiri, serta P0 yang dirata-ratakan
hanya berasal dari data latih. Pada skema pembagian waktu, lag baris uji 2023–2024
memakai P0 tetangga tahun 2022."

### 4.5 Teks metodologi: PCA eksploratif

"Sebelum PCA, dua belas indikator distandardisasi (z-score) memakai parameter
data latih. Analisis korelasi menunjukkan tujuh pasang indikator dengan |r| >= 0,80
dan VIF RLS mencapai 36,77, sehingga reduksi dimensi diperlukan. Jumlah komponen
m ditentukan dari ambang variansi kumulatif 85 % pada data latih setiap lipatan;
pada model akhir diperoleh m = 5 dengan variansi kumulatif 89,35 %. Loading
komponen ditafsirkan secara **eksploratif** sebagai kontribusi indikator terhadap
variansi, bukan sebagai hubungan sebab-akibat."

### 4.6 Teks metodologi: KNN sebagai pembanding

"K-nearest neighbors dipakai dalam dua peran. Pertama, sebagai pembangun fitur
spasial (bagian 4.4). Kedua, sebagai pembanding yang diuji pada skema validasi
yang sama: (i) KNN regresi pada skor PCA dengan jarak Euclidean, bobot seragam,
dan k dipilih dari {1,...,5}; (ii) KNN geospasial murni yang memprediksi P0
sebagai rata-rata P0 k tetangga terdekat. Sebagai klasifikator, P0 dikelompokkan
menjadi tiga kelas (Rendah, Menengah, Tinggi) memakai kuantil 1/3 dan 2/3 data
latih setiap lipatan, lalu dievaluasi dengan confusion matrix, akurasi, akurasi
seimbang, macro F1, dan presisi/recall/F1 per kelas. Baseline kelas mayoritas
dihitung pada skema yang sama agar akurasi tidak ditafsirkan tanpa pembanding."

### 4.7 Teks metodologi: skema validasi

"Validasi utama adalah leave-one-region-out (LORO): setiap lipatan memakai satu
wilayah (8 observasi) sebagai data uji dan sembilan wilayah lain (72 observasi)
sebagai data latih, sehingga seluruh 80 observasi memperoleh estimasi out-of-fold.
Skema kedua adalah pembagian waktu: latih 2017–2022 (60 observasi), uji 2023–2024
(20 observasi). Skema ini dipakai untuk menguji kemampuan ekstrapolasi, bukan
untuk menggantikan LORO. Validasi leave-one-out per observasi hanya dilaporkan
pada klasifikasi sebagai pembanding, karena pada regresi berspasi label tahun
lain di wilayah yang sama masih terbaca model."

---

## 5. Tabel hasil (siap disalin ke naskah)

### 5.1 Perbandingan model, validasi leave-one-region-out (80 observasi)

| Model | R² | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|
| PCA + MLR tanpa spatial lag | **0,4934** | **1,4975** | **1,7759** | 29,38 |
| **PCA + spatial lag + MLR (model utama)** | 0,4245 | 1,5220 | 1,8929 | **27,09** |
| KNN regresi pada skor PCA | 0,2389 | 1,8797 | 2,1768 | 36,14 |
| KNN geospasial (rata-rata P0 tetangga) | 0,0121 | 2,1950 | 2,4801 | 38,82 |
| Baseline (rata-rata P0 data latih) | −0,2183 | 2,4242 | 2,7541 | 49,01 |
| MLR tanpa PCA (12 fitur + lag) | −0,9092 | 2,7158 | 3,4477 | 53,79 |

Sumber: `outputs/perbandingan_model.csv`.

### 5.2 Pembagian waktu (latih 2017–2022, uji 2023–2024; 20 observasi)

| Model | R² | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|
| **PCA + spatial lag + MLR (model utama)** | **0,6427** | **1,0044** | **1,3834** | **20,24** |
| MLR tanpa PCA (12 fitur + lag) | 0,5906 | 1,2673 | 1,4809 | 26,18 |
| KNN regresi pada skor PCA | 0,5586 | 1,2150 | 1,5377 | 25,19 |
| PCA + MLR tanpa spatial lag | 0,5066 | 1,3706 | 1,6257 | 28,86 |
| Baseline (rata-rata P0 data latih) | −0,1020 | 2,1379 | 2,4297 | 53,51 |
| KNN geospasial (rata-rata P0 tetangga) | −0,1349 | 2,0555 | 2,4658 | 50,44 |

### 5.3 Koefisien model akhir (data penuh, 80 observasi)

| Koefisien | Nilai | Galat baku | t | p | Galat baku robust | p robust |
|---|---|---|---|---|---|---|
| Intersep | 9,009872 | 0,814263 | 11,0651 | 0,0000 | 1,6587 | 0,0000 |
| PC1 | −0,938101 | 0,070148 | −13,3731 | 0,0000 | 0,1641 | 0,0000 |
| PC2 | 0,199484 | 0,119771 | 1,6655 | 0,1001 | 0,2753 | 0,4687 |
| PC3 | 0,298471 | 0,141629 | 2,1074 | 0,0385 | 0,2918 | 0,3064 |
| PC4 | 0,081772 | 0,145560 | 0,5618 | 0,5760 | 0,2255 | 0,7169 |
| PC5 | 0,249577 | 0,180682 | 1,3813 | 0,1714 | 0,5748 | 0,6642 |
| Spatial lag | −0,374460 | 0,134851 | −2,7768 | 0,0070 | 0,2898 | **0,1963** |

Catatan: tanpa galat baku robust, PC1, PC3, dan spatial lag tampak signifikan; setelah
galat baku dikelompokkan per wilayah, hanya PC1 yang tetap signifikan. Sumber:
`outputs/koefisien_model_akhir.csv`, `outputs/galat_baku_robust.csv`.

### 5.4 Uji asumsi klasik

| Uji | Statistik | p | Kesimpulan |
|---|---|---|---|
| Shapiro-Wilk (normalitas residual) | 0,9671 | 0,0373 | residual tidak normal pada alpha 5 % |
| Breusch-Pagan (heteroskedastisitas) | 4,2046 | 0,6490 | ragam galat homogen |
| Durbin-Watson (autokorelasi) | 0,6642 | – | autokorelasi positif |
| Uji F keseluruhan | 36,0232 | 0,0000 | model signifikan |

### 5.5 Metrik pooled per tahun (model utama, LORO)

| Tahun | R² | MAE | RMSE |
|---|---|---|---|
| 2017 | 0,4192 | 1,5154 | 1,9216 |
| 2018 | 0,4809 | 1,4518 | 1,9905 |
| 2019 | 0,3356 | 1,8948 | 2,2830 |
| 2020 | 0,5711 | 1,2939 | 1,5438 |
| 2021 | 0,5185 | 1,4097 | 1,6054 |
| 2022 | 0,3985 | 1,6062 | 1,7961 |
| 2023 | 0,4322 | 1,4879 | 1,7248 |
| 2024 | 0,1334 | 1,5166 | 2,1536 |

Tahun 2019 dan 2024 paling sulit diestimasi, konsisten dengan catatan mutu data
(definisi laju pertumbuhan 2019 dan pencilan PPU 2024).

### 5.6 Multikolinearitas sebelum dan sesudah PCA

| Matriks | VIF tertinggi | Keterangan |
|---|---|---|
| 12 fitur mentah | RLS **36,768** | multikolinearitas tinggi; 7 pasang fitur dengan |r| >= 0,80 |
| Komponen PCA + spatial lag | SpatialLag 1,674; PC1 1,406; sisanya 1,00–1,19 | multikolinearitas teratasi |

### 5.7 Analisis sensitivitas fitur penyebut

| Varian | Jumlah fitur | R² | MAE | RMSE | Selisih RMSE |
|---|---|---|---|---|---|
| Model utama | 12 | 0,4245 | 1,5220 | 1,8929 | – |
| Tanpa `Jumlah_Penduduk_Ribu` | 11 | 0,4495 | 1,4518 | **1,8513** | −0,0416 |

Tafsiran: indikator jumlah penduduk (yang menjadi penyebut rumus P0) lemah
sebagai prediktor; hasil tanpa fitur itu justru sedikit lebih baik.

### 5.8 Audit target P0 (cuplikan)

| Wilayah | Tahun | Pembilang (ribu jiwa) | Penyebut (ribu jiwa) | P0 hitung | P0 panel | Selisih |
|---|---|---|---|---|---|---|
| Paser | 2024 | 25,28 | 287,0 | 8,8084 | 8,8084 | 0,000038 |
| Samarinda | 2021 | 42,84 | 831,0 | 5,1552 | 5,1552 | 0,000046 |
| Penajam Paser Utara | 2024 | 10,78 | 267,7 | 4,0269 | 4,0269 | 0,000022 |

Selisih maksimum seluruh 90 sel: **0,00005 poin** (murni pembulatan).

---

## 6. Draf paragraf pembahasan dan keterbatasan

### 6.1 Mengapa model utama tidak selalu terbaik

"Pada validasi leave-one-region-out, model utama menghasilkan RMSE 1,8929,
sedikit lebih tinggi daripada varian tanpa spatial lag (1,7759). Hal ini wajar
karena hanya tersedia sepuluh wilayah: rata-rata P0 dua tetangga terdekat
mencampur wilayah pesisir berpenduduk padat dengan wilayah pedalaman, sehingga
fitur itu menambah derau. Sebaliknya, pada pembagian waktu 2023–2024 model utama
menjadi yang terbaik (RMSE 1,3834), menunjukkan bahwa informasi tetangga lebih
berguna ketika model harus memperkirakan tahun yang belum pernah dilihat."

### 6.2 Tafsiran koefisien spatial lag

"Koefisien spatial lag bernilai −0,3745 dan tidak signifikan setelah galat baku
robust berkelompok wilayah (p = 0,1963). Dengan demikian data panel sepuluh
wilayah ini **belum cukup** untuk menyimpulkan adanya penularan kemiskinan
antar-wilayah. Arah negatif juga tidak dapat ditafsirkan sebagai efek kausal:
wilayah dengan tetangga yang lebih sejahtera (P0 tetangga rendah) cenderung
memiliki P0 sendiri yang lebih tinggi pada beberapa tahun, dan sebaliknya."

### 6.3 Keterbatasan (wajib dituliskan)

1. **Jumlah wilayah kecil (10).** Setiap lipatan LORO hanya menyediakan sembilan
   wilayah latih; kesimpulan inferensial bersifat konservatif.
2. **Observasi panel tidak independen.** Durbin-Watson 0,6642 menunjukkan
   autokorelasi positif; karena itu galat baku robust berkelompok wilayah dipakai.
3. **Residual tidak normal** (Shapiro-Wilk p = 0,0373), sehingga uji t/F harus
   dibaca hati-hati.
4. **Data agregat kabupaten/kota**, bukan rumah tangga, sehingga tidak dapat
   menjelaskan kemiskinan tingkat individu.
5. **Definisi variabel berubah antar tahun** (laju pertumbuhan 2019 dengan basis
   2010–2019, perubahan basis penduduk SP2020, dan koreksi salah ketik penduduk
   Samarinda 2021).
6. **Tidak ada klaim kausal.** Koefisien hanya menggambarkan asosiasi pada
   2017–2024; simulasi dashboard menahan spatial lag tetap.
7. **Target bukan angka resmi BPS** dan tidak memuat variabel program
   pemerintah, sehingga tidak tepat dipakai untuk mengevaluasi intervensi.

### 6.4 Draf kesimpulan

"PCA efektif mengatasi multikolinearitas (VIF 36,77 menjadi 1,00–1,67) dan
membantu menurunkan galat: RMSE leave-one-region-out turun dari 3,4477 (tanpa PCA)
menjadi 1,8929 (model utama) dan 1,7759 (tanpa spatial lag), dibandingkan
baseline 2,7541. Model utama unggul pada pembagian waktu 2023–2024 (RMSE 1,3834).
KNN yang diuji sebagai pembanding menghasilkan galat lebih besar (RMSE 2,1768 dan
2,4801) dan akurasi klasifikasi 45,00 % (geografis) hingga 65,00 % (skor PCA),
sehingga penelitian ini tidak mengklaim KNN sebagai model terbaik. Bukti statistik
untuk spatial lag masih lemah (koefisien −0,3745; p robust 0,1963). Karena itu
hasil penelitian ini sebaiknya dibaca sebagai kerangka estimasi dan pemetaan awal
yang transparan - lengkap dengan galat, uji asumsi, dan keterbatasan - bukan
sebagai penentu kebijakan."

---

## 7. Daftar suntingan teknis naskah

### 7.1 Ejaan dan istilah (contoh yang disebut reviewer)

| Salah | Benar |
|---|---|
| TINJUAN PUSTAKA | TINJAUAN PUSTAKA |
| Graphicel | Graphical |
| Perincipal Component Abalysis | Principal Component Analysis |
| principal component analysis (tegak) | *principal component analysis* (miring, istilah asing) |
| K-Nearest Neighbor | K-Nearest Neighbors (KNN) pada penyebutan pertama |
| GUI wireframe | dashboard terimplementasi (setelah screenshot dimasukkan) |
| P0 (tanpa penjelasan) | P0 atau persentase penduduk miskin, dijelaskan pada penyebutan pertama |

### 7.2 Konsistensi angka dan istilah

- Periode: tulis **2017–2024** di judul, abstrak, metode, hasil, dan kesimpulan
  (jangan menyisakan "2023" atau "2016–2024" tanpa penjelasan).
- Jumlah observasi: **80** (10 wilayah x 8 tahun) dan sebut sumber datanya.
- Jumlah komponen PCA: **5** (bukan 3) dan variansi kumulatif **89,35 %**
  (bukan 95,30 %), dengan ambang 85 %.
- Satuan: poin persentase (untuk galat) vs persen (untuk tingkat); desimal pakai
  koma.
- Singkatan ditulis lengkap pada penyebutan pertama: PCA, MLR, KNN, LORO, P0,
  MAE, RMSE, MAPE, VIF, BPS, Susenas.

### 7.3 Penomoran, daftar, dan identitas

- Perbarui daftar isi, daftar tabel, dan daftar gambar (Word: klik kanan ->
  *Update Field*).
- Pastikan setiap tabel dan gambar **dirujuk** di batang tubuh ("...disajikan
  pada Tabel 4.1").
- Cek penomoran bab/subbab tidak melompat, dan rujukan silang ("lihat Bab 3")
  benar.
- Identitas kelompok pada halaman judul, lembar pengesahan, dan daftar anggota
  harus sama (nama, NIM, kelas, dosen pembimbing); hapus penanda sementara.

### 7.4 Gambar

- Ganti seluruh gambar rancangan (wireframe, mockup) dengan **tangkapan layar
  7 tab dashboard** yang benar-benar dijalankan.
- Sertakan gambar hasil model: `eda_korelasi_heatmap.png`, `eda_tren_p0.png`,
  `pca_scree_plot.png`, `pca_loading_heatmap.png`,
  `plot_aktual_vs_estimasi_loro.png`, `plot_residual_model_akhir.png`,
  `perbandingan_model_rmse_loro.png`, dan confusion matrix KNN.
- Tulis sumber di bawah tabel/gambar: "Sumber: hasil pengolahan penulis
  (output `python -m src.evaluate`)."

---

## 8. Daftar pustaka yang disarankan

### 8.1 Rujukan metode (sudah terverifikasi, gunakan APA 7)

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
2. Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
3. Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (1996). Geographically weighted regression: a method for exploring spatial nonstationarity. *Geographical Analysis*, 28(4), 281–298.
4. Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
5. Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
6. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
7. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.
8. LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
9. Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.
10. Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B*, 36(2), 111–147.

### 8.2 Data

11. Badan Pusat Statistik Provinsi Kalimantan Timur. Tabel-tabel publikasi
    *Jumlah Penduduk Miskin*, *Pengeluaran per Kapita Disesuaikan*, *Rata-Rata
    Lama Sekolah*, *Umur Harapan Hidup*, *Harapan Lama Sekolah Perempuan*,
    *Indeks Pemberdayaan Gender*, *TPAK*, *TPT*, dan *Penduduk, Laju Pertumbuhan,
    Kepadatan, Rasio Jenis Kelamin* menurut kabupaten/kota, 2016–2024
    (https://kaltim.bps.go.id). Cantumkan tanggal akses sesuai pengambilan data.

### 8.3 Cara menambah rujukan terbaru (2019–2024)

Cari di Google Scholar atau Sinta dengan kata kunci:
"pemetaan kemiskinan", "KNN klasifikasi kemiskinan", "principal component
analysis data sosial ekonomi", "regresi data panel kabupaten", "spatial lag
kemiskinan Indonesia". Ambil 5–8 artikel yang **benar-benar dibaca**, catat
metode dan hasilnya, lalu tulis satu paragraf pembanding. Jangan menyalin
rujukan tanpa membacanya - reviewer menandai entri yang tidak dipakai.

---

## 9. Verifikasi: memastikan angka laporan sama dengan program

### 9.1 Perintah dan durasi (mesin pengembangan, Python 3.13, Windows)

| Perintah | Fungsi | Durasi |
|---|---|---|
| `python tools\bangun_dataset.py` | 90 CSV -> xlsx (90 baris x 16 kolom) | < 10 detik |
| `python -m src.eda` | validasi, audit P0, EDA, centroid, jarak | < 20 detik |
| `python -m src.evaluate` | LORO, pembagian waktu, KNN, uji asumsi, model akhir | **218,5 detik** |
| `python tools\uji_dashboard.py` | uji 7 tab dashboard | **15,9 detik** |

Hasil bersifat deterministik (`random_state = 42`, PCA dengan SVD penuh), sehingga
pengulangan menghasilkan angka yang sama.

### 9.2 Titik pemeriksaan (spot check)

| Yang diperiksa | Nilai yang harus muncul | Berkas/sumber |
|---|---|---|
| Bentuk panel | 90 baris x 16 kolom, 0 duplikat | xlsx `Data_Panel` |
| Baris periode model | **80** (10 wilayah x 8 tahun) | `outputs/kelengkapan_data.csv` |
| P0 Paser 2024 | **8,8084** | `outputs/audit_target_p0.csv` |
| P0 Samarinda 2021 | **5,1552** | `outputs/audit_target_p0.csv` |
| P0 PPU 2024 | **4,0269** | `outputs/audit_target_p0.csv` |
| Selisih audit maksimum | **0,00005** poin | `outputs/audit_target_p0.csv` |
| VIF RLS (fitur mentah) | **36,768** | `outputs/vif_fitur_mentah.csv` |
| m komponen & variansi | **5** komponen; **89,35 %** | `models/metadata.json` |
| k spatial lag | **2** | `models/metadata.json` |
| RMSE LORO model utama | **1,8929** (R² 0,4245) | `outputs/perbandingan_model.csv` |
| RMSE LORO tanpa lag | **1,7759** (R² 0,4934) | `outputs/perbandingan_model.csv` |
| RMSE waktu model utama | **1,3834** (R² 0,6427) | `outputs/perbandingan_model.csv` |
| Koefisien spatial lag | **−0,3745**; p robust **0,1963** | `outputs/galat_baku_robust.csv` |
| Shapiro-Wilk | W **0,9671**; p **0,0373** | `outputs/uji_asumsi.csv` |
| Durbin-Watson | **0,6642** | `outputs/uji_asumsi.csv` |
| Akurasi KNN geografis (LORO) | **0,4500** | `outputs/knn_klasifikasi_metrik.csv` |
| Akurasi KNN PCA (LORO) | **0,6500** | `outputs/knn_klasifikasi_metrik.csv` |
| Dashboard | 7 tab, 21 tabel, 14 metrik, 12 slider, 0 exception | keluaran `uji_dashboard.py` |

### 9.3 Cara cepat memeriksa konsistensi naskah

1. Buka `outputs/perbandingan_model.csv` dan bandingkan dengan tabel di naskah
   (R², MAE, RMSE, MAPE).
2. Buka `outputs/koefisien_model_akhir.csv` dan `outputs/galat_baku_robust.csv`
   untuk tabel koefisien.
3. Buka `outputs/knn_klasifikasi_metrik.csv` dan
   `outputs/knn_metrik_per_kelas.csv` untuk bagian KNN.
4. Pastikan setiap angka di naskah memiliki berkas sumber di `outputs/`.

---

## 10. Tindakan yang masih harus dilakukan di luar repositori

1. **Isi kolom `P0_resmi_BPS`** pada `outputs/audit_target_p0.csv` dengan angka
   publikasi BPS (lalu hitung `Selisih_vs_P0_resmi`). Ini menutup catatan C1
   secara tuntas.
2. **Ambil tangkapan layar 7 tab dashboard** dan ganti seluruh gambar wireframe
   pada naskah.
3. **Tambahkan 5–8 referensi 2019–2024** yang benar-benar dibaca, lalu hapus
   entri yang tidak dipakai (termasuk "Sepeda (Bike Sharing)").
4. **Sunting naskah** memakai daftar pada bagian 7 (ejaan, penomoran, identitas,
   daftar isi/gambar/tabel).
5. **Selaraskan abstrak** dengan draf pada bagian 4.2, termasuk angka R², RMSE,
   dan akurasi KNN yang baru.
6. **Cek format dan similaritas** sesuai ketentuan kampus (margin, font, sitasi
   APA 7, batas similaritas).
7. **Lampirkan kode atau arsip repositori** (tanpa folder `venv/`) beserta
   `README.md`, `outputs/`, dan `models/` agar hasil dapat direproduksi penguji.
8. **Diskusikan dengan dosen pembimbing** bila keputusan yang diambil berbeda
   (mis. KNN dihapus dari judul).

---

## 11. Ringkasan satu halaman untuk lembar tanggapan

| No | Catatan reviewer | Jawaban singkat | Bukti |
|---|---|---|---|
| 1 | Judul/abstrak tidak sesuai isi | Judul diarahkan pada estimasi + perbandingan; abstrak memuat angka baru | bagian 4.1–4.2 |
| 2 | Hasil MLR tidak ada | R² 0,4245; RMSE 1,8929; koefisien + uji asumsi + robust SE | §5.1–5.4 |
| 3 | Peran KNN kabur | KNN = pembangun fitur lag + pembanding; parameter dan contoh hitung dijelaskan | §C2 |
| 4 | Akurasi KNN 50 % diklaim kuat | Akurasi 45,00 % (geografis) / 65,00 % (PCA) + confusion matrix + baseline | §C3 |
| 5 | GUI masih wireframe | Dashboard 7 tab berjalan, diuji AppTest tanpa exception | §B1 |
| 6 | Target P0 tidak jelas | Rumus, sumber pembilang/penyebut, audit selisih, disclaimer | §C1, §5.8 |
| 7 | Periode tidak konsisten | 2017–2024, 80 observasi, 2016 dijelaskan | §A2 |
| 8 | PCA ditafsirkan kausal | Dinyatakan eksploratif; loading = kontribusi variansi | §C4 |
| 9 | Pustaka minim/tidak relevan | 11 rujukan metode + data; entri tidak relevan dihapus | bagian 8 |
| 10 | Sistematika dan salah ketik | Daftar perbaikan teknis lengkap | bagian 7 |

---

*Dokumen ini disusun berdasarkan eksekusi nyata pada repositori `pbl_estimasi/`
(Python 3.13, Windows). Seluruh angka dapat diperiksa ulang dengan menjalankan
empat perintah pada bagian 9.1. Bila parameter diubah (periode, jumlah fitur,
ambang PCA, kandidat k), jalankan ulang seluruh perintah sebelum menyalin angka
ke naskah.*