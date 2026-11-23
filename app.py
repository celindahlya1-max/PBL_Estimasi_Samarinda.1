# -*- coding: utf-8 -*-
"""Dashboard Streamlit: Estimasi dan Pemetaan Tingkat Kemiskinan Kabupaten/Kota
di Provinsi Kalimantan Timur Tahun 2017-2024.

Menjalankan:  streamlit run app.py

Alur metodologi yang ditampilkan: standardisasi -> PCA (reduksi dimensi) ->
KNN geospasial (fitur spatial lag: rata-rata P0 k tetangga) -> Regresi
Linear Multivariabel, ditambah pembanding KNN dan klasifikasi 3 kelas.

Catatan istilah: seluruh nilai hasil model disebut "estimasi". P0 adalah hasil
hitung (jumlah penduduk miskin / jumlah penduduk x 100), bukan angka resmi BPS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from shapely.geometry import shape
from streamlit_folium import st_folium

# Agar paket "src" selalu dapat diimpor, baik dijalankan lewat `streamlit run`
# maupun lewat pengujian (AppTest) dengan folder kerja berbeda.
AKAR_PROYEK = Path(__file__).resolve().parent
if str(AKAR_PROYEK) not in sys.path:
    sys.path.insert(0, str(AKAR_PROYEK))

from src import config as cfg  # noqa: E402
from src import data as mod_data  # noqa: E402
from src import geo as mod_geo  # noqa: E402
from src import model as mod_model  # noqa: E402

st.set_page_config(page_title="Estimasi Kemiskinan Kaltim 2017-2024",
                   page_icon="🗺️", layout="wide")

WARNA_KELAS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
KELAS_LABEL = ["Sangat rendah", "Rendah", "Sedang", "Tinggi", "Sangat tinggi"]


# ---------------------------------------------------------------------------
# Pemuat data (di-cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def muat_panel() -> pd.DataFrame:
    """Muat panel data lengkap 2016-2024."""
    return mod_data.muat_panel()


@st.cache_data(show_spinner=False)
def muat_data_model() -> pd.DataFrame:
    """Muat 80 baris periode pemodelan 2017-2024."""
    return mod_data.siapkan_data_model(muat_panel())


@st.cache_data(show_spinner=False)
def muat_csv(nama: str) -> pd.DataFrame | None:
    """Muat satu berkas CSV dari folder outputs/; None bila belum ada."""
    path = cfg.FOLDER_OUTPUT / nama
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def muat_geojson_fitur() -> list[dict]:
    """Fitur GeoJSON 10 kabupaten/kota dengan nama wilayah baku."""
    geometri = mod_geo.muat_geometri()
    fitur = []
    for _, baris in geometri.iterrows():
        fitur.append({
            "type": "Feature",
            "properties": {"Wilayah": baris["Wilayah"],
                           "Nama": cfg.NAMA_TAMPILAN.get(baris["Wilayah"],
                                                          baris["Wilayah"])},
            "geometry": shape(baris["geometry"]).__geo_interface__,
        })
    return fitur


@st.cache_resource(show_spinner=False)
def muat_model():
    """Muat model akhir (joblib) dari folder models/."""
    return mod_model.muat_model(cfg.FOLDER_MODEL / "model_akhir.joblib")


@st.cache_data(show_spinner=False)
def muat_metadata() -> dict:
    """Muat metadata.json model akhir."""
    path = cfg.FOLDER_MODEL / "metadata.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def periksa_berkas() -> list[str]:
    """Periksa keberadaan berkas yang dibutuhkan dashboard."""
    kurang = [str(p) for p in (cfg.BERKAS_PANEL, cfg.BERKAS_GEOJSON)
              if not p.exists()]
    if not (cfg.FOLDER_OUTPUT / "perbandingan_model.csv").exists():
        kurang.append(str(cfg.FOLDER_OUTPUT / "perbandingan_model.csv"))
    return kurang


def angka(x, nd: int = 2, satuan: str = "") -> str:
    """Format angka gaya Indonesia (koma desimal)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    teks = f"{x:,.{nd}f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return f"{teks} {satuan}".strip()

# ---------------------------------------------------------------------------
# Peta choropleth
# ---------------------------------------------------------------------------
def batas_kelas(nilai: list[float], jumlah: int = 5) -> list[float]:
    """Batas kelas warna berbasis kuantil (aman untuk data kecil)."""
    bersih = np.array([v for v in nilai if v is not None and not np.isnan(v)])
    if len(bersih) < jumlah:
        return []
    return list(np.quantile(bersih, np.linspace(0, 1, jumlah + 1)).round(4))


def kelas_nilai(nilai: float, batas: list[float]) -> int:
    """Indeks kelas (0-4) sebuah nilai terhadap batas kuantil."""
    if nilai is None or np.isnan(nilai) or not batas:
        return 0
    for i in range(len(batas) - 1):
        if batas[i] <= nilai <= batas[i + 1]:
            return i
    return len(batas) - 2 if nilai < batas[-1] else len(batas) - 2


def buat_peta(nilai: dict[str, float], aktual: dict[str, float], judul: str,
              keterangan: str) -> folium.Map:
    """Peta choropleth Folium dengan tooltip nama wilayah dan nilai."""
    fitur = muat_geojson_fitur()
    batas = batas_kelas(list(nilai.values()))
    for f in fitur:
        w = f["properties"]["Wilayah"]
        v = nilai.get(w)
        f["properties"]["nilai"] = None if v is None else round(float(v), 4)
        f["properties"]["aktual"] = (None if aktual.get(w) is None
                                     else round(float(aktual[w]), 4))
        f["properties"]["selisih"] = (None if (v is None or aktual.get(w) is None)
                                      else round(float(v) - float(aktual[w]), 4))
    peta = folium.Map(location=[0.4, 116.6], zoom_start=6,
                      tiles="OpenStreetMap", control_scale=True)

    def gaya(f):
        w = f["properties"]["Wilayah"]
        v = nilai.get(w)
        idx = kelas_nilai(v, batas)
        return {"fillColor": WARNA_KELAS[idx], "color": "#404040",
                "weight": 0.8, "fillOpacity": 0.75}

    lapisan = folium.GeoJson(
        {"type": "FeatureCollection", "features": fitur},
        style_function=gaya, name=judul,
        tooltip=folium.GeoJsonTooltip(
            fields=["Nama", "nilai", "aktual", "selisih"],
            aliases=["Wilayah:", "Nilai ditampilkan (estimasi/aktual):",
                     "P0 aktual (hasil hitung):", "Selisih (nilai - aktual):"],
            localize=True, sticky=False))
    lapisan.add_to(peta)

    legenda = ["<div style='position: fixed; bottom: 30px; left: 30px; z-index:9999;"
               "background: white; padding: 8px 10px; border: 1px solid #999;"
               "font-size: 12px; line-height: 1.5;'>",
               f"<b>{keterangan}</b>"]
    for i, label in enumerate(KELAS_LABEL):
        if batas:
            bagian = (f"{angka(batas[i])} - {angka(batas[i + 1])}"
                      if i < len(batas) - 1 else "")
        else:
            bagian = ""
        legenda.append(
            f"<div><span style='display:inline-block;width:12px;height:12px;"
            f"background:{WARNA_KELAS[i]};border:1px solid #666;'></span> "
            f"{label}: {bagian}</div>")
    legenda.append("</div>")
    peta.get_root().html.add_child(folium.Element("".join(legenda)))
    return peta


def tab_peta(data: pd.DataFrame) -> None:
    """Tab 1 - peta choropleth P0 per tahun (aktual atau estimasi)."""
    st.subheader("Peta Tingkat Kemiskinan (P0) per Kabupaten/Kota")
    kolom1, kolom2, kolom3 = st.columns([1, 1.4, 1.6])
    tahun = kolom1.selectbox("Tahun", cfg.TAHUN_MODEL, index=len(cfg.TAHUN_MODEL) - 1,
                             key="peta_tahun")
    sumber = kolom2.radio(
        "Sumber nilai",
        ["Aktual (hasil hitung)", "Estimasi model (uji LORO, out-of-fold)",
         "Estimasi model (dalam sampel)"],
        key="peta_sumber")

    baris = data[data[cfg.KOLOM_TAHUN] == tahun]
    aktual = dict(zip(baris[cfg.KOLOM_WILAYAH], baris[cfg.TARGET]))
    if sumber.startswith("Aktual"):
        nilai = dict(aktual)
        keterangan = f"P0 aktual hasil hitung, {tahun} (%)"
    else:
        berkas = ("estimasi_lipatan.csv" if "LORO" in sumber else "estimasi.csv")
        tabel = muat_csv(berkas)
        if tabel is None:
            st.error(f"Berkas outputs/{berkas} belum ada. Jalankan "
                     "`python -m src.evaluate` lebih dahulu.")
            return
        kolom_nilai = "Estimasi_utama" if "Estimasi_utama" in tabel.columns \
            else "P0_Estimasi"
        sub = tabel[tabel["Tahun"] == tahun]
        nilai = dict(zip(sub["Wilayah"], sub[kolom_nilai]))
        keterangan = (f"Estimasi model utama, {tahun} (%), "
                      f"{'validasi LORO' if 'LORO' in sumber else 'dalam sampel'}")

    kolom3.metric("Rata-rata P0 ditampilkan",
                  angka(float(np.nanmean(list(nilai.values()))), 3, "%"))
    st_folium(buat_peta(nilai, aktual, f"P0 {tahun}", keterangan),
              height=520, use_container_width=True, returned_objects=[])

    tabel = pd.DataFrame({
        "Wilayah": [cfg.NAMA_TAMPILAN.get(w, w) for w in cfg.WILAYAH],
        "P0 aktual (%)": [round(aktual.get(w, np.nan), 4) for w in cfg.WILAYAH],
        "Nilai ditampilkan (%)": [round(nilai.get(w, np.nan), 4) for w in cfg.WILAYAH],
    })
    tabel["Selisih"] = (tabel["Nilai ditampilkan (%)"] - tabel["P0 aktual (%)"]).round(4)
    tabel["Peringkat (nilai ditampilkan)"] = tabel["Nilai ditampilkan (%)"] \
        .rank(ascending=False, method="min").astype("Int64")
    st.dataframe(tabel, use_container_width=True, hide_index=True)
    with st.expander("Audit target P0 (rumus, pembilang, penyebut)"):
        audit = muat_csv("audit_target_p0.csv")
        if audit is None:
            st.info("Berkas outputs/audit_target_p0.csv belum ada. Jalankan "
                    "`python -m src.eda` lebih dahulu.")
        else:
            st.markdown("P0 dihitung sendiri dengan rumus **jumlah penduduk miskin "
                        "/ jumlah penduduk x 100**. Tabel ini menelusuri pembilang, "
                        "penyebut, hasil hitung, dan selisihnya terhadap nilai panel.")
            kolom_audit = ["Kabupaten_Kota", "Tahun",
                           "Pembilang_Jumlah_Penduduk_Miskin_Ribu",
                           "Penyebut_Jumlah_Penduduk_Ribu", "P0_hitung_ulang",
                           "P0_di_panel", "Selisih_poin", "Sumber_pembilang",
                           "Sumber_penyebut", "P0_resmi_BPS"]
            st.dataframe(audit[kolom_audit], use_container_width=True,
                         hide_index=True, height=300)
            st.caption("Kolom P0_resmi_BPS disediakan untuk diisi angka publikasi "
                       "BPS, sehingga selisih terhadap angka resmi dapat ditunjukkan. "
                       "Penyebab kemungkinan selisih dijelaskan pada kolom terakhir "
                       "berkas CSV.")

    st.caption("Peta memakai batas administrasi kabupaten/kota; poligon tanpa nama "
               "pada berkas GeoJSON dibuang. Nilai estimasi berasal dari model "
               "statistik, bukan angka resmi BPS.")



def tab_estimasi(data: pd.DataFrame) -> None:
    """Tab 2 - perbandingan estimasi vs aktual per wilayah dan tahun."""
    st.subheader("Estimasi vs Aktual per Wilayah dan Tahun")
    tabel = muat_csv("estimasi_lipatan.csv")
    if tabel is None:
        st.error("Berkas outputs/estimasi_lipatan.csv belum ada. Jalankan "
                 "`python -m src.evaluate` lebih dahulu.")
        return

    kolom_model = [k for k in tabel.columns if k.startswith("Estimasi_")]
    pilihan = st.multiselect(
        "Model yang ditampilkan", kolom_model,
        default=[k for k in kolom_model if k in
                 ("Estimasi_baseline", "Estimasi_pca_tanpa_lag",
                  "Estimasi_mlr_tanpa_pca", "Estimasi_utama")])
    wilayah = st.selectbox("Wilayah", ["Semua wilayah"] + list(cfg.WILAYAH))

    sub = tabel.copy()
    if wilayah != "Semua wilayah":
        sub = sub[sub["Wilayah"] == wilayah]
    sub = sub.sort_values(["Wilayah", "Tahun"])

    kiri, kanan = st.columns([1.15, 1])
    with kiri:
        panjang = sub.melt(id_vars=["Wilayah", "Tahun", "Aktual"],
                           value_vars=pilihan or ["Aktual"],
                           var_name="Model", value_name="Nilai")
        panjang["Model"] = panjang["Model"].str.replace("Estimasi_", "", regex=False)
        panjang = pd.concat([panjang, sub.assign(Model="aktual")[
            ["Wilayah", "Tahun", "Model", "Aktual"]].rename(
                columns={"Aktual": "Nilai"})], ignore_index=True)
        if wilayah == "Semua wilayah":
            rerata = (panjang.groupby(["Model", "Tahun"], as_index=False)["Nilai"].mean())
            fig = px.line(rerata, x="Tahun", y="Nilai", color="Model",
                          markers=True,
                          title="Rata-rata 10 wilayah: P0 aktual vs estimasi (%)")
        else:
            fig = px.line(panjang, x="Tahun", y="Nilai", color="Model",
                          markers=True,
                          title=f"{cfg.NAMA_TAMPILAN.get(wilayah, wilayah)}: "
                                "P0 aktual vs estimasi (%)")
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        if wilayah == "Semua wilayah":
            galat = sub.copy()
            for k in pilihan:
                galat[k] = (galat[k] - galat["Aktual"]).abs()
            ringkas = pd.DataFrame({
                "Model": [k.replace("Estimasi_", "") for k in pilihan],
                "MAE": [round(galat[k].mean(), 4) for k in pilihan],
                "Galat maksimum": [round(galat[k].max(), 4) for k in pilihan],
            })
            st.markdown("**Galat absolut (estimasi LORO)**")
            st.dataframe(ringkas, use_container_width=True, hide_index=True)
        st.markdown("**Tabel estimasi & galat**")
        tampil = sub.copy()
        kolom_tampil, kolom_galat = [], []
        for k in pilihan:
            nama_nilai = k.replace("Estimasi_", "Est_")
            nama_galat = k.replace("Estimasi_", "Selisih_")
            tampil[nama_nilai] = tampil[k].round(4)
            tampil[nama_galat] = (tampil[k] - tampil["Aktual"]).round(4)
            kolom_tampil.append(nama_nilai)
            kolom_galat.append(nama_galat)
        if not pilihan:
            st.info("Pilih minimal satu model untuk melihat tabel estimasi.")
        else:
            st.dataframe(tampil[["Wilayah", "Tahun", "Aktual"]
                                + kolom_tampil + kolom_galat],
                         use_container_width=True, hide_index=True, height=420)
    st.caption("Nilai estimasi pada tab ini berasal dari validasi leave-one-region-out: "
               "setiap wilayah pernah menjadi data uji, sehingga nilainya jujur "
               "(out-of-fold) dan tidak memakai label wilayah itu sendiri.")


def tab_pca(data: pd.DataFrame) -> None:
    """Tab 3 - hasil PCA: scree plot, variansi, loading, interpretasi."""
    st.subheader("Hasil Principal Component Analysis (PCA)")
    varians = muat_csv("pca_varians_model_akhir.csv")
    loadings = muat_csv("pca_loadings_model_akhir.csv")
    tafsir = muat_csv("pca_interpretasi_komponen.csv")
    metadata = muat_metadata()
    if varians is None or loadings is None:
        st.error("Berkas hasil PCA belum ada di outputs/. Jalankan "
                 "`python -m src.evaluate` lebih dahulu.")
        return

    m = metadata.get("m_komponen", "-")
    kumulatif = float(varians.loc[varians["Komponen"] == f"PC{m}",
                                  "Varians_kumulatif"].iloc[0]) if m != "-" else np.nan
    a, b, c = st.columns(3)
    a.metric("Jumlah komponen dipakai (m)", f"{m}")
    b.metric("Varians kumulatif m komponen", angka(kumulatif * 100, 2, "%"))
    c.metric("Ambang yang disyaratkan", angka(cfg.PCA_VAR_THRESHOLD * 100, 0, "%"))

    kiri, kanan = st.columns([1, 1])
    with kiri:
        gambar = cfg.FOLDER_OUTPUT / "pca_scree_plot.png"
        if gambar.exists():
            st.image(str(gambar), caption="Scree plot dan variansi kumulatif "
                     "(model akhir, 80 baris)")
    with kanan:
        gambar = cfg.FOLDER_OUTPUT / "pca_loading_heatmap.png"
        if gambar.exists():
            st.image(str(gambar), caption="Heatmap loading komponen utama")
    st.markdown("**Explained variance per komponen**")
    st.dataframe(varians, use_container_width=True, hide_index=True)
    st.markdown("**Matriks loading (m komponen yang dipakai)**")
    st.dataframe(loadings.set_index("Unnamed: 0").round(4) if "Unnamed: 0"
                 in loadings.columns else loadings.round(4),
                 use_container_width=True)
    if tafsir is not None:
        st.markdown("**Interpretasi singkat komponen**")
        st.dataframe(tafsir, use_container_width=True, hide_index=True)
    st.caption("Loading adalah korelasi antara indikator asli dan komponen utama. "
               "Nilai positif besar berarti indikator naik seiring komponen naik.")



def tab_model(data: pd.DataFrame) -> None:
    """Tab 4 - koefisien model, perbandingan model, dan plot residual."""
    st.subheader("Model dan Evaluasi")
    perbandingan = muat_csv("perbandingan_model.csv")
    koef = muat_csv("koefisien_model_akhir.csv")
    robust = muat_csv("galat_baku_robust.csv")
    asli = muat_csv("koefisien_variabel_asli.csv")
    uji = muat_csv("uji_asumsi.csv")
    if perbandingan is None:
        st.error("Berkas outputs/perbandingan_model.csv belum ada. Jalankan "
                 "`python -m src.evaluate` lebih dahulu.")
        return

    st.markdown("**Perbandingan model (metrik gabungan)**")
    st.dataframe(perbandingan, use_container_width=True, hide_index=True)
    kiri, kanan = st.columns([1, 1])
    with kiri:
        gambar = cfg.FOLDER_OUTPUT / "perbandingan_model_rmse_loro.png"
        if gambar.exists():
            st.image(str(gambar), caption="RMSE model pada validasi leave-one-region-out")
    with kanan:
        gambar = cfg.FOLDER_OUTPUT / "plot_residual_model_akhir.png"
        if gambar.exists():
            st.image(str(gambar), caption="Residual vs nilai cocok - model akhir "
                     "(data penuh 80 baris)")

    if koef is not None:
        st.markdown("**Koefisien Regresi Linear Multivariabel (model akhir)**")
        st.dataframe(koef, use_container_width=True, hide_index=True)
    if robust is not None:
        st.markdown("**Galat baku robust berkelompok per wilayah** "
                    "(observasi panel tidak sepenuhnya independen)")
        st.dataframe(robust, use_container_width=True)
    if asli is not None:
        st.markdown("**Koefisien dikembalikan ke variabel asli (loading x koefisien)**")
        st.dataframe(asli, use_container_width=True, hide_index=True)
    if uji is not None:
        st.markdown("**Uji asumsi klasik**")
        st.dataframe(uji, use_container_width=True, hide_index=True)

    metrik_wilayah = muat_csv("metrik_per_wilayah.csv")
    if metrik_wilayah is not None:
        st.markdown("**Metrik per wilayah (validasi LORO)**")
        st.dataframe(metrik_wilayah, use_container_width=True, hide_index=True)

    sensitivitas = muat_csv("sensitivitas_tanpa_penduduk.csv")
    if sensitivitas is not None:
        st.markdown("**Analisis sensitivitas fitur Jumlah_Penduduk_Ribu**")
        st.dataframe(sensitivitas, use_container_width=True, hide_index=True)
        st.caption("Jumlah_Penduduk_Ribu adalah penyebut rumus P0, sehingga "
                   "pengaruhnya perlu diuji terpisah dari model utama.")
    st.caption("R2 per wilayah bisa bernilai sangat negatif karena ragam P0 dalam "
               "satu wilayah kecil; angka utama yang dilaporkan adalah metrik "
               "gabungan 80 observasi.")


def tab_knn(data: pd.DataFrame) -> None:
    """Tab 7 - pembanding KNN, fitur spasial, dan klasifikasi kategori P0."""
    st.subheader("Pembanding KNN dan Klasifikasi Kategori Kemiskinan")
    regresi = muat_csv("knn_regresi_metrik.csv")
    klasifikasi = muat_csv("knn_klasifikasi_metrik.csv")
    per_kelas = muat_csv("knn_metrik_per_kelas.csv")
    fitur_spasial = muat_csv("knn_fitur_spasial.csv")
    detail = muat_csv("knn_klasifikasi_detail.csv")
    if regresi is None or klasifikasi is None:
        st.error("Berkas hasil KNN belum ada di outputs/. Jalankan "
                 "`python -m src.evaluate` lebih dahulu.")
        return

    kartu = st.columns(3)
    pasangan = [
        ("LORO", "knn_klas_pca", "Akurasi KNN pada skor PCA (LORO)"),
        ("LORO", "knn_klas_spasial", "Akurasi KNN geografis (LORO)"),
        ("LORO", "knn_klas_baseline", "Baseline kelas mayoritas (LORO)"),
    ]
    for kolom, (skema, varian, judul) in zip(kartu, pasangan):
        baris_kartu = klasifikasi[(klasifikasi["Skema"] == skema)
                                  & (klasifikasi["Varian"] == varian)]
        kolom.metric(judul,
                     angka(float(baris_kartu["Akurasi"].iloc[0]) * 100, 2, "%")
                     if len(baris_kartu) else "-")

    st.markdown("**Perbandingan model regresi (metrik out-of-fold)**")
    st.dataframe(regresi, use_container_width=True, hide_index=True)
    st.caption("KNN di sini adalah pembanding: estimasi P0 tetap memakai model "
               "PCA + spatial lag + MLR. RMSE KNN lebih besar daripada model "
               "utama, sehingga KNN tidak dipakai sebagai model akhir.")

    if fitur_spasial is not None:
        st.markdown("**Fitur spasial: tetangga terdekat dan nilai lag**")
        kolom1, kolom2 = st.columns([1, 1])
        wilayah = kolom1.selectbox("Wilayah", cfg.WILAYAH, key="knn_wilayah")
        daftar_tahun = sorted(int(t) for t in fitur_spasial["Tahun"].unique())
        tahun = kolom2.selectbox("Tahun", daftar_tahun,
                                 index=len(daftar_tahun) - 1, key="knn_tahun")
        sub = fitur_spasial[(fitur_spasial["Wilayah"] == wilayah)
                            & (fitur_spasial["Tahun"] == tahun)]
        st.dataframe(sub[["Peringkat_tetangga", "Wilayah_tetangga", "Jarak_km",
                          "P0_tetangga", "P0_lag", "P0_aktual", "Kelas_aktual",
                          "Kelas_lag"]], use_container_width=True, hide_index=True)
        st.caption("Lag = rata-rata P0 tetangga pada tahun yang sama. Nilai itulah "
                   "yang dipakai sebagai fitur spatial lag model utama dan "
                   "sekaligus menjadi estimasi varian KNN geospasial.")

    st.markdown("**Klasifikasi kategori kemiskinan (confusion matrix)**")
    opsi = [f"{r.Skema} · {r.Label}" for r in klasifikasi.itertuples()]
    pilihan = st.selectbox("Skema dan varian", opsi, key="knn_pilih")
    baris = klasifikasi.iloc[opsi.index(pilihan)]
    akhiran = str(baris["Varian"]).replace("knn_klas_", "")
    nama_csv = f"knn_confusion_{str(baris['Skema']).lower()}_{akhiran}.csv"
    conf = muat_csv(nama_csv)
    kiri, kanan = st.columns([1, 1])
    with kiri:
        if conf is not None:
            st.dataframe(conf.set_index(conf.columns[0]),
                         use_container_width=True)
        gambar = cfg.FOLDER_OUTPUT / nama_csv.replace(".csv", ".png")
        if gambar.exists():
            st.image(str(gambar), caption=f"Confusion matrix {baris['Label']}")
    with kanan:
        st.metric("Akurasi", angka(float(baris["Akurasi"]) * 100, 2, "%"))
        st.metric("Akurasi seimbang",
                  angka(float(baris["Akurasi_seimbang"]) * 100, 2, "%"))
        st.metric("Macro F1", angka(float(baris["Macro_F1"]), 4))
        st.metric("Baseline kelas mayoritas",
                  angka(float(baris["Akurasi_baseline_mayoritas"]) * 100, 2, "%"))
        if per_kelas is not None:
            sub_per = per_kelas[(per_kelas["Skema"] == baris["Skema"])
                                & (per_kelas["Varian"] == baris["Varian"])]
            st.dataframe(sub_per.drop(columns=["Skema", "Varian"]),
                         use_container_width=True, hide_index=True)

    if detail is not None:
        with st.expander("Detail prediksi per wilayah-tahun"):
            sub_detail = detail[(detail["Skema"] == baris["Skema"])
                                & (detail["Varian"] == baris["Varian"])]
            st.dataframe(sub_detail, use_container_width=True, hide_index=True,
                         height=280)

    st.warning("Ambang kelas memakai kuantil 1/3 dan 2/3 P0 data latih tiap "
               "lipatan (bukan ambang resmi BPS). Dengan hanya 10 wilayah, satu "
               "kesalahan klasifikasi setara sekitar 8 observasi, sehingga angka "
               "akurasi ini adalah indikasi awal - bukan dasar penentuan kebijakan.")


def tab_prioritas(data: pd.DataFrame) -> None:
    """Tab 5 - peringkat wilayah berdasarkan estimasi P0 tahun terpilih."""
    st.subheader("Prioritas Wilayah Berdasarkan Estimasi P0")
    tabel = muat_csv("estimasi_lipatan.csv")
    if tabel is None:
        st.error("Berkas outputs/estimasi_lipatan.csv belum ada. Jalankan "
                 "`python -m src.evaluate` lebih dahulu.")
        return
    kolom1, kolom2 = st.columns([1, 3])
    tahun = kolom1.selectbox("Tahun", sorted(tabel["Tahun"].unique()),
                             index=len(sorted(tabel["Tahun"].unique())) - 1)
    sub = tabel[tabel["Tahun"] == tahun].copy()
    sub["Estimasi_utama"] = sub["Estimasi_utama"].round(4)
    sub["Peringkat"] = sub["Estimasi_utama"].rank(ascending=False, method="min") \
        .astype(int)
    sub = sub.sort_values("Peringkat")

    fig = px.bar(sub, x="Estimasi_utama", y="Wilayah", orientation="h",
                 color="Estimasi_utama", color_continuous_scale="Reds",
                 text=sub["Estimasi_utama"].round(3),
                 title=f"Estimasi P0 per wilayah, {tahun} (model utama, uji LORO)")
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    tabel_tampil = pd.DataFrame({
        "Peringkat": sub["Peringkat"].values,
        "Wilayah": [cfg.NAMA_TAMPILAN.get(w, w) for w in sub["Wilayah"]],
        "P0 aktual (%)": sub["Aktual"].round(4).values,
        "Estimasi P0 (%)": sub["Estimasi_utama"].values,
    })
    st.dataframe(tabel_tampil, use_container_width=True, hide_index=True)
    st.caption("Peringkat ini menandai wilayah dengan estimasi P0 tertinggi untuk "
               "tahun terpilih. Bersifat statistik-deskriptif, bukan rekomendasi "
               "kebijakan dan bukan klaim sebab-akibat.")



def tab_simulasi(data: pd.DataFrame) -> None:
    """Tab 6 - simulasi perubahan indikator terhadap estimasi P0."""
    st.subheader("Simulasi Perubahan Indikator")
    st.info("Simulasi ini **menahan spatial lag tetap** (nilai tetangga tidak ikut "
            "berubah) dan hanya menghitung ulang komponen PCA memakai scaler dan "
            "PCA yang tersimpan. Ini **simulasi statistik, bukan bukti "
            "sebab-akibat**.")
    model = muat_model()
    kolom1, kolom2 = st.columns(2)
    wilayah = kolom1.selectbox("Wilayah", cfg.WILAYAH, key="sim_wilayah")
    tahun = kolom2.selectbox("Tahun", cfg.TAHUN_MODEL,
                             index=len(cfg.TAHUN_MODEL) - 1, key="sim_tahun")

    baris = data[(data[cfg.KOLOM_WILAYAH] == wilayah)
                 & (data[cfg.KOLOM_TAHUN] == tahun)]
    if baris.empty:
        st.warning("Data wilayah-tahun tersebut tidak tersedia.")
        return
    baris = baris.iloc[[0]].reset_index(drop=True)
    lag = float(model.lag_baris_uji(baris)[0])
    estimasi_awal = float(model.prediksi_dengan_lag(baris, lag)[0])
    aktual = float(baris[cfg.TARGET].iloc[0])

    st.markdown(f"**Nilai awal** — P0 aktual (hasil hitung): "
                f"**{angka(aktual, 4, '%')}**; estimasi model: "
                f"**{angka(estimasi_awal, 4, '%')}**; spatial lag: "
                f"**{angka(lag, 4, '%')}** (ditahan tetap)")

    nilai_baru = {}
    st.markdown("**Ubah indikator** (rentang slider: nilai minimum-maksimum "
                "indikator pada seluruh 80 baris data)")
    kolom = st.columns(3)
    for i, fitur in enumerate(model.kolom_fitur):
        asli = float(baris[fitur].iloc[0])
        bawah = float(data[fitur].min())
        atas = float(data[fitur].max())
        langkah = (atas - bawah) / 200.0 or 0.01
        with kolom[i % 3]:
            nilai_baru[fitur] = st.slider(
                cfg.LABEL_FITUR.get(fitur, fitur), min_value=float(bawah),
                max_value=float(atas), value=float(asli), step=float(langkah),
                key=f"sim_{fitur}")

    baris_baru = baris.copy()
    for fitur, nilai in nilai_baru.items():
        baris_baru[fitur] = nilai
    estimasi_baru = float(model.prediksi_dengan_lag(baris_baru, lag)[0])
    selisih = estimasi_baru - estimasi_awal

    a, b, c = st.columns(3)
    a.metric("Estimasi awal (%)", angka(estimasi_awal, 4))
    b.metric("Estimasi setelah simulasi (%)", angka(estimasi_baru, 4))
    c.metric("Perubahan (poin persentase)", angka(selisih, 4))

    z_lama = model.skor_pca(baris)[0]
    z_baru = model.skor_pca(baris_baru)[0]
    koef_pc = np.array([model.koef[1 + j] for j in range(model.m)])
    kontribusi = pd.DataFrame({
        "Komponen": [f"PC{j + 1}" for j in range(model.m)],
        "Skor sebelum": np.round(z_lama, 4),
        "Skor sesudah": np.round(z_baru, 4),
        "Koefisien": np.round(koef_pc, 4),
        "Kontribusi perubahan (poin %)": np.round(koef_pc * (z_baru - z_lama), 4),
    })
    st.markdown("**Rincian kontribusi tiap komponen terhadap perubahan estimasi**")
    st.dataframe(kontribusi, use_container_width=True, hide_index=True)
    st.caption("Indikator yang diubah: " + ", ".join(
        f"{cfg.LABEL_PENDEK.get(f, f)} = {angka(nilai_baru[f], 2)}"
        for f in model.kolom_fitur))
    st.warning("Catatan batasan: koefisien regresi menggambarkan asosiasi pada data "
               "2017-2024, bukan hubungan sebab-akibat. Perubahan satu indikator "
               "dalam kenyataan biasanya disertai perubahan indikator lain.")



def bilah_samping() -> None:
    """Sidebar: ringkasan proyek, catatan data, dan keputusan desain."""
    with st.sidebar:
        st.title("PBL Estimasi Kemiskinan Kaltim")
        st.caption("Politeknik Negeri Samarinda — 10 kabupaten/kota, 2017-2024")
        st.markdown(
            "**Alur metode**  \n"
            "1. Standardisasi (Z-score)  \n"
            "2. PCA (m komponen, variansi kumulatif ≥ 85%)  \n"
            "3. KNN geospasial → fitur *spatial lag*  \n"
            "4. Regresi Linear Multivariabel (MLR)")
        st.divider()
        st.markdown("**Ringkasan data**")
        st.markdown("- 90 baris (10 wilayah × 9 tahun)  \n"
                    "- Pemodelan: 80 baris (2017-2024)  \n"
                    "- 2016 tidak dipakai: HLS, IDG, TPAK, TPT kosong")
        st.divider()
        st.markdown("**Catatan mutu data**")
        catatan = mod_data.muat_catatan_mutu()
        if catatan.empty:
            st.write("Berkas catatan mutu belum tersedia.")
        else:
            for _, baris in catatan.iterrows():
                st.markdown(f"- **{baris['Kabupaten_Kota']} {int(baris['Tahun'])}** — "
                            f"{baris['Kolom']}: {baris['Tindakan']}")
        st.divider()
        if st.button("Muat ulang data dan model"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption("Semua nilai model disebut *estimasi*. P0 = jumlah penduduk "
                   "miskin / jumlah penduduk × 100 (hasil hitung, bukan angka "
                   "resmi publikasi BPS).")


def main() -> None:
    """Susun halaman dashboard: judul, peringatan, dan enam tab."""
    bilah_samping()
    st.title("Estimasi dan Pemetaan Tingkat Kemiskinan Kabupaten/Kota "
             "Provinsi Kalimantan Timur")
    st.markdown("**Periode 2017-2024 · Standardisasi → PCA → KNN geospasial "
                "(spatial lag) → Regresi Linear Multivariabel**")

    kurang = periksa_berkas()
    if kurang:
        st.error("Berkas berikut belum tersedia:\n\n"
                 + "\n".join(f"- `{p}`" for p in kurang)
                 + "\n\nJalankan berurutan:\n"
                 "```\npython tools\\bangun_dataset.py\npython -m src.eda\n"
                 "python -m src.evaluate\n```")
        st.stop()

    data = muat_data_model()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1 · Peta", "2 · Estimasi vs Aktual", "3 · Hasil PCA",
        "4 · Model & Evaluasi", "5 · Prioritas Wilayah", "6 · Simulasi",
        "7 · KNN & Klasifikasi"])

    with tab1:
        tab_peta(data)
    with tab2:
        tab_estimasi(data)
    with tab3:
        tab_pca(data)
    with tab4:
        tab_model(data)
    with tab5:
        tab_prioritas(data)
    with tab6:
        tab_simulasi(data)
    with tab7:
        tab_knn(data)

    st.divider()
    st.caption("Dibangun dengan Python, scikit-learn, statsmodels, geopandas, "
               "Folium, dan Streamlit. Sumber data: publikasi BPS Provinsi "
               "Kalimantan Timur (dihimpun dari 90 berkas CSV). Lihat README.md "
               "untuk metodologi, batasan, dan daftar keputusan desain.")


if __name__ == "__main__":
    main()

