# -*- coding: utf-8 -*-
"""Pembangun berkas panel ``data/dataset_panel_kaltim_2016_2024.xlsx``.

Skrip ini **tidak membuat data baru**. Ia hanya menghimpun (agregasi) 90 berkas
CSV publikasi BPS Provinsi Kalimantan Timur yang sudah tersedia di
``data/sumber_csv/`` menjadi satu tabel panel 10 kabupaten/kota x 9 tahun.

Aturan yang diterapkan:
1. Nilai sel dibaca apa adanya dari CSV sumber.
2. Kolom ``P0_Persen_Miskin`` dihitung dengan rumus
   ``Jumlah_Penduduk_Miskin_Ribu / Jumlah_Penduduk_Ribu x 100``.
3. Satu koreksi terdokumentasi: jumlah penduduk Samarinda 2021 pada berkas
   sumber tercatat 83,1 ribu jiwa (salah ketik) dan dipakai 831,0 ribu jiwa.
4. Sel yang kosong pada sumber (HLS/IDG/TPAK/TPT tahun 2016) dibiarkan kosong.

Menjalankan:  python tools\\bangun_dataset.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402

PROVINSI = "Kalimantan Timur"

#: Pola nama berkas CSV BPS -> kolom panel. ``kolom`` = indeks kolom nilai.
KELOMPOK: dict[str, dict] = {
    cfg.KOLOM_BANTU: {
        "pola": "Jumlah Penduduk Miskin menurut Kabupaten_Kota, {tahun}",
        "kolom": 1,
    },
    "Peng_NonMakanan_Rp": {
        "pola": ("Rata-Rata Pengeluaran Bukan Makanan per Kapita Sebulan "
                 "menurut Kabupaten_Kota, {tahun}"),
        "kolom": 1,
    },
    "Peng_Disesuaikan_RibuRp": {
        "pola": "[Metode Baru] Pengeluaran per Kapita Disesuaikan, {tahun}",
        "kolom": 1,
    },
    "RLS": {
        "pola": "[Metode Baru] Rata-Rata Lama Sekolah (RLS), {tahun}",
        "kolom": 1,
    },
    "HLS_Perempuan": {
        "pola": ("Harapan Lama Sekolah (HLS) Perempuan Menurut "
                 "Kabupaten_Kota, {tahun}"),
        "kolom": 1,
    },
    "UHH": {
        "pola": "[Metode Baru] Umur Harapan Hidup (UHH), {tahun}",
        "kolom": 1,
    },
    "IDG": {
        "pola": "Indeks Pemberdayaan Gender (IDG) Menurut Kabupaten_Kota, {tahun}",
        "kolom": 1,
    },
    "TPAK": {
        "pola": ("Tingkat Partisipasi Angkatan Kerja (TPAK) Menurut "
                 "Kabupaten_Kota, {tahun}"),
        "kolom": 1,
    },
    "TPT": {
        "pola": ("Tingkat Pengangguran Terbuka (TPT) Menurut "
                 "Kabupaten_Kota, {tahun}"),
        "kolom": 1,
    },
}

#: Berkas multi-kolom "Penduduk, Laju Pertumbuhan Penduduk, ...".
POLA_PENDUDUK = (
    "Penduduk, Laju Pertumbuhan Penduduk, Distribusi Persentase Penduduk "
    "Kepadatan Penduduk, Rasio Jenis Kelamin Penduduk Menurut Kabupaten_Kota "
    "di Provinsi Kalimantan Timur, {tahun}"
)

#: Indeks kolom pada berkas penduduk (0 = nama wilayah).
KOLOM_PENDUDUK = {
    "Jumlah_Penduduk_Ribu": 1,
    "Laju_Pertumbuhan_Penduduk": 2,
    "Distribusi_Penduduk_Persen": 3,
    "Kepadatan_per_km2": 4,
    "Rasio_Jenis_Kelamin": 5,
}

#: Koreksi terdokumentasi: (wilayah, tahun, kolom) -> nilai yang dipakai.
KOREKSI = {
    ("Samarinda", 2021, "Jumlah_Penduduk_Ribu"): {
        "nilai": 831.0,
        "alasan": (
            "Berkas sumber BPS 2021 mencatat 83,1 ribu jiwa; tidak konsisten "
            "dengan kolom distribusi penduduk (21,83% x 3.808,2 ribu = 831,3 "
            "ribu). Salah ketik pada sumber (digit '0' hilang) sehingga dipakai "
            "831,0 ribu jiwa."
        ),
    }
}


#: Nama wilayah pada CSV (variasi) -> nama baku dataset.
_NAMA_CSV = {}
for _w in cfg.WILAYAH:
    _NAMA_CSV[_w.lower()] = _w
    _NAMA_CSV[("kota " + _w).lower()] = _w
    _NAMA_CSV[("kabupaten " + _w).lower()] = _w

_NILAI_KOSONG = {"", "-", "--", "...", "na", "n/a", "\u2013", "\u2014"}


def normalisasi_wilayah(nama: str | None) -> str | None:
    """Ubah variasi nama wilayah menjadi nama baku; None bila bukan wilayah."""
    if not nama:
        return None
    bersih = str(nama).replace("\xa0", " ").strip().strip('"')
    return _NAMA_CSV.get(bersih.lower())


def angka(teks: str | None) -> float | None:
    """Konversi teks sel CSV menjadi float; None bila kosong/catatan."""
    if teks is None:
        return None
    bersih = str(teks).strip().strip('"').replace("\xa0", " ").strip()
    if bersih.lower() in _NILAI_KOSONG:
        return None
    bersih = bersih.replace(" ", "")
    try:
        return float(bersih)
    except ValueError:
        return None


def cari_file(pola: str, tahun: int) -> Path:
    """Temukan berkas CSV untuk pola dan tahun tertentu (tahan sufiks ' (1)')."""
    pola_glob = glob.escape(pola.format(tahun=tahun)) + "*.csv"
    kandidat = glob.glob(str(cfg.FOLDER_SUMBER_CSV / pola_glob))
    if not kandidat:
        raise FileNotFoundError(
            f"Berkas CSV untuk pola {pola!r} tahun {tahun} tidak ditemukan "
            f"di {cfg.FOLDER_SUMBER_CSV}"
        )
    tepat = [p for p in kandidat if os.path.basename(p).endswith(f"{tahun}.csv")]
    return Path(sorted(tepat)[0] if tepat else sorted(kandidat)[0])


def baca_tunggal(berkas: Path, indeks: int = 1) -> dict:
    """Baca berkas format judul-lalu-nilai; hasilkan {wilayah: nilai}."""
    nilai: dict[str, float | None] = {}
    with open(berkas, encoding="utf-8-sig", newline="") as f:
        for baris in csv.reader(f):
            if not baris or not baris[0].strip():
                continue
            nama = baris[0].strip()
            if nama == "Kabupaten/Kota" or nama.lower().startswith("keterangan"):
                continue
            if nama == PROVINSI:
                continue
            kode = normalisasi_wilayah(nama)
            if kode is not None:
                nilai[kode] = angka(baris[indeks]) if len(baris) > indeks else None
    return nilai


def baca_penduduk(berkas: Path) -> tuple[dict, dict]:
    """Baca berkas multi-kolom penduduk; hasilkan (per wilayah, baris provinsi)."""
    per_wilayah: dict[str, dict] = {}
    provinsi: dict = {}
    with open(berkas, encoding="utf-8-sig", newline="") as f:
        semua = list(csv.reader(f))
    for baris in semua[1:]:
        if not baris or not baris[0].strip():
            continue
        nama = baris[0].strip()
        if nama.lower().startswith("keterangan"):
            continue
        isi = {
            kolom: (angka(baris[i]) if len(baris) > i else None)
            for kolom, i in KOLOM_PENDUDUK.items()
        }
        if nama == PROVINSI:
            provinsi = isi
            continue
        kode = normalisasi_wilayah(nama)
        if kode is not None:
            per_wilayah[kode] = isi
    return per_wilayah, provinsi


def himpun_panel() -> tuple[pd.DataFrame, list[dict], dict]:
    """Himpun 90 berkas CSV menjadi tabel panel panjang (90 baris)."""
    baris: list[dict] = []
    catatan_mutu: list[dict] = []
    berkas_dipakai: dict[str, str] = {}

    for tahun in cfg.TAHUN_TERSEDIA:
        berkas_penduduk = cari_file(POLA_PENDUDUK, tahun)
        berkas_dipakai[berkas_penduduk.name] = "multi-kolom penduduk"
        per_wilayah, provinsi = baca_penduduk(berkas_penduduk)

        nilai: dict[str, dict] = {w: {} for w in cfg.WILAYAH}
        for kolom, info in KELOMPOK.items():
            berkas = cari_file(info["pola"], tahun)
            berkas_dipakai[berkas.name] = kolom
            isi = baca_tunggal(berkas, info["kolom"])
            for w in cfg.WILAYAH:
                nilai[w][kolom] = isi.get(w)

        for w in cfg.WILAYAH:
            rekam = {
                cfg.KOLOM_WILAYAH: w,
                cfg.KOLOM_TAHUN: tahun,
                cfg.TARGET: None,
                "Peng_NonMakanan_Rp": nilai[w].get("Peng_NonMakanan_Rp"),
                "Peng_Disesuaikan_RibuRp": nilai[w].get("Peng_Disesuaikan_RibuRp"),
                "RLS": nilai[w].get("RLS"),
                "HLS_Perempuan": nilai[w].get("HLS_Perempuan"),
                "UHH": nilai[w].get("UHH"),
                "IDG": nilai[w].get("IDG"),
                "TPAK": nilai[w].get("TPAK"),
                "TPT": nilai[w].get("TPT"),
                cfg.KOLOM_PENDUDUK: per_wilayah.get(w, {}).get(
                    "Jumlah_Penduduk_Ribu"),
                "Laju_Pertumbuhan_Penduduk": per_wilayah.get(w, {}).get(
                    "Laju_Pertumbuhan_Penduduk"),
                "Kepadatan_per_km2": per_wilayah.get(w, {}).get(
                    "Kepadatan_per_km2"),
                "Rasio_Jenis_Kelamin": per_wilayah.get(w, {}).get(
                    "Rasio_Jenis_Kelamin"),
                cfg.KOLOM_BANTU: nilai[w].get(cfg.KOLOM_BANTU),
            }

            # Pemeriksaan mutu: penduduk vs distribusi persentase provinsi
            total_prov = provinsi.get("Jumlah_Penduduk_Ribu")
            pct = per_wilayah.get(w, {}).get("Distribusi_Penduduk_Persen")
            if total_prov and pct and rekam[cfg.KOLOM_PENDUDUK]:
                diharap = pct / 100.0 * total_prov
                selisih = abs(rekam[cfg.KOLOM_PENDUDUK] - diharap) / diharap
                if selisih > 0.10:
                    kunci = (w, tahun, cfg.KOLOM_PENDUDUK)
                    catatan_mutu.append({
                        "Kabupaten_Kota": w,
                        "Tahun": tahun,
                        "Kolom": cfg.KOLOM_PENDUDUK,
                        "Nilai_Sumber": rekam[cfg.KOLOM_PENDUDUK],
                        "Nilai_Pembanding": round(diharap, 1),
                        "Jenis": "tidak konsisten (sumber)",
                        "Tindakan": "dikoreksi" if kunci in KOREKSI else "diperiksa",
                        "Catatan": KOREKSI.get(kunci, {}).get(
                            "alasan",
                            "Selisih > 10% terhadap kolom distribusi penduduk."),
                    })

            # Terapkan koreksi terdokumentasi
            for (kw, kt, kkolom), info in KOREKSI.items():
                if w == kw and tahun == kt:
                    rekam[kkolom] = info["nilai"]

            # P0 = penduduk miskin / penduduk x 100
            pm = rekam[cfg.KOLOM_BANTU]
            jumlah = rekam[cfg.KOLOM_PENDUDUK]
            rekam[cfg.TARGET] = (
                round(pm / jumlah * 100.0, 4) if pm and jumlah else None
            )
            baris.append(rekam)

    urut = [
        cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN, cfg.TARGET,
        "Peng_NonMakanan_Rp", "Peng_Disesuaikan_RibuRp", "RLS", "HLS_Perempuan",
        "UHH", "IDG", "TPAK", "TPT", cfg.KOLOM_PENDUDUK,
        "Laju_Pertumbuhan_Penduduk", "Kepadatan_per_km2", "Rasio_Jenis_Kelamin",
        cfg.KOLOM_BANTU,
    ]
    panel = pd.DataFrame(baris, columns=urut)

    # Pencilan yang sengaja dipertahankan (bukan salah ketik)
    mask = ((panel[cfg.KOLOM_WILAYAH] == "Penajam Paser Utara")
            & (panel[cfg.KOLOM_TAHUN] == 2024))
    if mask.any():
        catatan_mutu.append({
            "Kabupaten_Kota": "Penajam Paser Utara",
            "Tahun": 2024,
            "Kolom": "Laju_Pertumbuhan_Penduduk",
            "Nilai_Sumber": float(panel.loc[mask, "Laju_Pertumbuhan_Penduduk"].iloc[0]),
            "Nilai_Pembanding": None,
            "Jenis": "pencilan dipertahankan",
            "Tindakan": "dipertahankan",
            "Catatan": (
                "Laju pertumbuhan 11,38%/tahun jauh di atas wilayah lain "
                "(< 4%); diduga terkait pembangunan IKN. Nilai dipertahankan "
                "dan ditandai, tidak dihapus."
            ),
        })

    catatan_mutu.extend([
        {
            "Kabupaten_Kota": "Semua wilayah",
            "Tahun": 2019,
            "Kolom": "Laju_Pertumbuhan_Penduduk",
            "Nilai_Sumber": "5,07 - 46,00",
            "Nilai_Pembanding": None,
            "Jenis": "diskontinuitas definisi (sumber)",
            "Tindakan": "dipertahankan",
            "Catatan": (
                "Nilai laju pertumbuhan penduduk pada berkas BPS 2019 "
                "menggambarkan akumulasi 2010-2019 sehingga jauh lebih besar "
                "(5,07-46,00%) daripada tahun lain (0,16-11,38%). Nilai "
                "dipertahankan apa adanya dan perlu ditafsirkan hati-hati."
            ),
        },
        {
            "Kabupaten_Kota": "Semua wilayah",
            "Tahun": 2020,
            "Kolom": "Jumlah_Penduduk_Ribu",
            "Nilai_Sumber": "-",
            "Nilai_Pembanding": None,
            "Jenis": "diskontinuitas basis data (sumber)",
            "Tindakan": "dipertahankan",
            "Catatan": (
                "Angka penduduk mulai 2020 memakai basis Sensus Penduduk 2020 "
                "sehingga levelnya tidak menyambung persis dengan proyeksi "
                "2016-2019. Nilai dipertahankan apa adanya."
            ),
        },
    ])
    return panel, catatan_mutu, berkas_dipakai



def tulis_xlsx(panel: pd.DataFrame, catatan_mutu: list[dict]) -> None:
    """Tulis panel ke xlsx: sheet Data_Panel + Catatan_Mutu_Data + Metadata."""
    cfg.FOLDER_DATA.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame({
        "Butir": [
            "Sumber", "Cakupan", "Rumus P0_Persen_Miskin", "Kolom bantu",
            "Koreksi diterapkan", "Pencilan dipertahankan", "Sel kosong",
            "Imputasi",
        ],
        "Keterangan": [
            "90 berkas CSV publikasi BPS Provinsi Kalimantan Timur "
            "(10 kelompok variabel x 9 tahun) pada data/sumber_csv/.",
            "10 kabupaten/kota x 9 tahun = 90 baris; pemodelan memakai "
            "2017-2024 (80 baris) karena 2016 tidak lengkap.",
            "Jumlah_Penduduk_Miskin_Ribu / Jumlah_Penduduk_Ribu x 100 "
            "(nilai hasil hitung, dibulatkan 4 desimal).",
            "Jumlah_Penduduk_Miskin_Ribu adalah turunan target dan TIDAK "
            "dipakai sebagai fitur (mencegah kebocoran data).",
            "Samarinda 2021: Jumlah_Penduduk_Ribu 83,1 -> 831,0 ribu jiwa "
            "(salah ketik pada berkas sumber).",
            "Penajam Paser Utara 2024: Laju_Pertumbuhan_Penduduk 11,38% "
            "dipertahankan dan ditandai pada EDA.",
            "HLS_Perempuan, IDG, TPAK, TPT kosong 40 sel pada tahun 2016 "
            "(hanya tersedia tingkat provinsi).",
            "Tidak ada imputasi/interpolasi; sel kosong dibiarkan kosong.",
        ],
    })
    with pd.ExcelWriter(cfg.BERKAS_PANEL, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name=cfg.SHEET_PANEL, index=False)
        pd.DataFrame(catatan_mutu).to_excel(
            writer, sheet_name="Catatan_Mutu_Data", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)


def main() -> int:
    print("=" * 78)
    print("MEMBANGUN PANEL DATA KALTIM 2016-2024 DARI 90 BERKAS CSV BPS")
    print("=" * 78)
    panel, catatan_mutu, berkas_dipakai = himpun_panel()
    tulis_xlsx(panel, catatan_mutu)
    pd.DataFrame(catatan_mutu).to_csv(cfg.BERKAS_CATATAN_MUTU, index=False)

    print(f"Berkas CSV dibaca   : {len(berkas_dipakai)}")
    print(f"Panel disimpan ke   : {cfg.BERKAS_PANEL}")
    print(f"Catatan mutu data   : {cfg.BERKAS_CATATAN_MUTU}")
    print(f"Bentuk panel        : {panel.shape[0]} baris x {panel.shape[1]} kolom")
    print(f"Jumlah wilayah      : {panel[cfg.KOLOM_WILAYAH].nunique()}")
    print(f"Tahun               : {sorted(panel[cfg.KOLOM_TAHUN].unique())}")
    print(f"Duplikat kunci      : "
          f"{int(panel.duplicated([cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN]).sum())}")

    kosong = panel.isna().sum()
    kosong = kosong[kosong > 0]
    print("\nSel kosong per kolom:")
    print(kosong.to_string() if len(kosong) else "  (tidak ada)")

    print("\nKelengkapan per tahun (terisi / total sel):")
    for tahun in sorted(panel[cfg.KOLOM_TAHUN].unique()):
        sub = panel[panel[cfg.KOLOM_TAHUN] == tahun]
        print(f"  {tahun}: {int(sub.notna().sum().sum()):3d} / "
              f"{sub.size} terisi, {int(sub.isna().sum().sum()):2d} kosong")

    print("\nCatatan mutu data:")
    print(pd.DataFrame(catatan_mutu).to_string(index=False))

    print("\nCuplik verifikasi (P0, penduduk, laju):")
    cuplik = panel[panel[cfg.KOLOM_TAHUN].isin([2021, 2024])]
    print(cuplik[[cfg.KOLOM_WILAYAH, cfg.KOLOM_TAHUN, cfg.TARGET,
                  cfg.KOLOM_PENDUDUK, "Laju_Pertumbuhan_Penduduk"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

