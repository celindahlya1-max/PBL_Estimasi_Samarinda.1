# -*- coding: utf-8 -*-
"""Uji cepat dashboard tanpa membuka browser (streamlit.testing.AppTest).

AppTest menjalankan app.py persis seperti sesi Streamlit sungguhan: seluruh isi
keenam tab dieksekusi, widget diisi nilai bawaannya, dan setiap exception
ditangkap. Skrip ini **tidak** mengubah berkas apa pun.

Menjalankan:  python tools\\uji_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

AKAR = Path(__file__).resolve().parent.parent


def ringkas(daftar, nama: str) -> str:
    """Ringkasan aman untuk widget/elemen yang mungkin tidak didukung AppTest."""
    try:
        return f"{nama}: {len(daftar)}"
    except Exception:  # pragma: no cover - hanya untuk pelaporan
        return f"{nama}: tidak dapat dibaca"


def main() -> int:
    app = AKAR / "app.py"
    print(f"Menjalankan {app.name} lewat AppTest ...")
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()

    if at.exception:
        print(f"DITEMUKAN {len(at.exception)} EXCEPTION:")
        for exc in at.exception:
            print("-" * 70)
            print(exc.value)
        return 1

    print("Tidak ada exception.")
    print(ringkas(at.tabs, "Komponen tab"))
    print(ringkas(at.dataframe, "Tabel (dataframe)"))
    print(ringkas(at.metric, "Kartu metrik"))
    print(ringkas(at.selectbox, "Selectbox"))
    print(ringkas(at.multiselect, "Multiselect"))
    print(ringkas(at.radio, "Radio"))
    print(ringkas(at.slider, "Slider"))
    print(ringkas(at.button, "Tombol"))
    print("\nPesan peringatan/informasi yang tampil:")

    def tampilkan_teks(label: str, elemen) -> None:
        for e in elemen:
            if getattr(e, "value", None):
                print(f"  [{label}] {e.value}")

    tampilkan_teks("error", at.error)
    tampilkan_teks("warning", at.warning)
    tampilkan_teks("info", at.info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
