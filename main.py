# ============================================================================
# main.py — Entry Point Aplikasi
# ============================================================================
#
# Jalankan file ini untuk memulai aplikasi:
#   $ python main.py
#
# Pastikan dependensi sudah terinstal:
#   $ pip install -r requirements.txt
#
# (Opsional) Letakkan folder "ml-latest-small" di direktori yang sama
# agar program menggunakan dataset MovieLens asli.
# Jika tidak ada, program otomatis menggunakan sample data bawaan.
# ============================================================================

import tkinter as tk
import sys
import os

# Tambahkan direktori saat ini ke path agar import antar file berjalan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_app import MovieRecommenderApp


def main():
    """Fungsi utama — inisialisasi Tkinter dan jalankan aplikasi."""
    root = tk.Tk()

    # Ikon aplikasi (opsional, abaikan jika tidak ada file .ico)
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass

    # Jalankan aplikasi
    app = MovieRecommenderApp(root)

    # Mulai event loop Tkinter
    root.mainloop()


if __name__ == "__main__":
    main()
