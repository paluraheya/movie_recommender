import sys
import os

# Menambahkan folder tempat file ini berada ke sys.path 
# agar modul-modul lain di folder yang sama (seperti gui_app) bisa di-import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
# Mengimpor fungsi main dari file gui_app.py
from gui_app import main
 
# Blok ini memastikan kode di dalamnya hanya berjalan jika file dieksekusi langsung
# (bukan saat di-import sebagai modul oleh file lain)
if __name__ == "__main__":
    # Menjalankan fungsi utama aplikasi GUI
    main()
