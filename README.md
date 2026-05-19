# 🎬 Sistem Rekomendasi Film Berbasis Graph
**Final Project — Struktur Data dan Algoritma**

---

## Deskripsi Singkat

Program ini mengimplementasikan sistem rekomendasi film menggunakan:

| Konsep | Implementasi |
|---|---|
| Weighted Bipartite Graph | `BipartiteGraph` (adjacency list via `dict`) |
| Graph Traversal | `BFSTraversal` (BFS depth-2, `collections.deque`) |
| User Similarity | `CollaborativeFilter.cosine_similarity()` |
| Rekomendasi | Collaborative Filtering (weighted average rating) |
| Dataset | MovieLens Small (`ratings.csv`, `movies.csv`) |
| GUI | PyQt5 + Matplotlib embedded (dengan fitur Light/Dark Mode) |

Terinspirasi dari paper:
> *"User profile correlation-based similarity (UPCSim) algorithm in movie recommendation system"*

---

## Struktur File

```
movie_recommender/
├── main.py          ← Entry point, jalankan file ini
├── graph_ds.py      ← Kelas BipartiteGraph (adjacency list)
├── algorithms.py    ← Kelas BFSTraversal + CollaborativeFilter
├── data_loader.py   ← Kelas DataLoader (MovieLens / sample data)
├── gui_app.py       ← Kelas MovieRecommenderApp (PyQt5 GUI)
├── requirements.txt ← Dependensi Python
└── README.md        ← Dokumentasi ini
```

---

## Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. (Opsional) Siapkan dataset MovieLens
- Download dari: https://grouplens.org/datasets/movielens/latest/
- Pilih file: `ml-latest-small.zip`
- Ekstrak ke folder yang sama → pastikan ada folder bernama `ml-latest-small`
- Jika tidak ada, program otomatis menggunakan **sample data bawaan** (50 user, 30 film)

### 3. Jalankan program
```bash
python main.py
```

---

## Cara Menggunakan GUI

1. **Login Screen**: Tunggu loading dataset selesai, kemudian masukkan **User ID** di kolom input (contoh: `1`, `5`, `12`) dan klik **Login & Generate**.
2. **Dashboard Utama**:
   - Tab **Recommendations** menampilkan Top 5 Rekomendasi Film dan daftar seluruh film yang sudah pernah ditonton (bisa difilter lewat pencarian).
   - Sidebar sebelah kiri menampilkan daftar **Similar Users** secara dinamis beserta metrik kemiripannya.
   - Tab **Graph Visualization** menampilkan graf bipartite interaktif (Target User, Similar Users, Watched, dan Recommended Movies).
   - Tab **Adjacency List** menampilkan teks graf secara terstruktur.
3. Fitur Tambahan: Terdapat fitur interaktif untuk mengganti tema (**Dark Mode** / **Light Mode**) di pojok kiri bawah, dan tombol **Switch User** untuk kembali ke halaman login.

---

## Alur Algoritma (Flow Program)

```
Load Dataset CSV (pandas)
        │
        ▼
Build Weighted Bipartite Graph
  graph[user][movie] = rating
        │
        ▼
Input User ID
        │
        ▼
BFS Traversal (depth=2)
  Level 0: target_user
  Level 1: movie yang ditonton target_user
  Level 2: user lain yang menonton movie yang sama
        │
        ▼
Hitung Cosine Similarity
  sim(u,v) = (u·v) / (‖u‖ × ‖v‖)
        │
        ▼
BFS Candidate Movies
  Film yang ditonton similar users tapi BELUM ditonton target user
        │
        ▼
Hitung Recommendation Score
  score = Σ(sim × rating) / Σ|sim|
        │
        ▼
Sort Descending → Tampilkan Top-5
```

---

## Penjelasan Struktur Data Utama

### Weighted Graph (Adjacency List)

```python
# Contoh struktur internal:
user_graph = {
    1: {1: 5.0, 3: 4.0, 7: 3.5},   # User 1 menilai Film 1 (5★), 3 (4★), 7 (3.5★)
    2: {1: 4.0, 5: 5.0},             # User 2 menilai Film 1 (4★), 5 (5★)
    3: {3: 3.0, 7: 4.5, 9: 2.0},
}
movie_graph = {
    1: {1: 5.0, 2: 4.0},             # Film 1 dinilai User 1 dan User 2
    3: {1: 4.0, 3: 3.0},
    7: {1: 3.5, 3: 4.5},
}
```

**Kenapa Adjacency List?**
- Cocok untuk sparse graph (tidak semua user menonton semua film).
- Lookup O(1) menggunakan Python `dict`.
- Hemat memori: O(V + E) vs O(V²) untuk adjacency matrix.

---

### BFS pada Graph

```
Queue awal: [(user_1, 'user', depth=0)]

Iterasi 1 — proses user_1 (depth=0):
  → Ambil semua film user_1: [film_A, film_B, film_C]
  → Masukkan ke queue sebagai ('movie', depth=1)
  Queue: [(film_A,'movie',1), (film_B,'movie',1), (film_C,'movie',1)]

Iterasi 2 — proses film_A (depth=1):
  → Ambil semua user yang menilai film_A: [user_4, user_7]
  → Masukkan ke queue sebagai ('user', depth=2)
  → Catat user_4 dan user_7 sebagai similar users
  Queue: [(film_B,'movie',1), ..., (user_4,'user',2), (user_7,'user',2)]

... (lanjut sampai queue kosong atau depth ≥ max_depth)
```

---

## Analisis Kompleksitas

| Operasi | Kompleksitas Waktu | Keterangan |
|---|---|---|
| `add_edge()` | O(1) | Dictionary insert |
| `get_user_movies()` | O(1) | Dictionary lookup |
| `build_graph()` | O(R) | R = jumlah rating |
| BFS `find_similar_users()` | O(V + E) | V=node, E=edge |
| BFS `get_candidate_movies()` | O(K × M) | K=similar users, M=films/user |
| `cosine_similarity()` | O(M) | M=co-rated movies |
| `get_top_similar_users()` | O(K × M + K log K) | K similar users |
| `get_recommendations()` | O(C × K + C log C) | C=candidates, K=similar users |

---

## Kustomisasi Tampilan

Aplikasi ini menggunakan **PyQt5** dan mendukung tema dinamis secara langsung lewat *Stylesheet* (QSS). Konfigurasi warna berpusat pada dictionaries global (file `gui_app.py`):

- **THEME_LIGHT** dan **THEME_DARK**: Menyimpan token warna untuk background, text primer/sekunder, borders, dsb.
- Fungsi **get_qss(P)** akan me-render seluruh *stylesheet* untuk antarmuka.
- Untuk mengubah visual *node* & *edge* di Matplotlib, modifikasi variabel `C_TARGET`, `C_SIMILAR`, `C_REC`, dan `C_WATCHED` di dalam fungsi `draw_graph()`.

---

## Dependensi

| Library | Versi Min | Kegunaan |
|---|---|---|
| `pandas` | 1.5.0 | Baca file CSV dataset |
| `numpy` | 1.23.0 | Komputasi vektor (cosine sim) |
| `matplotlib` | 3.6.0 | Visualisasi graph di GUI |
| `PyQt5` | 5.15.0 | GUI utama yang dinamis & modern |
| `collections` | bawaan Python | `deque` untuk BFS |

> Tidak menggunakan library ML tingkat tinggi (TensorFlow, Surprise, PyTorch, NetworkX).
> Semua algoritma diimplementasikan secara manual.
