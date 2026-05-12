# ============================================================================
# graph_ds.py — Struktur Data: Weighted Bipartite Graph
# ============================================================================
#
# KONSEP WEIGHTED BIPARTITE GRAPH:
#   - Bipartite: dua set node yang berbeda (User dan Movie).
#     Edge hanya boleh menghubungkan node dari set berbeda.
#   - Weighted: setiap edge punya bobot (nilai rating 1.0–5.0).
#   - Representasi: Adjacency List menggunakan nested dictionary.
#
# KENAPA ADJACENCY LIST?
#   - Efisien untuk sparse graph (jarang terhubung penuh).
#   - Lookup O(1) menggunakan dictionary Python.
#   - Hemat memori: O(V + E) vs O(V²) untuk adjacency matrix.
#
# CONTOH STRUKTUR:
#   user_graph = {
#       1: {1: 5.0, 2: 3.5, 7: 4.0},   # User 1 merating film 1, 2, 7
#       2: {1: 4.0, 3: 2.0}              # User 2 merating film 1, 3
#   }
#   movie_graph = {
#       1: {1: 5.0, 2: 4.0},             # Film 1 dirating User 1 dan 2
#       2: {1: 3.5},                      # Film 2 hanya dirating User 1
#   }
#
# KOMPLEKSITAS:
#   add_edge()         : O(1) — dictionary insert
#   get_user_movies()  : O(1) — dictionary lookup
#   get_movie_users()  : O(1) — dictionary lookup
#   Ruang              : O(V + E) — V=node, E=edge
# ============================================================================

from collections import defaultdict
import numpy as np


class BipartiteGraph:
    """
    Weighted Bipartite Graph berbasis Adjacency List.

    Dua set node:
        - LEFT  : User nodes   (diidentifikasi dengan integer userId)
        - RIGHT : Movie nodes  (diidentifikasi dengan integer movieId)

    Edge:
        - Menghubungkan satu user dengan satu movie.
        - Bobot edge = nilai rating yang diberikan user ke movie tersebut.

    Internal storage:
        - self.user_graph  : dict[userId  -> dict[movieId -> rating]]
        - self.movie_graph : dict[movieId -> dict[userId  -> rating]]
        Dua arah agar traversal dari kedua sisi efisien O(1).
    """

    def __init__(self):
        # ── Adjacency List Utama ──────────────────────────────────────────
        # Arah user  → movie  (digunakan saat mencari film yang sudah ditonton)
        self.user_graph: dict = defaultdict(dict)

        # Arah movie → user  (digunakan saat BFS mencari penonton film yang sama)
        self.movie_graph: dict = defaultdict(dict)

        # ── Set Node ─────────────────────────────────────────────────────
        self.users: set = set()    # semua userId yang terdaftar
        self.movies: set = set()   # semua movieId yang terdaftar

        # ── Metadata Film ────────────────────────────────────────────────
        # Menyimpan title & genres agar tidak perlu meng-query DataFrame
        # Struktur: { movieId: {'title': str, 'genres': str} }
        self.movie_info: dict = {}

    # ──────────────────────────────────────────────────────────────────────
    # Operasi Dasar Graph
    # ──────────────────────────────────────────────────────────────────────

    def add_edge(self, user_id: int, movie_id: int, rating: float) -> None:
        """
        Menambahkan edge berbobot antara user dan movie.

        Kompleksitas Waktu : O(1) — dictionary assignment
        Kompleksitas Ruang : O(1) per pemanggilan (amortized)

        Parameters
        ----------
        user_id  : ID user (node sisi kiri bipartite graph)
        movie_id : ID movie (node sisi kanan bipartite graph)
        rating   : Bobot edge, berupa nilai rating (0.5–5.0)
        """
        self.user_graph[user_id][movie_id] = rating
        self.movie_graph[movie_id][user_id] = rating
        self.users.add(user_id)
        self.movies.add(movie_id)

    def get_user_movies(self, user_id: int) -> dict:
        """
        Mengembalikan semua film beserta rating-nya untuk satu user.

        Return: dict { movieId: rating } atau {} jika user tidak ada.
        Kompleksitas: O(1)
        """
        return self.user_graph.get(user_id, {})

    def get_movie_users(self, movie_id: int) -> dict:
        """
        Mengembalikan semua user beserta rating-nya untuk satu film.

        Return: dict { userId: rating } atau {} jika film tidak ada.
        Kompleksitas: O(1)
        """
        return self.movie_graph.get(movie_id, {})

    def has_user(self, user_id: int) -> bool:
        """Cek apakah user ada di graph. O(1)."""
        return user_id in self.users

    # ──────────────────────────────────────────────────────────────────────
    # Representasi & Statistik
    # ──────────────────────────────────────────────────────────────────────

    def get_adjacency_list_str(self, user_id: int, limit: int = 10) -> str:
        """
        Mengembalikan string representasi adjacency list untuk satu user.
        Berguna untuk ditampilkan di GUI.

        Kompleksitas: O(min(degree(user), limit))
        """
        movies = self.get_user_movies(user_id)
        lines = [f"User {user_id}:  [{len(movies)} film dirating]"]

        # Urutkan berdasarkan rating tertinggi
        sorted_movies = sorted(movies.items(), key=lambda x: x[1], reverse=True)

        for movie_id, rating in sorted_movies[:limit]:
            info  = self.movie_info.get(movie_id, {})
            title = info.get('title', f'Movie {movie_id}')
            stars = '★' * int(rating) + '☆' * (5 - int(rating))
            lines.append(f"  ├─ [{movie_id:4}] {title[:32]:<32}  {stars}  ({rating})")

        if len(movies) > limit:
            lines.append(f"  └─ ... dan {len(movies) - limit} film lainnya")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """
        Mengembalikan statistik ringkas tentang graph.

        Kompleksitas: O(U) — U = jumlah user (untuk mean)
        """
        degrees = [len(v) for v in self.user_graph.values()]
        return {
            'total_users'           : len(self.users),
            'total_movies'          : len(self.movies),
            'total_edges'           : sum(degrees),
            'avg_ratings_per_user'  : float(np.mean(degrees)) if degrees else 0.0,
            'max_ratings_per_user'  : max(degrees) if degrees else 0,
            'min_ratings_per_user'  : min(degrees) if degrees else 0,
        }

    def get_user_top_movies(self, user_id: int, top_n: int = 5) -> list:
        """
        Mengembalikan top-N film favorit user (rating tertinggi).

        Return: list of (movie_id, title, rating)
        Kompleksitas: O(M log M) — M = film yang dirating user ini
        """
        movies = self.get_user_movies(user_id)
        sorted_movies = sorted(movies.items(), key=lambda x: x[1], reverse=True)
        result = []
        for movie_id, rating in sorted_movies[:top_n]:
            title = self.movie_info.get(movie_id, {}).get('title', f'Movie {movie_id}')
            result.append((movie_id, title, rating))
        return result
