# ============================================================================
# algorithms.py — BFS Traversal + Collaborative Filtering
# ============================================================================
#
# FILE INI BERISI DUA ALGORITMA UTAMA:
#
#  1. BFSTraversal
#     ─ Breadth First Search untuk menjelajahi bipartite graph.
#     ─ Menemukan similar users dan candidate movies.
#
#  2. CollaborativeFilter
#     ─ Menghitung Cosine Similarity antar user.
#     ─ Menghasilkan recommendation score.
#     ─ Menghasilkan top-N rekomendasi.
#
# ============================================================================
#
# ─── BFS (Breadth First Search) ───────────────────────────────────────────
#
#  BFS menjelajahi graph secara "melebar" level per level.
#  Menggunakan struktur data QUEUE (deque) yang bersifat FIFO.
#
#  Cara kerja BFS pada Bipartite Graph ini:
#
#    Level 0 : [target_user]                ← titik awal
#    Level 1 : [movie_A, movie_B, ...]      ← semua film yg ditonton target
#    Level 2 : [user_X, user_Y, ...]        ← semua user yg nonton film yg sama
#
#    Visualisasi:
#    target_user ─── movie_A ─── user_X   ← user_X menonton movie_A juga!
#              └─── movie_B ─── user_Y
#                         └─── user_Z
#
#  Kompleksitas BFS: O(V + E)
#    V = jumlah node yang dikunjungi (users + movies)
#    E = jumlah edge yang diperiksa (ratings)
#
# ─── COSINE SIMILARITY ────────────────────────────────────────────────────
#
#  Terinspirasi dari paper UPCSim: mengukur kemiripan user berdasarkan
#  pola rating film yang sama (co-rated movies).
#
#  Formula:
#    sim(u, v) = (u · v) / (‖u‖ × ‖v‖)
#
#  Contoh:
#    User 1 merating: [Film_A=5, Film_B=3, Film_C=4]
#    User 2 merating: [Film_A=4, Film_B=4, Film_C=3]
#    Common movies  : [Film_A, Film_B, Film_C]
#
#    u · v = (5×4) + (3×4) + (4×3) = 20 + 12 + 12 = 44
#    ‖u‖   = √(5²+3²+4²) = √50 ≈ 7.07
#    ‖v‖   = √(4²+4²+3²) = √41 ≈ 6.40
#    sim   = 44 / (7.07 × 6.40) ≈ 0.97  ← sangat mirip!
#
#  Nilai: 0.0 (tidak mirip) hingga 1.0 (sangat mirip)
#
# ─── COLLABORATIVE FILTERING ──────────────────────────────────────────────
#
#  Ide dasar: "Orang yang punya selera mirip akan suka film yang sama."
#
#  Langkah:
#    1. Temukan user dengan selera mirip (BFS + cosine similarity).
#    2. Kumpulkan film yang ditonton mereka tapi belum ditonton target.
#    3. Hitung recommendation score = weighted average rating.
#       score(film) = Σ(sim(u,v) × rating(v,film)) / Σ|sim(u,v)|
#
#  Kompleksitas recommendation: O(K × M)
#    K = jumlah similar users
#    M = jumlah candidate movies
# ============================================================================

from collections import deque, defaultdict
import numpy as np
from graph_ds import BipartiteGraph


class BFSTraversal:
    """
    Implementasi Breadth First Search pada Weighted Bipartite Graph.

    Digunakan untuk:
    - Menemukan user-user yang memiliki riwayat menonton serupa.
    - Mengumpulkan candidate movie untuk rekomendasi.
    """

    def __init__(self, graph: BipartiteGraph):
        self.graph = graph

    # ──────────────────────────────────────────────────────────────────────
    # BFS: Temukan Similar Users
    # ──────────────────────────────────────────────────────────────────────

    def find_similar_users(self, target_user_id: int,
                           max_depth: int = 2) -> tuple:
        """
        BFS 2-level untuk menemukan user dengan film yang sama.

        Alur traversal:
          Depth 0 : Start dari target_user
          Depth 1 : Eksplorasi semua movie yang ditonton target_user
          Depth 2 : Dari setiap movie, temukan user lain yang juga menontonnya

        Parameter max_depth=2 berarti kita berhenti di Level 2 (similar users).

        Returns
        -------
        similar_users  : dict { userId: co_rated_count }
                         co_rated_count = berapa banyak film bersama yang ditemukan
        visited_nodes  : set semua node yang dikunjungi selama traversal

        Kompleksitas: O(U × M)
          U = avg jumlah film yang dirating setiap user
          M = avg jumlah user yang merating setiap film
        """
        visited_nodes = set()       # Mencegah kunjungan berulang
        similar_users = {}          # { userId: co_rated_count }

        # ── Inisialisasi Queue BFS ──────────────────────────────────────
        # Setiap elemen queue: (node_id, node_type, depth)
        queue = deque()
        queue.append((target_user_id, 'user', 0))
        visited_nodes.add(f"user_{target_user_id}")

        # ── Loop Utama BFS ──────────────────────────────────────────────
        while queue:
            current_id, node_type, depth = queue.popleft()   # FIFO

            # Hentikan eksplorasi jika sudah melebihi max_depth
            if depth >= max_depth:
                continue

            if node_type == 'user':
                # ── Node User → kunjungi semua Movie-nya (Level naik 1) ──
                movies_rated = self.graph.get_user_movies(current_id)

                for movie_id in movies_rated:
                    movie_key = f"movie_{movie_id}"
                    if movie_key not in visited_nodes:
                        visited_nodes.add(movie_key)
                        # Masukkan movie ke queue dengan depth+1
                        queue.append((movie_id, 'movie', depth + 1))

            elif node_type == 'movie':
                # ── Node Movie → kunjungi semua User yang merating ──────
                users_rated = self.graph.get_movie_users(current_id)

                for user_id, _ in users_rated.items():
                    if user_id == target_user_id:
                        continue  # skip target user itu sendiri

                    user_key = f"user_{user_id}"
                    if user_key not in visited_nodes:
                        visited_nodes.add(user_key)
                        similar_users[user_id] = 1
                        # Masukkan user ke queue (opsional, untuk eksplorasi lebih)
                        queue.append((user_id, 'user', depth + 1))
                    else:
                        # User sudah dikunjungi → increment co-rated count
                        if user_id in similar_users:
                            similar_users[user_id] += 1

        return similar_users, visited_nodes

    # ──────────────────────────────────────────────────────────────────────
    # BFS: Kumpulkan Candidate Movies
    # ──────────────────────────────────────────────────────────────────────

    def get_candidate_movies(self, target_user_id: int,
                             similar_users: dict) -> dict:
        """
        BFS untuk mengumpulkan film yang belum ditonton target_user
        dari semua similar users.

        Candidate movie = film yang ditonton similar_user
                          TAPI belum ditonton target_user.

        Returns
        -------
        candidate_movies : dict { movieId: [(userId, rating), ...] }
                           Setiap film → list pasangan (user, rating) dari
                           similar users yang sudah menontonnya.

        Kompleksitas: O(K × M)  K=similar users, M=film per user
        """
        target_movies = set(self.graph.get_user_movies(target_user_id).keys())
        candidate_movies = defaultdict(list)

        # BFS sederhana: setiap similar user sebagai titik start
        queue = deque(similar_users.keys())
        visited = set()

        while queue:
            user_id = queue.popleft()
            if user_id in visited:
                continue
            visited.add(user_id)

            user_movies = self.graph.get_user_movies(user_id)
            for movie_id, rating in user_movies.items():
                # Hanya masukkan film yang BELUM ditonton target_user
                if movie_id not in target_movies:
                    candidate_movies[movie_id].append((user_id, rating))

        return dict(candidate_movies)


# ============================================================================
# Collaborative Filtering + Cosine Similarity
# ============================================================================

class CollaborativeFilter:
    """
    User-Based Collaborative Filtering dengan Cosine Similarity.

    Terinspirasi dari UPCSim: menggunakan korelasi profil rating user
    untuk mengukur kemiripan dan menghasilkan rekomendasi.
    """

    def __init__(self, graph: BipartiteGraph):
        self.graph = graph

    # ──────────────────────────────────────────────────────────────────────
    # Cosine Similarity
    # ──────────────────────────────────────────────────────────────────────

    def cosine_similarity(self, user1_id: int, user2_id: int) -> float:
        """
        Menghitung Cosine Similarity antara dua user.

        Formula: sim(u,v) = (u · v) / (‖u‖ × ‖v‖)

        Langkah implementasi:
          1. Cari intersection film (co-rated movies).
          2. Buat vektor rating (hanya untuk film bersama).
          3. Hitung dot product.
          4. Hitung norma L2 masing-masing vektor.
          5. Bagi dot product dengan perkalian norma.

        Return: float dalam rentang [0.0, 1.0]
        Kompleksitas: O(M)  M = jumlah co-rated movies
        """
        movies_u1 = self.graph.get_user_movies(user1_id)
        movies_u2 = self.graph.get_user_movies(user2_id)

        # ── Step 1: Co-rated movies (intersection) ──────────────────────
        common_movies = set(movies_u1.keys()) & set(movies_u2.keys())

        if not common_movies:
            return 0.0   # Tidak ada film bersama → similarity = 0

        # ── Step 2: Vektor rating ────────────────────────────────────────
        # Urutan film harus sama untuk kedua user
        movie_list = list(common_movies)
        ratings_u1 = np.array([movies_u1[m] for m in movie_list], dtype=float)
        ratings_u2 = np.array([movies_u2[m] for m in movie_list], dtype=float)

        # ── Step 3: Dot Product ──────────────────────────────────────────
        dot_product = float(np.dot(ratings_u1, ratings_u2))

        # ── Step 4: Norma L2 ─────────────────────────────────────────────
        norm_u1 = float(np.linalg.norm(ratings_u1))
        norm_u2 = float(np.linalg.norm(ratings_u2))

        if norm_u1 == 0.0 or norm_u2 == 0.0:
            return 0.0   # Hindari pembagian nol

        # ── Step 5: Cosine Similarity ────────────────────────────────────
        similarity = dot_product / (norm_u1 * norm_u2)

        # Clamp ke [0, 1] (rating selalu positif sehingga cosine ≥ 0)
        return max(0.0, min(1.0, similarity))

    # ──────────────────────────────────────────────────────────────────────
    # Top-K Similar Users
    # ──────────────────────────────────────────────────────────────────────

    def get_top_similar_users(self, target_user_id: int,
                               candidate_users,
                               top_k: int = 10) -> list:
        """
        Menghitung cosine similarity ke semua candidate users
        dan mengembalikan top-K yang paling mirip.

        Returns
        -------
        list of (userId, similarity_score)  — diurutkan descending

        Kompleksitas: O(K × M) hitung sim + O(K log K) sort
          K = jumlah candidate users
          M = rata-rata co-rated movies
        """
        similarities = {}

        for user_id in candidate_users:
            sim = self.cosine_similarity(target_user_id, user_id)
            if sim > 0.0:
                similarities[user_id] = sim

        # Sorting descending berdasarkan nilai similarity
        sorted_users = sorted(
            similarities.items(),
            key=lambda pair: pair[1],
            reverse=True
        )

        return sorted_users[:top_k]

    # ──────────────────────────────────────────────────────────────────────
    # Recommendation Score
    # ──────────────────────────────────────────────────────────────────────

    def calculate_recommendation_score(self, movie_id: int,
                                        similar_users_scores: list,
                                        candidate_movies: dict) -> float:
        """
        Menghitung skor rekomendasi sebuah film menggunakan
        weighted average rating dari similar users.

        Formula:
          score(film) = Σ( sim(u,v) × rating(v, film) )
                        ─────────────────────────────────
                              Σ | sim(u,v) |

        Intuisi: film yang dirating tinggi oleh user yang sangat mirip
                 dengan target user mendapat skor lebih tinggi.

        Kompleksitas: O(K)  K = similar users yang merating film ini
        """
        if movie_id not in candidate_movies:
            return 0.0

        sim_dict = dict(similar_users_scores)

        numerator   = 0.0   # Σ( sim × rating )
        denominator = 0.0   # Σ| sim |

        for user_id, rating in candidate_movies[movie_id]:
            if user_id in sim_dict:
                sim          = sim_dict[user_id]
                numerator   += sim * rating
                denominator += abs(sim)

        if denominator == 0.0:
            return 0.0

        return numerator / denominator

    # ──────────────────────────────────────────────────────────────────────
    # Generate Top-N Recommendations
    # ──────────────────────────────────────────────────────────────────────

    def get_recommendations(self, target_user_id: int,
                             similar_users_scores: list,
                             candidate_movies: dict,
                             top_n: int = 5) -> list:
        """
        Menghasilkan top-N rekomendasi film untuk target user.

        Langkah:
          1. Untuk setiap candidate movie, hitung recommendation score.
          2. Filter yang score-nya > 0.
          3. Sort descending berdasarkan score.
          4. Kembalikan top-N.

        Returns
        -------
        list of dict dengan kunci:
            movie_id, title, genres, score, rated_by (jumlah similar user yang menilai)

        Kompleksitas: O(M × K + M log M)
          M = candidate movies, K = similar users
        """
        recommendations = []

        for movie_id, raters in candidate_movies.items():
            score = self.calculate_recommendation_score(
                movie_id, similar_users_scores, candidate_movies
            )

            if score > 0.0:
                info   = self.graph.movie_info.get(movie_id, {})
                title  = info.get('title',  f'Movie {movie_id}')
                genres = info.get('genres', 'Unknown')

                recommendations.append({
                    'movie_id': movie_id,
                    'title'   : title,
                    'genres'  : genres,
                    'score'   : round(score, 4),
                    'rated_by': len(raters),   # berapa similar user yang merating
                })

        # Sorting descending — O(M log M)
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations[:top_n]
