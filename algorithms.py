from collections import deque, defaultdict
import numpy as np
from graph_ds import BipartiteGraph, merge_sort


class BFSTraversal:
    def __init__(self, graph: BipartiteGraph):
        # Menyimpan instance dari BipartiteGraph
        self.graph = graph

    # BFS: Temukan Similar Users
    def find_similar_users(self, target_user_id: int,
                           max_depth: int = 2) -> tuple:
       
        visited_nodes = set()       # Mencegah kunjungan berulang untuk efisiensi
        similar_users = {}          # Menyimpan id user yang mirip dengan frekuensi pertemuannya
        
        # Inisialisasi Queue BFS (Antrean) dengan format (id, tipe_node, kedalaman)
        queue = deque()
        queue.append((target_user_id, 'user', 0))
        visited_nodes.add(f"user_{target_user_id}")

        # Loop Utama BFS: Terus berjalan selama queue tidak kosong
        while queue:
            current_id, node_type, depth = queue.popleft()   # Ambil data dari depan antrean (FIFO)

            # Hentikan eksplorasi untuk jalur ini jika sudah melebihi max_depth
            if depth >= max_depth:
                continue

            if node_type == 'user':
                # Dari Node User, kunjungi semua Node Movie yang pernah ia rating
                movies_rated = self.graph.get_user_movies(current_id)

                for movie_id in movies_rated:
                    movie_key = f"movie_{movie_id}"
                    if movie_key not in visited_nodes:
                        visited_nodes.add(movie_key)
                        # Masukkan movie ke antrean untuk dieksplorasi di level berikutnya
                        queue.append((movie_id, 'movie', depth + 1))

            elif node_type == 'movie':
                # Dari Node Movie, kunjungi semua Node User yang juga merating movie ini
                users_rated = self.graph.get_movie_users(current_id)

                for user_id, _ in users_rated.items():
                    if user_id == target_user_id:
                        continue  # Skip target user itu sendiri

                    user_key = f"user_{user_id}"
                    if user_key not in visited_nodes:
                        visited_nodes.add(user_key)
                        similar_users[user_id] = 1 # Inisialisasi frekuensi rating bersama
                        # Masukkan user ke antrean (opsional, untuk eksplorasi lebih dalam)
                        queue.append((user_id, 'user', depth + 1))
                    else:
                        # User sudah dikunjungi dari jalur lain, increment co-rated count (jumlah irisan film)
                        if user_id in similar_users:
                            similar_users[user_id] += 1

        return similar_users, visited_nodes

    # BFS: Kumpulkan Candidate Movies

    def get_candidate_movies(self, target_user_id: int,
                             similar_users: dict) -> dict:
        
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


# Collaborative Filtering + Cosine Similarity

class CollaborativeFilter:
    def __init__(self, graph: BipartiteGraph):
        # Menyimpan instance graph untuk mengakses data
        self.graph = graph

    # Cosine Similarity
    def cosine_similarity(self, user1_id: int, user2_id: int) -> float:
        # Menghitung kemiripan (similarity) antara dua user menggunakan metode Cosine Similarity
        movies_u1 = self.graph.get_user_movies(user1_id)
        movies_u2 = self.graph.get_user_movies(user2_id)

        # Step 1: Cari irisan film yang dirating oleh kedua user (Co-rated movies) 
        common_movies = set(movies_u1.keys()) & set(movies_u2.keys())

        if not common_movies:
            return 0.0   # Jika tidak ada film bersama, similarity otomatis = 0

        # Step 2: Buat vektor rating dari irisan film tersebut
        # Memastikan urutan film sama di kedua vektor
        movie_list = list(common_movies)
        ratings_u1 = np.array([movies_u1[m] for m in movie_list], dtype=float)
        ratings_u2 = np.array([movies_u2[m] for m in movie_list], dtype=float)

        # Step 3: Hitung Dot Product (perkalian titik dari kedua vektor)
        dot_product = float(np.dot(ratings_u1, ratings_u2))

        # Step 4: Hitung Norma L2 (panjang/magnitude vektor)
        norm_u1 = float(np.linalg.norm(ratings_u1))
        norm_u2 = float(np.linalg.norm(ratings_u2))

        if norm_u1 == 0.0 or norm_u2 == 0.0:
            return 0.0   # Hindari error pembagian dengan nol

        # Step 5: Dapatkan nilai Cosine Similarity
        similarity = dot_product / (norm_u1 * norm_u2)

        # Batasi output agar nilainya selalu antara 0.0 hingga 1.0
        return max(0.0, min(1.0, similarity))

    # Top-K Similar Users

    def get_top_similar_users(self, target_user_id: int,
                               candidate_users,
                               top_k: int = 10) -> list:
       
        similarities = {}

        for user_id in candidate_users:
            sim = self.cosine_similarity(target_user_id, user_id)
            if sim > 0.0:
                similarities[user_id] = sim

        # Sorting descending berdasarkan nilai similarity
        sorted_users = merge_sort(
            similarities.items(),
            key=lambda pair: pair[1],
            reverse=True
        )

        return sorted_users[:top_k]

    # Recommendation Score

    def calculate_recommendation_score(self, movie_id: int,
                                        similar_users_scores: list,
                                        candidate_movies: dict) -> float:
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

    # Generate Top-N Recommendations

    def get_recommendations(self, target_user_id: int,
                             similar_users_scores: list,
                             candidate_movies: dict,
                             top_n: int = 5) -> list:
       
        recommendations = []

        for movie_id, raters in candidate_movies.items():
            score = self.calculate_recommendation_score(
                movie_id, similar_users_scores, candidate_movies
            )

            if score > 0.0:
                info   = self.graph.movie_info.get(movie_id, {})
                title  = info.get('title',  f'Movie {movie_id}')
                genres = info.get('genres', 'Unknown')
                
                sim_dict = dict(similar_users_scores)
                contributors = []
                for user_id, rating in raters:
                    if user_id in sim_dict:
                        sim = sim_dict[user_id]
                        contributors.append({
                            'user_id': user_id,
                            'rating': rating,
                            'similarity': round(sim, 4),
                            'contribution': round(sim * rating, 4)
                        })
                merge_sort(contributors, key=lambda c: c['contribution'], reverse=True, in_place=True)

                recommendations.append({
                    'movie_id': movie_id,
                    'title'   : title,
                    'genres'  : genres,
                    'score'   : round(score, 4),
                    'rated_by': len(raters),   # berapa similar user yang merating
                    'contributors': contributors
                })

        # Sorting descending 
        merge_sort(recommendations, key=lambda x: x['score'], reverse=True, in_place=True)

        return recommendations[:top_n]