# ============================================================================
# data_loader.py — Memuat Dataset MovieLens & Membangun Graph
# ============================================================================
#
# Dataset yang digunakan: MovieLens Small
#   URL    : https://grouplens.org/datasets/movielens/latest/
#   File   : ml-latest-small.zip
#
# Struktur file CSV:
#   ratings.csv : userId, movieId, rating, timestamp
#   movies.csv  : movieId, title, genres
#
# Jika file tidak ditemukan, program otomatis menggunakan
# sample dataset bawaan (50 user, 30 film populer).
#
# CARA MENGGUNAKAN DATASET MOVIELENS ASLI:
#   1. Download ml-latest-small.zip dari link di atas.
#   2. Ekstrak ke folder yang sama dengan file .py ini.
#   3. Pastikan folder bernama "ml-latest-small".
#   4. Jalankan program — data akan otomatis terdeteksi.
#
# Kompleksitas build_graph: O(R)  R = jumlah baris rating
# ============================================================================

import os
import pandas as pd
import numpy as np
from graph_ds import BipartiteGraph


class DataLoader:
    """
    Memuat dataset MovieLens (atau sample data) dan
    membangun Weighted Bipartite Graph darinya.
    """

    # Folder default dataset MovieLens small
    DEFAULT_DIR = 'ml-latest-small'

    def __init__(self, data_dir: str = None):
        self.data_dir   = data_dir or self.DEFAULT_DIR
        self.ratings_df = None
        self.movies_df  = None

    # ──────────────────────────────────────────────────────────────────────
    # Load Data
    # ──────────────────────────────────────────────────────────────────────

    def load_data(self) -> tuple:
        """
        Memuat ratings.csv dan movies.csv dari data_dir.
        Jika file tidak ada, buat sample dataset bawaan.

        Returns: (ratings_df, movies_df)
        """
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')
        movies_path  = os.path.join(self.data_dir, 'movies.csv')

        if os.path.exists(ratings_path) and os.path.exists(movies_path):
            print(f"[DataLoader] Memuat MovieLens dari '{self.data_dir}'...")
            self.ratings_df = pd.read_csv(ratings_path)
            self.movies_df  = pd.read_csv(movies_path)
            print(f"[DataLoader] Berhasil: {len(self.ratings_df):,} ratings, "
                  f"{len(self.movies_df):,} movies.")
        else:
            print("[DataLoader] Dataset MovieLens tidak ditemukan.")
            print("[DataLoader] Menggunakan sample dataset bawaan (50 user, 30 film).")
            self._create_sample_data()

        return self.ratings_df, self.movies_df

    # ──────────────────────────────────────────────────────────────────────
    # Sample Data (fallback jika MovieLens tidak ada)
    # ──────────────────────────────────────────────────────────────────────

    def _create_sample_data(self) -> None:
        """
        Membuat dataset sample berisi 50 user dan 30 film populer.
        Digunakan sebagai fallback jika dataset MovieLens tidak tersedia.
        """
        # ── 30 Film Populer ─────────────────────────────────────────────
        movies_data = {
            'movieId': list(range(1, 31)),
            'title'  : [
                'The Shawshank Redemption (1994)',
                'The Godfather (1972)',
                'The Dark Knight (2008)',
                'Schindler\'s List (1993)',
                'Pulp Fiction (1994)',
                'The Lord of the Rings: The Return of the King (2003)',
                'Forrest Gump (1994)',
                'Inception (2010)',
                'The Matrix (1999)',
                'Interstellar (2014)',
                'Goodfellas (1990)',
                'Fight Club (1999)',
                'The Silence of the Lambs (1991)',
                'Saving Private Ryan (1998)',
                'Gladiator (2000)',
                'Avengers: Endgame (2019)',
                'The Lion King (1994)',
                'Jurassic Park (1993)',
                'Back to the Future (1985)',
                'Titanic (1997)',
                'The Sixth Sense (1999)',
                'Parasite (2019)',
                'Whiplash (2014)',
                'La La Land (2016)',
                'The Grand Budapest Hotel (2014)',
                'Get Out (2017)',
                'Mad Max: Fury Road (2015)',
                'Knives Out (2019)',
                'Dune (2021)',
                'Everything Everywhere All at Once (2022)',
            ],
            'genres' : [
                'Crime|Drama',
                'Crime|Drama',
                'Action|Crime|Drama|Thriller',
                'Biography|Drama|History',
                'Crime|Drama',
                'Action|Adventure|Drama|Fantasy',
                'Drama|Romance',
                'Action|Sci-Fi|Thriller',
                'Action|Sci-Fi',
                'Adventure|Drama|Sci-Fi',
                'Biography|Crime|Drama',
                'Drama|Thriller',
                'Crime|Drama|Thriller',
                'Drama|War',
                'Action|Adventure|Drama',
                'Action|Adventure|Sci-Fi',
                'Animation|Adventure|Drama|Family|Musical',
                'Adventure|Sci-Fi|Thriller',
                'Adventure|Comedy|Sci-Fi',
                'Drama|Romance',
                'Drama|Mystery|Thriller',
                'Comedy|Drama|Thriller',
                'Drama|Music',
                'Comedy|Drama|Music|Romance',
                'Comedy|Drama|Romance',
                'Horror|Mystery|Thriller',
                'Action|Adventure|Sci-Fi|Thriller',
                'Comedy|Crime|Drama|Mystery|Thriller',
                'Adventure|Drama|Sci-Fi',
                'Action|Adventure|Comedy|Fantasy|Sci-Fi',
            ]
        }

        # ── Simulasi Rating 50 User ──────────────────────────────────────
        # Kelompok user dengan preferensi berbeda untuk rekomendasi realistis
        np.random.seed(2024)

        rating_choices = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        # Probability: lebih banyak rating tinggi (orang cenderung menonton
        # film yang mereka suka)
        probs = [0.02, 0.03, 0.07, 0.10, 0.15, 0.18, 0.22, 0.13, 0.10]

        # Kelompok genre favorit untuk setiap cluster user
        genre_groups = {
            'action'  : [3, 6, 9, 15, 16, 26, 27, 29],     # movieId action/sci-fi
            'drama'   : [1, 2, 4, 7, 11, 14, 20, 22, 23],   # movieId drama
            'thriller': [5, 8, 12, 13, 21, 25, 28, 30],      # movieId thriller
        }

        ratings_data = []
        for user_id in range(1, 51):   # 50 users
            # Tentukan cluster preferensi user
            cluster = list(genre_groups.keys())[user_id % 3]
            preferred = genre_groups[cluster]
            other     = [m for m in range(1, 31) if m not in preferred]

            # Pilih film dari preferensi (rating lebih tinggi)
            n_pref  = np.random.randint(5, 12)
            chosen_pref = list(np.random.choice(preferred, size=min(n_pref, len(preferred)),
                                                replace=False))
            # Pilih beberapa film di luar preferensi
            n_other = np.random.randint(2, 7)
            chosen_other = list(np.random.choice(other, size=n_other, replace=False))

            all_chosen = chosen_pref + chosen_other

            for movie_id in all_chosen:
                if movie_id in chosen_pref:
                    # Rating lebih tinggi untuk film favorit
                    rating = np.random.choice([3.5, 4.0, 4.5, 5.0],
                                              p=[0.2, 0.3, 0.3, 0.2])
                else:
                    rating = np.random.choice(rating_choices, p=probs)

                ratings_data.append({
                    'userId'   : user_id,
                    'movieId'  : int(movie_id),
                    'rating'   : float(rating),
                    'timestamp': 0,
                })

        self.movies_df  = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data)

        print(f"[DataLoader] Sample data dibuat: "
              f"{len(self.ratings_df)} ratings, "
              f"{len(self.movies_df)} movies, "
              f"50 users.")

    # ──────────────────────────────────────────────────────────────────────
    # Build Graph
    # ──────────────────────────────────────────────────────────────────────

    def build_graph(self, graph: BipartiteGraph,
                    max_users: int = 300) -> BipartiteGraph:
        """
        Membangun Weighted Bipartite Graph dari DataFrame rating.

        Setiap baris di ratings.csv menjadi satu edge di graph:
            userId ──(rating)──► movieId

        Parameter max_users membatasi jumlah user untuk menjaga
        performa GUI saat demo/presentasi.

        Kompleksitas: O(R)  R = jumlah baris rating yang diproses
        """
        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError("Panggil load_data() terlebih dahulu.")

        # ── Load movie metadata ke graph ─────────────────────────────────
        for _, row in self.movies_df.iterrows():
            graph.movie_info[int(row['movieId'])] = {
                'title' : str(row['title']),
                'genres': str(row['genres']),
            }

        # ── Filter user (ambil max_users user pertama) ───────────────────
        unique_users = self.ratings_df['userId'].unique()
        if len(unique_users) > max_users:
            unique_users = unique_users[:max_users]

        filtered = self.ratings_df[
            self.ratings_df['userId'].isin(unique_users)
        ]

        # ── Tambahkan edge ke graph ──────────────────────────────────────
        # Kompleksitas: O(R)
        for _, row in filtered.iterrows():
            graph.add_edge(
                user_id  = int(row['userId']),
                movie_id = int(row['movieId']),
                rating   = float(row['rating']),
            )

        stats = graph.get_stats()
        print(f"[DataLoader] Graph berhasil dibangun:")
        print(f"  Users  : {stats['total_users']}")
        print(f"  Movies : {stats['total_movies']}")
        print(f"  Edges  : {stats['total_edges']}")
        print(f"  Avg ratings/user: {stats['avg_ratings_per_user']:.1f}")

        return graph
