import os
import pandas as pd
import numpy as np
from graph_ds import BipartiteGraph

class DataLoader:
    # Class untuk memuat dataset CSV (ratings dan movies) dan mengonversinya ke graf
    DEFAULT_DIR = '.' # Direktori default jika tidak ada direktori yang diberikan

    def __init__(self, data_dir: str = None):
        # Inisialisasi direktori data dan dataframe
        self.data_dir   = data_dir or self.DEFAULT_DIR
        self.ratings_df = None # Menyimpan data dari ratings.csv
        self.movies_df  = None # Menyimpan data dari movies.csv

    def load_data(self) -> tuple:
        # Fungsi utama untuk membaca file CSV menjadi Pandas DataFrame
        ratings_path = os.path.join(self.data_dir, 'ratings.csv') # Path lengkap ke ratings.csv
        movies_path = os.path.join(self.data_dir, 'movies.csv')   # Path lengkap ke movies.csv

        print(f"[DataLoader] Cari ratings di: {ratings_path}")
        print(f"[DataLoader] Cari movies di : {movies_path}")

        # Pengecekan apakah file ada di direktori yang dituju
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"ratings.csv tidak ditemukan di {ratings_path}")

        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"movies.csv tidak ditemukan di {movies_path}")

        print("[DataLoader] Memuat dataset MovieLens...")

        # Membaca file CSV menggunakan pandas dan menyimpannya di atribut class
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)

        print(
            f"[DataLoader] Berhasil load "
            f"{len(self.ratings_df):,} ratings dan "
            f"{len(self.movies_df):,} movies."
        )

        return self.ratings_df, self.movies_df

    def build_graph(self, graph: BipartiteGraph,
                    max_users: int = None,
                    max_movies: int = None) -> BipartiteGraph:
        # Fungsi untuk memasukkan data dari DataFrame ke dalam struktur graf bipartit
        # Pastikan data sudah diload sebelumnya
        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError("Panggil load_data() terlebih dahulu.")

        # Memilih user unik sejumlah max_users
        selected_users = (
            self.ratings_df['userId']
            .drop_duplicates()
            .head(max_users)
            .tolist()
        )

        # Memilih movie id unik sejumlah max_movies dari user yang terpilih
        selected_movie_ids = (
            self.ratings_df[self.ratings_df['userId'].isin(selected_users)]['movieId']
            .drop_duplicates()
            .head(max_movies)
            .tolist()
        )

        selected_movie_ids = set(selected_movie_ids) # Ubah ke set agar lookup lebih cepat

        # Ambil detail movie dari dataframe movies.csv berdasarkan selected_movie_ids
        limited_movies = self.movies_df[self.movies_df['movieId'].isin(selected_movie_ids)]

        # Masukkan informasi tiap film (id, judul, genre) ke dalam struktur graph
        for _, row in limited_movies.iterrows():
            graph.movie_info[int(row['movieId'])] = {
                'title': str(row['title']),
                'genres': str(row['genres'])
            }

        # Filter data ratings yang hanya melibatkan user dan movie yang sudah terpilih
        filtered_ratings = self.ratings_df[
            (self.ratings_df['userId'].isin(selected_users)) &
            (self.ratings_df['movieId'].isin(selected_movie_ids))
        ]

        # Tambahkan edge (hubungan) ke dalam graph antara user dan movie dengan nilai bobot berupa rating
        for _, row in filtered_ratings.iterrows():
            graph.add_edge(
                user_id=int(row['userId']),
                movie_id=int(row['movieId']),
                rating=float(row['rating'])
            )

        # Dapatkan statistik graph (jumlah user, movie, rata-rata rating)
        stats = graph.get_stats()

        print("\n[DataLoader] Graph berhasil dibangun:")
        print(f"Users  : {stats['total_users']}")
        print(f"Movies : {stats['total_movies']}")
        print(f"Edges  : {stats['total_edges']}")
        print(f"Avg ratings/user: {stats['avg_ratings_per_user']:.1f}")

        return graph
