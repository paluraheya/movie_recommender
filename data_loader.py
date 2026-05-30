
import os
import pandas as pd
import numpy as np
from graph_ds import BipartiteGraph


class DataLoader:
    

    DEFAULT_DIR = '.'

    def __init__(self, data_dir: str = None):
        self.data_dir   = data_dir or self.DEFAULT_DIR
        self.ratings_df = None
        self.movies_df  = None

    def load_data(self) -> tuple:
        
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')
        movies_path = os.path.join(self.data_dir, 'movies.csv')

        print(f"[DataLoader] Cari ratings di: {ratings_path}")
        print(f"[DataLoader] Cari movies di : {movies_path}")

        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"ratings.csv tidak ditemukan di {ratings_path}"
            )

        if not os.path.exists(movies_path):
            raise FileNotFoundError(
                f"movies.csv tidak ditemukan di {movies_path}"
            )

        print("[DataLoader] Memuat dataset MovieLens...")

        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)

        print(
            f"[DataLoader] Berhasil load "
            f"{len(self.ratings_df):,} ratings dan "
            f"{len(self.movies_df):,} movies."
        )

        return self.ratings_df, self.movies_df

    

    
    def build_graph(self, graph: BipartiteGraph,
                    max_users: int = 50,
                    max_movies: int = None) -> BipartiteGraph:
        
       

        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError(
                "Panggil load_data() terlebih dahulu."
            )

        selected_users = (
            self.ratings_df['userId']
            .drop_duplicates()
            .head(max_users)
            .tolist()
        )

        selected_movie_ids = (
    self.ratings_df[
        self.ratings_df['userId'].isin(selected_users)
    ]['movieId']
    .drop_duplicates()
    .head(max_movies)
    .tolist()
)

        selected_movie_ids = set(selected_movie_ids)

        # Ambil detail movie dari movies.csv
        limited_movies = self.movies_df[
            self.movies_df['movieId'].isin(selected_movie_ids)
        ]

        for _, row in limited_movies.iterrows():
            graph.movie_info[int(row['movieId'])] = {
                'title': str(row['title']),
                'genres': str(row['genres'])
            }

        filtered_ratings = self.ratings_df[
            (self.ratings_df['userId'].isin(selected_users)) &
            (self.ratings_df['movieId'].isin(selected_movie_ids))
        ]

        for _, row in filtered_ratings.iterrows():
            graph.add_edge(
                user_id=int(row['userId']),
                movie_id=int(row['movieId']),
                rating=float(row['rating'])
            )

        stats = graph.get_stats()

        print("\n[DataLoader] Graph berhasil dibangun:")
        print(f"Users  : {stats['total_users']}")
        print(f"Movies : {stats['total_movies']}")
        print(f"Edges  : {stats['total_edges']}")
        print(
            f"Avg ratings/user: "
            f"{stats['avg_ratings_per_user']:.1f}"
        )

        return graph
