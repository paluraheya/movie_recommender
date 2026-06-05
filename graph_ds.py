from collections import defaultdict
import numpy as np

def merge_sort(iterable_or_list, key=None, reverse=False, in_place=False):
    # Fungsi pembantu untuk menentukan kunci pengurutan (default: nilai asli)
    if key is None:
        key = lambda x: x
        
    def _merge_sort(arr):
        # Base case rekursi: list dengan 1 elemen sudah terurut
        if len(arr) <= 1:
            return arr
        # Pecah list menjadi dua bagian (kiri dan kanan)
        mid = len(arr) // 2
        left = _merge_sort(arr[:mid])
        right = _merge_sort(arr[mid:])
        return _merge(left, right) # Gabungkan kembali

    def _merge(left, right):
        result = []
        i = j = 0
        # Iterasi dan gabungkan elemen berdasarkan aturan urutan (ascending/descending)
        while i < len(left) and j < len(right):
            val_left = key(left[i])
            val_right = key(right[j])
            if reverse:
                # Descending: nilai yang lebih besar didahulukan
                if val_left >= val_right:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            else:
                # Ascending: nilai yang lebih kecil didahulukan
                if val_left <= val_right:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
        # Masukkan sisa elemen yang belum terpilih
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    # Konversi iterable ke list dan panggil sort
    arr_list = list(iterable_or_list)
    sorted_list = _merge_sort(arr_list)
    
    # Update list asli jika in_place True
    if in_place:
        if not isinstance(iterable_or_list, list):
            raise TypeError("in_place=True requires a list")
        iterable_or_list[:] = sorted_list
        return None
    else:
        return sorted_list


class BipartiteGraph:
    def __init__(self):
        # Adjacency List Utama 
        # Menyimpan hubungan: User -> Movie (beserta nilai ratingnya)
        self.user_graph: dict = defaultdict(dict)

        # Menyimpan hubungan: Movie -> User (beserta nilai ratingnya)
        self.movie_graph: dict = defaultdict(dict)

        # Himpunan (Set) Node 
        self.users: set = set()    # Kumpulan unik semua ID user
        self.movies: set = set()   # Kumpulan unik semua ID movie
        
        # Menyimpan informasi metadata film: title & genres
        self.movie_info: dict = {}

    # Operasi Dasar Graph

    def add_edge(self, user_id: int, movie_id: int, rating: float) -> None:
        # Menambahkan edge (garis hubungan) dari user ke movie dan sebaliknya
        self.user_graph[user_id][movie_id] = rating
        self.movie_graph[movie_id][user_id] = rating
        
        # Tambahkan node ke dalam himpunan user dan movie
        self.users.add(user_id)
        self.movies.add(movie_id)

    def get_user_movies(self, user_id: int) -> dict:
        # Mengambil daftar film beserta rating yang sudah ditonton oleh seorang user
        return self.user_graph.get(user_id, {})

    def get_movie_users(self, movie_id: int) -> dict:
        # Mengambil daftar user beserta rating yang telah menonton sebuah film
        return self.movie_graph.get(movie_id, {})

    def has_user(self, user_id: int) -> bool:
        # Memeriksa apakah user ada di dalam graph
        return user_id in self.users

    # Representasi & Statistik

    def get_adjacency_list_str(self, user_id: int, limit: int = 10) -> str:
        # Mengembalikan representasi string list adjacency dari sisi user
        movies = self.get_user_movies(user_id)
        lines = [f"User {user_id}:  [{len(movies)} film dirating]"]

        # Urutkan film yang dirating oleh user berdasarkan rating tertinggi
        sorted_movies = merge_sort(movies.items(), key=lambda x: x[1], reverse=True)

        for movie_id, rating in sorted_movies[:limit]:
            info  = self.movie_info.get(movie_id, {})
            title = info.get('title', f'Movie {movie_id}')
            # Membuat format bintang untuk visualisasi rating
            stars = '★' * int(rating) + '☆' * (5 - int(rating))
            lines.append(f"  ├─ [{movie_id:4}] {title[:32]:<32}  {stars}  ({rating})")

        # Tampilkan ellipsis jika film melebihi batas limit
        if len(movies) > limit:
            lines.append(f"  └─ ... dan {len(movies) - limit} film lainnya")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        # Mengambil data statistik tentang jumlah edges, nodes, dan rata-rata rating
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
        # Mendapatkan daftar N film dengan rating tertinggi dari seorang user
        movies = self.get_user_movies(user_id)
        sorted_movies = merge_sort(movies.items(), key=lambda x: x[1], reverse=True)
        result = []
        for movie_id, rating in sorted_movies[:top_n]:
            title = self.movie_info.get(movie_id, {}).get('title', f'Movie {movie_id}')
            result.append((movie_id, title, rating))
        return result
