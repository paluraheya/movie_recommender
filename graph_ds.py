
from collections import defaultdict
import numpy as np


def merge_sort(iterable_or_list, key=None, reverse=False, in_place=False):
    if key is None:
        key = lambda x: x
        
    def _merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = _merge_sort(arr[:mid])
        right = _merge_sort(arr[mid:])
        return _merge(left, right)

    def _merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            val_left = key(left[i])
            val_right = key(right[j])
            if reverse:
                if val_left >= val_right:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            else:
                if val_left <= val_right:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    arr_list = list(iterable_or_list)
    sorted_list = _merge_sort(arr_list)
    
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
        # Arah user ke movie
        self.user_graph: dict = defaultdict(dict)

        # Arah movie ke user
        self.movie_graph: dict = defaultdict(dict)

        # Set Node 
        self.users: set = set()    
        self.movies: set = set()   

        
        # Menyimpan title & genres
        self.movie_info: dict = {}

    # Operasi Dasar Graph

    def add_edge(self, user_id: int, movie_id: int, rating: float) -> None:
        
        self.user_graph[user_id][movie_id] = rating
        self.movie_graph[movie_id][user_id] = rating
        self.users.add(user_id)
        self.movies.add(movie_id)

    def get_user_movies(self, user_id: int) -> dict:
        
        return self.user_graph.get(user_id, {})

    def get_movie_users(self, movie_id: int) -> dict:
        
        return self.movie_graph.get(movie_id, {})

    def has_user(self, user_id: int) -> bool:
        return user_id in self.users

    # Representasi & Statistik

    def get_adjacency_list_str(self, user_id: int, limit: int = 10) -> str:
        
        movies = self.get_user_movies(user_id)
        lines = [f"User {user_id}:  [{len(movies)} film dirating]"]

        # Urutkan berdasarkan rating tertinggi
        sorted_movies = merge_sort(movies.items(), key=lambda x: x[1], reverse=True)

        for movie_id, rating in sorted_movies[:limit]:
            info  = self.movie_info.get(movie_id, {})
            title = info.get('title', f'Movie {movie_id}')
            stars = '★' * int(rating) + '☆' * (5 - int(rating))
            lines.append(f"  ├─ [{movie_id:4}] {title[:32]:<32}  {stars}  ({rating})")

        if len(movies) > limit:
            lines.append(f"  └─ ... dan {len(movies) - limit} film lainnya")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        
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
       
        movies = self.get_user_movies(user_id)
        sorted_movies = merge_sort(movies.items(), key=lambda x: x[1], reverse=True)
        result = []
        for movie_id, rating in sorted_movies[:top_n]:
            title = self.movie_info.get(movie_id, {}).get('title', f'Movie {movie_id}')
            result.append((movie_id, title, rating))
        return result
