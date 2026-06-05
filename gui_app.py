import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QScrollArea, QTextEdit,
    QSizePolicy, QGridLayout, QGraphicsDropShadowEffect,
    QProgressBar, QStackedWidget, QMessageBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPainter,
    QBrush, QPen, QPixmap, QIcon
)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


try:
    from graph_ds    import BipartiteGraph, merge_sort
    from algorithms  import BFSTraversal, CollaborativeFilter
    from data_loader import DataLoader
    _DEMO_MODE = False
except ImportError:
    _DEMO_MODE = True
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


if _DEMO_MODE:
    import random, time

    class _DemoGraph:
        def __init__(self):
            self.users = set(range(1, 51))
            self.movie_info = {
                i: {'title': t, 'genres': g}
                for i, (t, g) in enumerate([
                    ("Shawshank Redemption, The (1994)", "Crime | Drama"),
                    ("Schindler's List (1993)", "Drama | War"),
                    ("Aladdin (1992)", "Adventure | Animation | Children"),
                    ("Dark Knight, The (2008)", "Action | Crime | Drama"),
                    ("Forrest Gump (1994)", "Drama | Romance"),
                ], start=1)
            }
            self._ratings = {
                u: {random.randint(1,5): round(random.uniform(3,5),1)
                    for _ in range(random.randint(4,12))}
                for u in self.users
            }
        def get_stats(self):
            edges = sum(len(v) for v in self._ratings.values())
            return {'total_users':50,'total_movies':15,
                    'total_edges':edges,
                    'avg_ratings_per_user':edges/50,
                    'max_ratings_per_user':12}
        def has_user(self, u): return u in self.users
        def get_user_movies(self, u): return self._ratings.get(u, {})
        def get_user_top_movies(self, u, top_n=10):
            movies = self._ratings.get(u, {})
            top = merge_sort(movies.items(), key=lambda x: -x[1])[:top_n]
            return [(mid, self.movie_info.get(mid,{}).get('title',f'Movie {mid}'), r) for mid,r in top]
        def get_adjacency_list_str(self, u, limit=12):
            movies = list(self._ratings.get(u, {}).items())[:limit]
            lines = [f"User {u}:"]
            for mid, r in movies:
                t = self.movie_info.get(mid,{}).get('title','?')[:28]
                lines.append(f"  ├── Movie {mid:>3}  [{r:.1f}★]  {t}")
            return "\n".join(lines)

    class _DemoBFS:
        def __init__(self, g): self.g = g
        def find_similar_users(self, uid, max_depth=2):
            others = [u for u in self.g.users if u != uid]
            samp = random.sample(others, min(15, len(others)))
            return {u: random.uniform(0.2,0.98) for u in samp}, set(samp)
        def get_candidate_movies(self, uid, sim_dict):
            watched = set(self.g.get_user_movies(uid).keys())
            cands = {}
            for su in list(sim_dict.keys())[:5]:
                for mid, r in self.g.get_user_movies(su).items():
                    if mid not in watched:
                        cands[mid] = cands.get(mid, 0) + r * sim_dict[su]
            return cands

    class _DemoCF:
        def __init__(self, g): self.g = g
        def get_top_similar_users(self, uid, candidates, top_k=10):
            return merge_sort(
                [(u, round(random.uniform(0.15, 0.52), 4)) for u in list(candidates)[:top_k]],
                key=lambda x: -x[1]
            )
        def get_recommendations(self, uid, sim_scores, candidates, top_n=5):
            items = merge_sort(candidates.items(), key=lambda x: -x[1])[:top_n]
            res = []
            for mid, score in items:
                info = self.g.movie_info.get(mid, {})
                contributors = []
                for _ in range(random.randint(2, 5)):
                    sim = random.uniform(0.15, 0.52)
                    rating = random.randint(3, 5)
                    contributors.append({
                        'user_id': random.randint(1, 50),
                        'rating': rating,
                        'similarity': round(sim, 4),
                        'contribution': round(sim * rating, 4)
                    })
                contributors = merge_sort(contributors, key=lambda c: c['contribution'], reverse=True)
                res.append({
                    'movie_id': mid,
                    'title': info.get('title', f'Movie {mid}'),
                    'genres': info.get('genres', 'Unknown'),
                    'score': 5.0,
                    'rated_by': len(contributors),
                    'contributors': contributors
                })
            return res

    class _DemoLoader:
        def load_data(self): time.sleep(0.3)
        def build_graph(self, g, **kw): time.sleep(0.3)

    BipartiteGraph       = _DemoGraph
    BFSTraversal         = _DemoBFS
    CollaborativeFilter  = _DemoCF
    DataLoader           = _DemoLoader


# ============================================================================
# THEME PALETTES
# ============================================================================
THEME_LIGHT = {
    "bg_app":       "#F8FAFC",   # Slate 50
    "bg_panel":     "#FFFFFF",   # White
    "text_primary": "#0F172A",   # Slate 900
    "text_second":  "#64748B",   # Slate 500
    "text_light":   "#94A3B8",   # Slate 400
    "primary_blue": "#3B82F6",   # Blue 500
    "bg_blue":      "#EFF6FF",   # Blue 50
    "border":       "#E2E8F0",   # Slate 200
    "danger":       "#EF4444",   # Red 500
    "warning":      "#F59E0B",   # Amber 500
    "is_dark":      False
}

THEME_DARK = {
    "bg_app":       "#0F172A",   # Slate 900
    "bg_panel":     "#1E293B",   # Slate 800
    "text_primary": "#F8FAFC",   # Slate 50
    "text_second":  "#94A3B8",   # Slate 400
    "text_light":   "#64748B",   # Slate 500
    "primary_blue": "#3B82F6",   # Blue 500
    "bg_blue":      "#172554",   # Blue 950
    "border":       "#334155",   # Slate 700
    "danger":       "#EF4444",   # Red 500
    "warning":      "#F59E0B",   # Amber 500
    "is_dark":      True
}

# GLOBAL STYLESHEET GENERATOR
def get_qss(P):
    return f"""
    QMainWindow, QWidget {{
        background: {P['bg_app']};
        color: {P['text_primary']};
        font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial;
        font-size: 14px;
    }}
    
    QFrame[cssClass="panel"] {{
        background: {P['bg_panel']};
        border: 1px solid {P['border']};
        border-radius: 16px;
    }}
    QFrame[cssClass="card"] {{
        background: {P['bg_panel']};
        border: 1px solid {P['border']};
        border-radius: 12px;
    }}
    QFrame[cssClass="sidebar"] {{
        background: {P['bg_panel']};
        border-right: 1px solid {P['border']};
    }}
    QFrame[cssClass="header"] {{
        background: {P['bg_panel']};
        border-bottom: 1px solid {P['border']};
    }}
    QFrame[cssClass="watched_row"] {{
        border: none; 
        border-bottom: 1px solid {P['border']};
    }}
    QFrame[cssClass="sim_card"] {{
        background: {P['bg_app']};
        border-radius: 12px;
        border: none;
    }}
    QFrame[cssClass="score_box"] {{
        background: {P['bg_blue']};
        border-radius: 8px;
        border: none;
    }}
    
    /* ScrollBars */
    QScrollBar:vertical {{ background: transparent; width: 6px; border: none; }}
    QScrollBar::handle:vertical {{ background: {P['border']}; border-radius: 3px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {P['text_light']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

    /* Inputs */
    QLineEdit {{
        background: {P['bg_panel']};
        border: 1px solid {P['border']};
        border-radius: 6px;
        color: {P['text_primary']};
        padding: 8px 12px;
        font-size: 14px;
    }}
    QLineEdit:focus {{ border: 1.5px solid {P['primary_blue']}; }}
    
    /* TextEdit */
    QTextEdit {{
        background: {P['bg_panel']};
        border: 1px solid {P['border']};
        border-radius: 8px;
        color: {P['text_primary']};
        font-family: "Cascadia Code", "Consolas", monospace;
        font-size: 14px;
        padding: 14px;
    }}

    /* ProgressBar */
    QProgressBar {{
        background: {P['border']};
        border: none;
        border-radius: 3px;
        height: 6px;
    }}
    QProgressBar::chunk {{
        background: {P['primary_blue']};
        border-radius: 3px;
    }}
    
    /* Buttons */
    QPushButton[cssClass="btn_primary"] {{
        background: {P['primary_blue']};
        color: white;
        font-weight: 600;
        font-size: 15px;
        border-radius: 8px;
        border: none;
    }}
    QPushButton[cssClass="btn_primary"]:hover {{ background: #2563EB; }}
    QPushButton[cssClass="btn_primary"]:disabled {{ background: {P['text_light']}; }}
    
    QPushButton[cssClass="btn_nav"] {{
        background: transparent;
        color: {P['text_second']};
        border: none;
        border-radius: 8px;
        text-align: left;
        padding-left: 16px;
        font-size: 14px;
        font-weight: 500;
    }}
    QPushButton[cssClass="btn_nav"]:hover {{
        background: {P['bg_app']};
        color: {P['text_primary']};
    }}
    QPushButton[cssClass="btn_nav_active"] {{
        background: {P['bg_blue']};
        color: {P['primary_blue']};
        border: none;
        border-radius: 8px;
        text-align: left;
        padding-left: 16px;
        font-size: 14px;
        font-weight: 600;
    }}
    
    QPushButton[cssClass="btn_danger"] {{
        background: transparent;
        color: {P['danger']};
        font-size: 14px;
        font-weight: 600;
        border: none;
        text-align: left;
        padding-left: 10px;
    }}
    QPushButton[cssClass="btn_danger"]:hover {{ color: #B91C1C; }}
    
    /* Labels */
    QLabel[cssClass="text_primary"] {{ color: {P['text_primary']}; border: none; background: transparent; }}
    QLabel[cssClass="text_second"]  {{ color: {P['text_second']}; border: none; background: transparent; }}
    QLabel[cssClass="text_light"]   {{ color: {P['text_light']}; border: none; background: transparent; }}
    QLabel[cssClass="text_blue"]    {{ color: {P['primary_blue']}; border: none; background: transparent; }}
    QLabel[cssClass="text_danger"]  {{ color: {P['danger']}; border: none; background: transparent; }}
    
    QLabel[cssClass="icon_box"] {{
        background: {P['bg_app']};
        border: none;
        border-radius: 10px;
        color: {P['text_light']};
    }}
    QLabel[cssClass="logo_icon"] {{
        background: {P['primary_blue']};
        color: white;
        border-radius: 12px;
        border: none;
    }}
    """

def make_shadow(color="#000000", blur=15, x_off=0, y_off=2, alpha=20):
    eff = QGraphicsDropShadowEffect()
    c = QColor(color)
    c.setAlpha(alpha)
    eff.setColor(c)
    eff.setBlurRadius(blur)
    eff.setOffset(x_off, y_off)
    return eff

# CUSTOM WIDGETS
class NavButton(QPushButton):
    def __init__(self, text, icon_path="", parent=None):
        super().__init__(parent)
        if icon_path:
            self.setIcon(QIcon(icon_path))
        self.setText(f" {text}")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(40)
        self.setProperty("cssClass", "btn_nav")
    
    def set_active(self, active=True):
        self.setProperty("cssClass", "btn_nav_active" if active else "btn_nav")
        self.style().unpolish(self)
        self.style().polish(self)

class RecCard(QFrame):
    def __init__(self, rec_data, parent=None):
        super().__init__(parent)
        self.rec_data = rec_data
        self.on_click_callback = None
        self.setCursor(Qt.PointingHandCursor)
        title = rec_data.get('title', '')
        genres = rec_data.get('genres', '')
        score = rec_data.get('score', 0.0)
        self.setFixedHeight(100)
        self.setProperty("cssClass", "card")
        self.setGraphicsEffect(make_shadow(blur=10, alpha=15))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(16)

        icon_box = QLabel()
        icon_box.setPixmap(QPixmap("icons/movie.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_box.setFixedSize(50, 50)
        icon_box.setAlignment(Qt.AlignCenter)
        icon_box.setProperty("cssClass", "icon_box")
        lay.addWidget(icon_box)

        text_lay = QVBoxLayout()
        text_lay.setSpacing(2)
        text_lay.setAlignment(Qt.AlignVCenter)
        
        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        t_lbl.setProperty("cssClass", "text_primary")
        text_lay.addWidget(t_lbl)
        
        g_lbl = QLabel(genres.replace('|', ' • '))
        g_lbl.setFont(QFont("Segoe UI", 11))
        g_lbl.setProperty("cssClass", "text_light")
        text_lay.addWidget(g_lbl)
        
        lay.addLayout(text_lay)
        lay.addStretch()

        score_box = QFrame()
        score_box.setFixedSize(60, 44)
        score_box.setProperty("cssClass", "score_box")
        sb_lay = QVBoxLayout(score_box)
        sb_lay.setContentsMargins(0, 8, 0, 8)
        sb_lay.setSpacing(0)
        
        s_lbl1 = QLabel(f"{score:.1f}")
        s_lbl1.setFont(QFont("Segoe UI", 13, QFont.Bold))
        s_lbl1.setProperty("cssClass", "text_blue")
        s_lbl1.setAlignment(Qt.AlignCenter)
        sb_lay.addWidget(s_lbl1)
        
        s_lbl2 = QLabel("SCORE")
        s_lbl2.setFont(QFont("Segoe UI", 10, QFont.Bold))
        s_lbl2.setProperty("cssClass", "text_blue")
        s_lbl2.setAlignment(Qt.AlignCenter)
        sb_lay.addWidget(s_lbl2)

        lay.addWidget(score_box)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.on_click_callback:
            self.on_click_callback(self.rec_data)

class WatchedRow(QFrame):
    def __init__(self, title, rating, parent=None):
        super().__init__(parent)
        self.title_text = title.lower()
        self.setFixedHeight(60)
        self.setProperty("cssClass", "watched_row")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        
        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Segoe UI", 12, QFont.Medium))
        t_lbl.setProperty("cssClass", "text_primary")
        lay.addWidget(t_lbl)
        lay.addStretch()
        
        r_lbl = QLabel(f"{int(rating)} ")
        r_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
        r_lbl.setProperty("cssClass", "text_primary")
        lay.addWidget(r_lbl)
        
        star = QLabel()
        star.setPixmap(QPixmap("icons/star.png").scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(star)

class SimUserRow(QFrame):
    def __init__(self, uid, sim_pct, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setStyleSheet("background: transparent; border: none;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        
        u_lbl = QLabel(f"User {uid}")
        u_lbl.setFixedWidth(50)
        u_lbl.setFont(QFont("Segoe UI", 12))
        u_lbl.setProperty("cssClass", "text_second")
        lay.addWidget(u_lbl)
        
        prog = QProgressBar()
        prog.setFixedHeight(6)
        prog.setRange(0, 100)
        prog.setValue(int(sim_pct * 100))
        prog.setTextVisible(False)
        lay.addWidget(prog)
        
        p_lbl = QLabel(f"{int(sim_pct*100)}%")
        p_lbl.setFixedWidth(35)
        p_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        p_lbl.setFont(QFont("Segoe UI", 11))
        p_lbl.setProperty("cssClass", "text_light")
        lay.addWidget(p_lbl)

# DATA LOADER & THREADS
class LoaderThread(QThread):
    done    = pyqtSignal(object)
    error   = pyqtSignal(str)
    status  = pyqtSignal(str)

    def run(self):
        try:
            self.status.emit("Loading dataset...")
            loader = DataLoader()
            loader.load_data()
            self.status.emit("Building Graph...")
            graph = BipartiteGraph()
            loader.build_graph(graph, max_users=None,max_movies=None)
            bfs = BFSTraversal(graph)
            cf  = CollaborativeFilter(graph)
            self.done.emit((graph, bfs, cf))
        except Exception as e:
            self.error.emit(str(e))

class RecommendThread(QThread):
    done   = pyqtSignal(object)
    error  = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, graph, bfs, cf, uid):
        super().__init__()
        self.graph, self.bfs, self.cf, self.uid = graph, bfs, cf, uid

    def run(self):
        try:
            uid = self.uid
            self.status.emit("Running BFS traversal...")
            sim_raw, visited = self.bfs.find_similar_users(uid, max_depth=2)
            self.status.emit("Calculating similarity...")
            sim_scores = self.cf.get_top_similar_users(uid, sim_raw.keys(), top_k=10)
            self.status.emit("Finding candidates...")
            candidates = self.bfs.get_candidate_movies(uid, dict(sim_scores))
            self.status.emit("Scoring...")
            recs = self.cf.get_recommendations(uid, sim_scores, candidates, top_n=5)
            self.done.emit({
                'uid': uid, 'sim_scores': sim_scores,
                'recs': recs, 'visited': visited,
                'sim_raw': sim_raw, 'candidates': candidates
            })
        except Exception as e:
            self.error.emit(str(e))

# MATPLOTLIB GRAPH CANVAS
class GraphCanvas(FigureCanvas):
    """
    Visualisasi bipartite graph secara bertahap (step-by-step).
    Navigasi menggunakan tombol panah kiri/kanan di luar canvas ini.
    """

    STEP_TITLES = [
        ("Step 1 / 4", "Target User — Menampilkan film yang sudah ditonton"),
        ("Step 2 / 4", "Similar Users — Menemukan user dengan tontonan yang sama (Co-rated)"),
        ("Step 3 / 4", "Candidate Movies — Mengumpulkan film rekomendasi dari Similar Users"),
        ("Step 4 / 4", "Final Output — Film dengan skor tertinggi direkomendasikan"),
    ]

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._step       = 0
        self._graph_data = None
        self._P          = None

    # ── Public API ───────────────────────────────────────────────────────────

    def _draw_empty(self, P):
        self._P = P
        self.ax.clear()
        self.ax.set_facecolor(P['bg_panel'])
        self.fig.patch.set_facecolor(P['bg_panel'])
        self.ax.text(0.5, 0.5, "Generate recommendations to view graph.",
                     ha='center', va='center', fontsize=12, color=P['text_light'])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.draw()

    def load_graph(self, graph, target_uid, sim_scores, recs, P, candidates=None, sim_raw=None):
        """Simpan semua data, reset ke step 0, lalu render."""
        self._P    = P
        self._step = 0

        self.step_top_movies = []
        sim_users = [u for u, _ in sim_scores[:5]]
        
        from algorithms import CollaborativeFilter
        cf = CollaborativeFilter(graph)
        
        for s in range(6):
            if s == 0 or not candidates:
                self.step_top_movies.append([])
                continue
            if s == 1: active_sims = sim_users[:1]
            elif s == 2: active_sims = sim_users[:2]
            elif s == 3: active_sims = sim_users[:3]
            else: active_sims = sim_users[:5]
            
            if s == 5:
                self.step_top_movies.append(recs)
                continue
                
            active_sim_scores = [(u, sim) for u, sim in sim_scores if u in active_sims]
            step_recs = cf.get_recommendations(target_uid, active_sim_scores, candidates, top_n=8)
            self.step_top_movies.append(step_recs)

        rec_ids  = [r['movie_id'] for r in recs]
        rec_set  = set(rec_ids)
        rec_info = {r['movie_id']: r for r in recs}

        target_watched_list = list(graph.get_user_movies(target_uid).keys())
        target_watched_set = set(target_watched_list)
        
        # Count how many similar users watched each target movie
        movie_sim_counts = {}
        for m in target_watched_set:
            if m in rec_set:
                continue
            count = sum(1 for u in sim_users if m in graph.get_user_movies(u))
            if count > 0:
                movie_sim_counts[m] = count
                
        # Sort by most shared, then take top 5
        sorted_shared = sorted(movie_sim_counts.keys(), key=lambda x: movie_sim_counts[x], reverse=True)
        ctx_ids = sorted_shared[:5]
        
        # Pad with other watched movies if we have less than 5
        if len(ctx_ids) < 5:
            for m in target_watched_list:
                if m not in rec_set and m not in ctx_ids:
                    ctx_ids.append(m)
                if len(ctx_ids) == 5:
                    break
        
        all_movies = set(rec_ids + ctx_ids)
        if len(self.step_top_movies) > 4:
            for r in self.step_top_movies[4]:
                all_movies.add(r['movie_id'])
                if r['movie_id'] not in rec_info:
                    rec_info[r['movie_id']] = r
                    
        movie_list = list(all_movies)

        sim_users = [u for u, _ in sim_scores[:5]]

        PAD = 0.12
        def _x(i, n):
            return (PAD + (1 - 2*PAD) / max(n - 1, 1) * i) if n > 1 else 0.5

        n_u = 1 + len(sim_users)
        n_m = len(movie_list)
        all_users = [target_uid] + sim_users
        upos = {u: (_x(i, n_u), 0.90) for i, u in enumerate(all_users)}
        mpos = {m: (_x(i, n_m), 0.12) for i, m in enumerate(movie_list)}

        self._graph_data = dict(
            graph        = graph,
            target_uid   = target_uid,
            sim_users    = sim_users,
            sim_scores   = dict(sim_scores),
            rec_set      = rec_set,
            rec_info     = rec_info,
            movie_list   = movie_list,
            ctx_ids      = set(ctx_ids),
            upos         = upos,
            mpos         = mpos,
            sim_raw      = sim_raw,
        )
        self._render_step()

    def go_next(self):
        if self._graph_data and self._step < len(self.STEP_TITLES) - 1:
            self._step += 1
            self._render_step()

    def go_prev(self):
        if self._graph_data and self._step > 0:
            self._step -= 1
            self._render_step()

    @property
    def current_step(self):
        return self._step

    @property
    def total_steps(self):
        return len(self.STEP_TITLES)

    # ── Internal render ──────────────────────────────────────────────────────

    def _render_step(self):
        if not self._graph_data:
            return
        P  = self._P
        d  = self._graph_data
        s  = self._step

        C_TARGET  = '#da42f5'
        C_SIMILAR = '#60A5FA' if not P['is_dark'] else '#3B82F6'
        C_REC     = P['warning']
        C_WATCHED = P['text_second']
        C_EDGE_W  = P['text_second']
        C_EDGE_R  = P['primary_blue']
        C_LABEL   = P['text_primary']
        C_DIM     = P['border']

        self.ax.clear()
        self.ax.set_facecolor(P['bg_panel'])
        self.fig.patch.set_facecolor(P['bg_panel'])

        graph      = d['graph']
        target_uid = d['target_uid']
        sim_users  = d['sim_users']
        sim_scores = d['sim_scores']
        rec_set    = d['rec_set']
        rec_info   = d['rec_info']
        movie_list = d['movie_list']
        ctx_ids    = d['ctx_ids'] # Target's watched movies (subset shown in graph)
        upos       = d['upos']
        mpos       = d['mpos']

        # Determine active elements based on the 4 steps
        if s == 0:
            active_users = [target_uid]
            active_movies = ctx_ids
        elif s == 1:
            active_users = [target_uid] + sim_users
            active_movies = ctx_ids
        elif s == 2:
            active_users = sim_users
            active_movies = set(movie_list) - ctx_ids
        else:
            active_users = sim_users
            active_movies = rec_set

        # ── Edges ─────────────────────────────────────────────────────────
        for uid in [target_uid] + sim_users:
            ux, uy = upos[uid]
            
            # Hide edges for inactive users to avoid clutter
            if uid not in active_users:
                continue

            for mid, r in graph.get_user_movies(uid).items():
                if mid not in mpos:
                    continue
                if mid not in active_movies:
                    continue

                mx, my = mpos[mid]
                is_rec = mid in rec_set
                
                # Edge Highlight Logic
                if s == 3 and uid in sim_users and is_rec:
                    lw, color, alpha, zo = 1.5 + (r/5)*2.0, C_EDGE_R, 0.85, 4
                elif s == 1 and uid in sim_users and mid in ctx_ids:
                    lw, color, alpha, zo = 1.5, C_SIMILAR, 0.6, 3
                elif s == 0 and uid == target_uid:
                    lw, color, alpha, zo = 1.5, C_TARGET, 0.6, 3
                elif s == 2 and uid in sim_users:
                    lw, color, alpha, zo = 1.0, C_EDGE_W, 0.4, 2
                else:
                    lw, color, alpha, zo = 0.5, C_DIM, 0.1, 1

                if s == 1 and uid in sim_users:
                    start_pos = (mx, my)
                    end_pos = (ux, uy)
                else:
                    start_pos = (ux, uy)
                    end_pos = (mx, my)

                if alpha > 0.15:
                    self.ax.annotate('', xy=end_pos, xytext=start_pos,
                                     arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha,
                                                     shrinkA=15, shrinkB=15),
                                     zorder=zo)

        # ── Node film ─────────────────────────────────────────────────────
        for mid in movie_list:
            x, y   = mpos[mid]
            info   = graph.movie_info.get(mid, {})
            title  = info.get('title', f'Movie {mid}')
            is_rec = mid in rec_set
            short  = (title[:11] + '…') if len(title) > 11 else title

            # Check if node is active
            if s == 0 or s == 1:
                is_active = mid in ctx_ids
            elif s == 2:
                is_active = mid not in ctx_ids
            elif s == 3:
                is_active = is_rec
            else:
                is_active = True

            if is_active:
                if s == 3 and is_rec:
                    c, sz, alpha = C_REC, 320, 1.0
                    score = rec_info[mid]['score']
                    self.ax.text(x, y + 0.07, f'★{score:.2f}',
                                 fontsize=7, color=C_REC, ha='center',
                                 va='bottom', fontweight='bold', zorder=8)
                elif s == 2:
                    c, sz, alpha = P['primary_blue'], 180, 0.8
                else:
                    c = C_WATCHED if mid in ctx_ids else C_DIM
                    sz = 160
                    alpha = 0.9
            else:
                c, sz, alpha = C_DIM, 100, 0.2

            self.ax.scatter(x, y, s=sz, c=c, alpha=alpha,
                            marker='o' if (is_rec and s == 3) else 's',
                            zorder=6, edgecolors=P['bg_panel'], linewidths=1)
            
            fa = 1.0 if is_active else 0.2
            self.ax.text(x, y - 0.07, short, fontsize=7, color=C_LABEL,
                         ha='right', va='top', alpha=fa, rotation=45,
                         fontweight='bold' if (is_active and s==3) else 'normal', zorder=7)

        # ── Node user ─────────────────────────────────────────────────────
        for uid in [target_uid] + sim_users:
            x, y   = upos[uid]
            is_t   = uid == target_uid
            
            if s == 0:
                is_active = is_t
            elif s == 1:
                is_active = True
            elif s == 2:
                is_active = not is_t
            elif s == 3:
                is_active = not is_t
            else:
                is_active = True

            if is_t:
                c, sz, alpha = C_TARGET, 420, 1.0 if is_active else 0.2
            else:
                c, sz, alpha = C_SIMILAR, 220, 1.0 if is_active else 0.2

            self.ax.scatter(x, y, s=sz, c=c, alpha=alpha, zorder=6,
                            edgecolors=P['bg_panel'], linewidths=1.5)

            lbl = f"U{uid}"
            if not is_t:
                lbl = f"U{uid} | {sim_scores.get(uid, 0):.2f}"
            fa = 1.0 if is_active else 0.2
            self.ax.text(x, y + 0.06, lbl, fontsize=8, color=C_LABEL,
                         ha='center', va='bottom', alpha=fa,
                         fontweight='bold' if is_active else 'normal',
                         zorder=7)

        # ── Label sisi USER / FILM ────────────────────────────────────────
        self.ax.text(0.01, 0.90, 'USER', fontsize=9, color=C_LABEL,
                     va='center', ha='left', alpha=0.45,
                     fontweight='bold', transform=self.ax.transAxes)
        self.ax.text(0.01, 0.12, 'FILM', fontsize=9, color=C_LABEL,
                     va='center', ha='left', alpha=0.45,
                     fontweight='bold', transform=self.ax.transAxes)

        # ── Garis pemisah bipartite ────────────────────────────────────────
        self.ax.axhline(y=0.52, color=C_DIM, linewidth=0.6,
                        linestyle='--', alpha=0.4, zorder=0)

        # ── Legend ────────────────────────────────────────────────────────
        handles = [
            mpatches.Patch(color=C_TARGET,  label='Target User'),
            mpatches.Patch(color=C_SIMILAR, label='Similar User'),
            mpatches.Patch(color=C_REC,     label='Rekomendasi'),
            mpatches.Patch(color=C_WATCHED, label='Sudah Ditonton'),
        ]
        leg = self.ax.legend(handles=handles, loc='upper center',
                             bbox_to_anchor=(0.5, 1.08), ncol=4,
                             frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(C_LABEL)

        # ── Deskripsi step di bawah ───────────────────────────────────────
        _, step_desc = self.STEP_TITLES[s]
        self.ax.text(0.5, 0.01, step_desc, fontsize=9, color=C_LABEL,
                     ha='center', va='bottom', alpha=0.7,
                     transform=self.ax.transAxes, style='italic')

        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.25, 1.15)
        self.ax.axis('off')
        self.fig.tight_layout(pad=0.5)
        self.draw()

# MAIN APPLICATION
class MovieRecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Abiyu TV — Movie Recommender")
        self.setMinimumSize(1024, 720) # Responsive
        self.resize(1280, 800)
        
        self.P = THEME_DARK
        self.setStyleSheet(get_qss(self.P))

        self.graph = None
        self.bfs   = None
        self.cf    = None
        self.is_loaded = False
        self.cur_uid   = None
        self.last_result = None

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self._build_login_screen()
        self._build_main_dashboard()

        self.stacked_widget.setCurrentWidget(self.login_widget)

        self._prog_timer = QTimer()
        self._prog_timer.timeout.connect(self._tick_progress)
        self._prog_val = 0

        QTimer.singleShot(300, self._load_data)

    def _apply_theme(self):
        
        self.setStyleSheet(get_qss(self.P))
        # Reforce rendering on dynamic properties
        for widget in self.findChildren(QWidget):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        
        # Redraw graph canvas properly
        if self.cur_uid is not None and self.is_loaded and self.last_result:
            res = self.last_result
            self.graph_canvas.load_graph(self.graph, res['uid'], res['sim_scores'], res['recs'], self.P, res.get('candidates'), res.get('sim_raw'))
            self._update_graph_nav()
        else:
            self.graph_canvas._draw_empty(self.P)
            self._update_graph_nav()

    def _toggle_theme(self):
        if self.P['is_dark']:
            self.P = THEME_LIGHT
            self.btn_theme.setText(" Dark Mode")
            self.btn_theme.setIcon(QIcon("icons/night-mode.png"))
        else:
            self.P = THEME_DARK
            self.btn_theme.setText(" Light Mode")
            self.btn_theme.setIcon(QIcon("icons/light-mode.png"))
        self._apply_theme()

    # Login Screen 
    def _build_login_screen(self):
        self.login_widget = QWidget()
        lay = QVBoxLayout(self.login_widget)
        lay.setAlignment(Qt.AlignCenter)
        
        card = QFrame()
        card.setFixedSize(600, 650)
        card.setProperty("cssClass", "panel")
        card.setGraphicsEffect(make_shadow(blur=25, alpha=10, y_off=5))
        
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(60, 70, 60, 70)
        c_lay.setSpacing(25)
        
        # Logo
        logo_lay = QHBoxLayout()
        logo_icon = QLabel()
        logo_icon.setPixmap(QPixmap("icons/logo.png").scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_icon.setFixedSize(80, 80)
        logo_icon.setAlignment(Qt.AlignCenter)
        logo_icon.setProperty("cssClass", "logo_icon")
        logo_lay.addStretch()
        logo_lay.addWidget(logo_icon)
        logo_lay.addStretch()
        c_lay.addLayout(logo_lay)
        
        title = QLabel("Abiyu TV")
        title.setFont(QFont("Monaco", 30, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setProperty("cssClass", "text_primary")
        c_lay.addWidget(title)
        
        subtitle = QLabel("Movie Recommendation System")
        subtitle.setFont(QFont("Segoe UI", 18))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setProperty("cssClass", "text_second")
        c_lay.addWidget(subtitle)
        
        c_lay.addSpacing(20)
        
        self.login_status = QLabel("Loading dataset...")
        self.login_status.setFont(QFont("Segoe UI", 16))
        self.login_status.setAlignment(Qt.AlignCenter)
        self.login_status.setProperty("cssClass", "text_second")
        c_lay.addWidget(self.login_status)
        
        self.login_progress = QProgressBar()
        self.login_progress.setFixedHeight(8)
        self.login_progress.setRange(0, 100)
        self.login_progress.setTextVisible(False)
        c_lay.addWidget(self.login_progress)
        
        c_lay.addSpacing(25)
        
        hint_lbl = QLabel("Tersedia User ID: 1 - 611")
        hint_lbl.setFont(QFont("Segoe UI", 11))
        hint_lbl.setAlignment(Qt.AlignCenter)
        hint_lbl.setProperty("cssClass", "text_second")
        c_lay.addWidget(hint_lbl)
        
        self.login_inp_uid = QLineEdit()
        self.login_inp_uid.setPlaceholderText("Enter User ID (e.g. 1)")
        self.login_inp_uid.setFont(QFont("Segoe UI", 18))
        self.login_inp_uid.setAlignment(Qt.AlignCenter)
        self.login_inp_uid.setFixedHeight(60)
        self.login_inp_uid.setEnabled(False)
        self.login_inp_uid.returnPressed.connect(self._on_login_generate)
        c_lay.addWidget(self.login_inp_uid)
        
        self.login_btn = QPushButton("Login & Generate")
        self.login_btn.setFixedHeight(60)
        self.login_btn.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.login_btn.setCursor(Qt.PointingHandCursor)
        self.login_btn.setEnabled(False)
        self.login_btn.setProperty("cssClass", "btn_primary")
        self.login_btn.clicked.connect(self._on_login_generate)
        c_lay.addWidget(self.login_btn)
        
        lay.addWidget(card)
        self.stacked_widget.addWidget(self.login_widget)

    # Main Dashboard 
    def _build_main_dashboard(self):
        self.main_widget = QWidget()
        lay = QHBoxLayout(self.main_widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        
        self._build_sidebar(lay)
        self._build_content(lay)
        
        self.stacked_widget.addWidget(self.main_widget)

    def _build_sidebar(self, parent_lay):
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setProperty("cssClass", "sidebar")
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(20, 30, 20, 30)
        lay.setSpacing(10)
        
        # Logo
        logo_lay = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(QPixmap("icons/logo.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon.setFixedSize(50, 50)
        icon.setAlignment(Qt.AlignCenter)
        icon.setProperty("cssClass", "logo_icon")
        logo_lay.addWidget(icon)
        
        title = QLabel("Abiyu TV")
        title.setFont(QFont("Monaco", 45, QFont.Bold))
        title.setProperty("cssClass", "text_primary")
        logo_lay.addWidget(title)
        logo_lay.addStretch()
        lay.addLayout(logo_lay)
        
        lay.addSpacing(30)
        
        # Nav Buttons
        self.nav_btns = []
        self.btn_rec = NavButton("Recommendations", "icons/like.png")
        self.btn_rec.set_active(True)
        self.btn_rec.clicked.connect(lambda: self._switch_tab(0))
        lay.addWidget(self.btn_rec)
        self.nav_btns.append(self.btn_rec)
        
        self.btn_graph = NavButton("Graph Visualization", "icons/data-analytics.png")
        self.btn_graph.clicked.connect(lambda: self._switch_tab(1))
        lay.addWidget(self.btn_graph)
        self.nav_btns.append(self.btn_graph)
        
        self.btn_adj = NavButton("Adjacency List", "icons/list-text.png")
        self.btn_adj.clicked.connect(lambda: self._switch_tab(2))
        lay.addWidget(self.btn_adj)
        self.nav_btns.append(self.btn_adj)
        
        lay.addSpacing(40)
        
        # Similar Users Section
        sim_card = QFrame()
        sim_card.setProperty("cssClass", "sim_card")
        sim_lay = QVBoxLayout(sim_card)
        sim_lay.setContentsMargins(16, 16, 16, 16)
        
        hdr_lay = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(QPixmap("icons/user.png").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        ic.setStyleSheet("border: none;")
        hdr_lay.addWidget(ic)
        
        vh = QVBoxLayout()
        vh.setSpacing(0)
        lh = QLabel("SIMILAR USERS")
        lh.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lh.setProperty("cssClass", "text_light")
        vh.addWidget(lh)
        self.lbl_profile = QLabel("User Profiling")
        self.lbl_profile.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_profile.setProperty("cssClass", "text_primary")
        vh.addWidget(self.lbl_profile)
        hdr_lay.addLayout(vh)
        hdr_lay.addStretch()
        sim_lay.addLayout(hdr_lay)
        
        sim_lay.addSpacing(10)
        
        self.sim_rows_lay = QVBoxLayout()
        self.sim_rows_lay.setSpacing(6)
        sim_lay.addLayout(self.sim_rows_lay)
        
        lay.addWidget(sim_card)
        
        lay.addStretch()
        
        # Theme Toggle
        self.btn_theme = QPushButton(" Dark Mode")
        self.btn_theme.setIcon(QIcon("icons/night-mode.png"))
        self.btn_theme.setCursor(Qt.PointingHandCursor)
        self.btn_theme.setProperty("cssClass", "btn_nav")
        self.btn_theme.clicked.connect(self._toggle_theme)
        lay.addWidget(self.btn_theme)
        
        # Reset Button (Logout / Switch User)
        self.btn_reset = QPushButton(" Switch User")
        self.btn_reset.setIcon(QIcon("icons/logout.png"))
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setProperty("cssClass", "btn_danger")
        self.btn_reset.clicked.connect(self._on_reset)
        lay.addWidget(self.btn_reset)
        
        parent_lay.addWidget(sidebar)

    def _build_content(self, parent_lay):
        content_wrapper = QWidget()
        lay = QVBoxLayout(content_wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setFixedHeight(60)
        header.setProperty("cssClass", "header")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(30, 0, 30, 0)
        
        self.lbl_breadcrumb = QLabel("DASHBOARD  >  Target User")
        self.lbl_breadcrumb.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_breadcrumb.setProperty("cssClass", "text_second")
        h_lay.addWidget(self.lbl_breadcrumb)
        h_lay.addStretch()
        
        # Search Bar (Hidden per user request, avoids "dead widget" confusion)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search dataset...")
        self.search_bar.addAction(QIcon("icons/search.png"), QLineEdit.LeadingPosition)
        self.search_bar.setFixedWidth(250)
        self.search_bar.setVisible(False) 
        h_lay.addWidget(self.search_bar)
        
        lay.addWidget(header)
        
        # Stacked Tabs
        self.tabs = QStackedWidget()
        self.tabs.setStyleSheet("background: transparent;")
        
        self._build_tab_recs()
        self._build_tab_graph()
        self._build_tab_adj()
        
        lay.addWidget(self.tabs)
        parent_lay.addWidget(content_wrapper)

    def _build_tab_recs(self):
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.setSpacing(30)
        
        # Left: Top Recs
        left_w = QWidget()
        l_lay = QVBoxLayout(left_w)
        l_lay.setContentsMargins(0, 0, 0, 0)
        
        h1_lay = QHBoxLayout()
        h1_ic = QLabel()
        h1_ic.setPixmap(QPixmap("icons/like.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h1 = QLabel("Top 5 Recommendations")
        h1.setFont(QFont("Segoe UI", 18, QFont.Bold))
        h1.setProperty("cssClass", "text_primary")
        h1_lay.addWidget(h1_ic)
        h1_lay.addWidget(h1)
        h1_lay.addStretch()
        l_lay.addLayout(h1_lay)
        l_lay.addSpacing(10)
        
        scr1 = QScrollArea()
        scr1.setWidgetResizable(True)
        scr1.setFrameShape(QFrame.NoFrame)
        self.rec_container = QWidget()
        self.rec_lay = QVBoxLayout(self.rec_container)
        self.rec_lay.setContentsMargins(0, 0, 0, 0)
        self.rec_lay.setSpacing(16)
        self.rec_lay.addStretch()
        scr1.setWidget(self.rec_container)
        l_lay.addWidget(scr1)
        
        lay.addWidget(left_w, 3)
        
        # Right: Watched Movies
        right_w = QWidget()
        r_lay = QVBoxLayout(right_w)
        r_lay.setContentsMargins(0, 0, 0, 0)
        
        h2_lay = QHBoxLayout()
        h2_ic = QLabel()
        h2_ic.setPixmap(QPixmap("icons/tv.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h2 = QLabel("Watched List Movies")
        h2.setFont(QFont("Segoe UI", 18, QFont.Bold))
        h2.setProperty("cssClass", "text_primary")
        h2_lay.addWidget(h2_ic)
        h2_lay.addWidget(h2)
        h2_lay.addStretch()
        r_lay.addLayout(h2_lay)
        
        # Search bar for rated movies
        self.search_rated = QLineEdit()
        self.search_rated.setPlaceholderText("Find Movie...")
        self.search_rated.addAction(QIcon("icons/search.png"), QLineEdit.LeadingPosition)
        self.search_rated.textChanged.connect(self._filter_watched_movies)
        r_lay.addWidget(self.search_rated)
        r_lay.addSpacing(10)
        
        panel = QFrame()
        panel.setProperty("cssClass", "panel")
        p_lay = QVBoxLayout(panel)
        p_lay.setContentsMargins(20, 20, 20, 20)
        
        # Table Header
        th = QHBoxLayout()
        l1 = QLabel("MOVIE TITLE")
        l1.setFont(QFont("Segoe UI", 10, QFont.Bold))
        l1.setProperty("cssClass", "text_light")
        th.addWidget(l1)
        th.addStretch()
        l2 = QLabel("USER RATING")
        l2.setFont(QFont("Segoe UI", 10, QFont.Bold))
        l2.setProperty("cssClass", "text_light")
        th.addWidget(l2)
        p_lay.addLayout(th)
        
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {self.P['border']}; border: none; margin: 10px 0;")
        p_lay.addWidget(div)
        
        # Scroll area for rated movies
        scroll_watched = QScrollArea()
        scroll_watched.setWidgetResizable(True)
        scroll_watched.setFrameShape(QFrame.NoFrame)
        scroll_watched.setStyleSheet("background: transparent;")
        
        self.watched_container = QWidget()
        self.watched_container.setStyleSheet("background: transparent;")
        self.watched_lay = QVBoxLayout(self.watched_container)
        self.watched_lay.setContentsMargins(0, 0, 0, 0)
        self.watched_lay.setSpacing(0)
        self.watched_lay.addStretch()
        
        scroll_watched.setWidget(self.watched_container)
        p_lay.addWidget(scroll_watched)
        
        r_lay.addWidget(panel)
        lay.addWidget(right_w, 1)
        
        self.tabs.addWidget(tab)

    def _build_tab_graph(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(40, 30, 40, 30)

        # ── Header ──────────────────────────────────────────────────────────
        h_lay = QHBoxLayout()
        h1_ic = QLabel()
        h1_ic.setPixmap(QPixmap("icons/data-analytics.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h1 = QLabel("Graph Visualization")
        h1.setFont(QFont("Segoe UI", 18, QFont.Bold))
        h1.setProperty("cssClass", "text_primary")
        h_lay.addWidget(h1_ic)
        h_lay.addWidget(h1)
        h_lay.addStretch()

        self.lbl_stat_users  = QLabel("Users: 0")
        self.lbl_stat_movies = QLabel("Movies: 0")
        self.lbl_stat_edges  = QLabel("Edges: 0")
        for lbl in (self.lbl_stat_users, self.lbl_stat_movies, self.lbl_stat_edges):
            lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
            lbl.setProperty("cssClass", "text_second")
            h_lay.addWidget(lbl)
            h_lay.addSpacing(15)
        lay.addLayout(h_lay)
        lay.addSpacing(8)

        # ── Step label ──────────────────────────────────────────────────────
        self.lbl_step = QLabel("Generate recommendations to start")
        self.lbl_step.setAlignment(Qt.AlignCenter)
        self.lbl_step.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.lbl_step.setProperty("cssClass", "text_primary")
        lay.addWidget(self.lbl_step)
        lay.addSpacing(4)

        # ── Canvas + Sidebar Layout ─────────────────────────────────────────
        main_h_lay = QHBoxLayout()
        main_h_lay.setSpacing(15)

        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)
        
        canvas_row = QHBoxLayout()
        canvas_row.setSpacing(0)

        self.btn_graph_prev = QPushButton("❮")
        self.btn_graph_prev.setFixedSize(44, 44)
        self.btn_graph_prev.setFont(QFont("Segoe UI", 16))
        self.btn_graph_prev.setEnabled(False)
        self.btn_graph_prev.clicked.connect(self._graph_prev)
        self.btn_graph_prev.setStyleSheet(
            "QPushButton { border-radius: 22px; background: transparent; }"
            "QPushButton:hover:enabled { background: rgba(83,74,183,0.15); }"
            "QPushButton:disabled { color: #aaa; }"
        )

        self.graph_canvas = GraphCanvas()
        self.graph_canvas.setProperty("cssClass", "card")

        self.btn_graph_next = QPushButton("❯")
        self.btn_graph_next.setFixedSize(44, 44)
        self.btn_graph_next.setFont(QFont("Segoe UI", 16))
        self.btn_graph_next.setEnabled(False)
        self.btn_graph_next.clicked.connect(self._graph_next)
        self.btn_graph_next.setStyleSheet(
            "QPushButton { border-radius: 22px; background: transparent; }"
            "QPushButton:hover:enabled { background: rgba(83,74,183,0.15); }"
            "QPushButton:disabled { color: #aaa; }"
        )

        canvas_row.addWidget(self.btn_graph_prev, 0, Qt.AlignVCenter)
        canvas_row.addWidget(self.graph_canvas, 1)
        canvas_row.addWidget(self.btn_graph_next, 0, Qt.AlignVCenter)
        left_lay.addLayout(canvas_row, 1)

        # ── Dot indikator step ──────────────────────────────────────────────
        dots_lay = QHBoxLayout()
        dots_lay.setAlignment(Qt.AlignCenter)
        dots_lay.setSpacing(8)
        self._step_dots = []
        for i in range(len(GraphCanvas.STEP_TITLES)):
            dot = QLabel("●")
            dot.setFont(QFont("Segoe UI", 10))
            dot.setAlignment(Qt.AlignCenter)
            dots_lay.addWidget(dot)
            self._step_dots.append(dot)
        left_lay.addLayout(dots_lay)
        left_lay.addSpacing(4)
        main_h_lay.addWidget(left_w, 3)
        
        # ── Sidebar ─────────────────────────────────────────────────────────
        right_w = QFrame()
        right_w.setFixedWidth(260)
        right_w.setProperty("cssClass", "panel")
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(15, 15, 15, 15)
        
        self.lbl_sb = QLabel("Top Candidates")
        self.lbl_sb.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_sb.setProperty("cssClass", "text_primary")
        self.lbl_sb.setAlignment(Qt.AlignCenter)
        right_lay.addWidget(self.lbl_sb)
        
        scr_sb = QScrollArea()
        scr_sb.setWidgetResizable(True)
        scr_sb.setFrameShape(QFrame.NoFrame)
        scr_sb.setStyleSheet("background: transparent;")
        
        self.step_recs_container = QWidget()
        self.step_recs_container.setStyleSheet("background: transparent;")
        self.step_recs_lay = QVBoxLayout(self.step_recs_container)
        self.step_recs_lay.setContentsMargins(0, 0, 0, 0)
        self.step_recs_lay.setSpacing(5)
        self.step_recs_lay.addStretch()
        scr_sb.setWidget(self.step_recs_container)
        
        right_lay.addWidget(scr_sb)
        main_h_lay.addWidget(right_w, 1)
        
        lay.addLayout(main_h_lay, 1)

        self._update_graph_nav()
        self.tabs.addWidget(tab)

    def _build_tab_adj(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(40, 30, 40, 30)
        
        h1_lay = QHBoxLayout()
        h1_ic = QLabel()
        h1_ic.setPixmap(QPixmap("icons/list-text.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h1 = QLabel("Adjacency List")
        h1.setFont(QFont("Segoe UI", 18, QFont.Bold))
        h1.setProperty("cssClass", "text_primary")
        h1_lay.addWidget(h1_ic)
        h1_lay.addWidget(h1)
        h1_lay.addStretch()
        lay.addLayout(h1_lay)
        
        self.txt_adj = QTextEdit()
        self.txt_adj.setReadOnly(True)
        lay.addWidget(self.txt_adj, 1)
        self.tabs.addWidget(tab)

    # UI Interactions 
    def _switch_tab(self, idx):
        self.tabs.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            btn.set_active(i == idx)

    def _graph_next(self):
        self.graph_canvas.go_next()
        self._update_graph_nav()

    def _graph_prev(self):
        self.graph_canvas.go_prev()
        self._update_graph_nav()

    def _update_graph_nav(self):
        """Perbarui tombol panah, label step, dan dot indikator."""
        has_data = self.graph_canvas._graph_data is not None
        cur      = self.graph_canvas.current_step
        total    = self.graph_canvas.total_steps

        self.btn_graph_prev.setEnabled(has_data and cur > 0)
        self.btn_graph_next.setEnabled(has_data and cur < total - 1)

        if has_data:
            lbl, _ = GraphCanvas.STEP_TITLES[cur]
            self.lbl_step.setText(lbl)
        else:
            self.lbl_step.setText("Generate recommendations to start")

        if hasattr(self, 'step_recs_lay'):
            for i in reversed(range(self.step_recs_lay.count() - 1)):
                w = self.step_recs_lay.itemAt(i).widget()
                if w: w.setParent(None)
                
            if has_data and hasattr(self.graph_canvas, '_graph_data'):
                d = self.graph_canvas._graph_data
                if cur == 0:
                    self.lbl_sb.setText("Watched Movies")
                    target_movies = d['graph'].get_user_movies(d['target_uid'])
                    sorted_movies = sorted(target_movies.items(), key=lambda x: x[1], reverse=True)[:10]
                    for idx, (mid, rating) in enumerate(sorted_movies):
                        info = d['graph'].movie_info.get(mid, {})
                        title = info.get('title', f'Movie {mid}')
                        row = QFrame()
                        row.setFixedHeight(45)
                        row.setStyleSheet("background: transparent; border-bottom: 1px solid #334155;")
                        r_lay = QHBoxLayout(row)
                        r_lay.setContentsMargins(5, 5, 5, 5)
                        short_title = (title[:18] + '..') if len(title) > 18 else title
                        t_lbl = QLabel(f"{idx+1}. {short_title}")
                        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
                        t_lbl.setProperty("cssClass", "text_primary")
                        s_lbl = QLabel(f"★ {rating}")
                        s_lbl.setFont(QFont("Segoe UI", 10))
                        s_lbl.setProperty("cssClass", "text_second")
                        r_lay.addWidget(t_lbl, 1)
                        r_lay.addWidget(s_lbl, 0)
                        self.step_recs_lay.insertWidget(self.step_recs_lay.count()-1, row)
                elif cur == 1:
                    self.lbl_sb.setText("Top Similar Users")
                    sim_users = d['sim_users']
                    sim_scores = d['sim_scores']
                    sim_raw = d.get('sim_raw', {})
                    for idx, uid in enumerate(sim_users):
                        score = sim_scores.get(uid, 0)
                        co_rated = sim_raw.get(uid, 0)
                        row = QFrame()
                        row.setFixedHeight(45)
                        row.setStyleSheet("background: transparent; border-bottom: 1px solid #334155;")
                        r_lay = QHBoxLayout(row)
                        r_lay.setContentsMargins(5, 5, 5, 5)
                        t_lbl = QLabel(f"{idx+1}. User {uid}")
                        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
                        t_lbl.setProperty("cssClass", "text_primary")
                        v_lay = QVBoxLayout()
                        v_lay.setSpacing(0)
                        s_lbl = QLabel(f"Sim: {score:.2f}")
                        s_lbl.setFont(QFont("Segoe UI", 9))
                        s_lbl.setProperty("cssClass", "text_blue")
                        s_lbl.setAlignment(Qt.AlignRight)
                        c_lbl = QLabel(f"Co-rated: {co_rated}")
                        c_lbl.setFont(QFont("Segoe UI", 8))
                        c_lbl.setProperty("cssClass", "text_light")
                        c_lbl.setAlignment(Qt.AlignRight)
                        v_lay.addWidget(s_lbl)
                        v_lay.addWidget(c_lbl)
                        r_lay.addWidget(t_lbl, 1)
                        r_lay.addLayout(v_lay, 0)
                        self.step_recs_lay.insertWidget(self.step_recs_lay.count()-1, row)
                elif cur == 2:
                    self.lbl_sb.setText("Candidate Movies")
                    # show top candidate movies overall across the 5 similar users (without target watched)
                    if hasattr(self.graph_canvas, 'step_top_movies') and len(self.graph_canvas.step_top_movies) > 4:
                        candidates = self.graph_canvas.step_top_movies[4] # step 4 had the recs generated by all 5 sim users
                    else:
                        candidates = []
                    for idx, movie in enumerate(candidates):
                        row = QFrame()
                        row.setFixedHeight(45)
                        row.setStyleSheet("background: transparent; border-bottom: 1px solid #334155;")
                        r_lay = QHBoxLayout(row)
                        r_lay.setContentsMargins(5, 5, 5, 5)
                        title = movie['title']
                        short_title = (title[:18] + '..') if len(title) > 18 else title
                        t_lbl = QLabel(f"{idx+1}. {short_title}")
                        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
                        t_lbl.setProperty("cssClass", "text_primary")
                        s_lbl = QLabel(f"★ {movie['score']:.2f}")
                        s_lbl.setFont(QFont("Segoe UI", 10))
                        s_lbl.setProperty("cssClass", "text_warning" if 'text_warning' in self.P else "text_blue")
                        s_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        r_lay.addWidget(t_lbl, 1)
                        r_lay.addWidget(s_lbl, 0)
                        self.step_recs_lay.insertWidget(self.step_recs_lay.count()-1, row)
                elif cur == 3:
                    self.lbl_sb.setText("Top Recommendations")
                    recs = d.get('rec_info', {}).values()
                    sorted_recs = sorted(recs, key=lambda x: x['score'], reverse=True)[:5]
                    for idx, movie in enumerate(sorted_recs):
                        row = QFrame()
                        row.setFixedHeight(45)
                        row.setStyleSheet("background: transparent; border-bottom: 1px solid #334155;")
                        r_lay = QHBoxLayout(row)
                        r_lay.setContentsMargins(5, 5, 5, 5)
                        title = movie['title']
                        short_title = (title[:18] + '..') if len(title) > 18 else title
                        t_lbl = QLabel(f"{idx+1}. {short_title}")
                        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
                        t_lbl.setProperty("cssClass", "text_primary")
                        s_lbl = QLabel(f"★ {movie['score']:.2f}")
                        s_lbl.setFont(QFont("Segoe UI", 10))
                        s_lbl.setProperty("cssClass", "text_danger")
                        s_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        r_lay.addWidget(t_lbl, 1)
                        r_lay.addWidget(s_lbl, 0)
                        self.step_recs_lay.insertWidget(self.step_recs_lay.count()-1, row)

        for i, dot in enumerate(self._step_dots):
            if not has_data:
                dot.setStyleSheet("color: #cccccc;")
            elif i == cur:
                dot.setStyleSheet("color: #534AB7; font-size: 13px;")
            elif i < cur:
                dot.setStyleSheet("color: #534AB7; opacity: 0.5;")
            else:
                dot.setStyleSheet("color: #cccccc;")

    def _tick_progress(self):
        self._prog_val = min(self._prog_val + 3, 90)
        self.login_progress.setValue(self._prog_val)

    def _stop_progress(self):
        self._prog_timer.stop()
        self.login_progress.setValue(100)
        QTimer.singleShot(800, lambda: self.login_progress.setValue(0))

    # Logic 
    def _load_data(self):
        self._prog_val = 0
        self._prog_timer.start(60)
        self.login_status.setText("Loading dataset...")
        
        self._thread_load = LoaderThread()
        self._thread_load.status.connect(lambda s: self.login_status.setText(s))
        self._thread_load.done.connect(self._on_load_done)
        self._thread_load.error.connect(lambda e: self.login_status.setText(f"Error: {e}"))
        self._thread_load.start()

    def _on_load_done(self, data):
        self.graph, self.bfs, self.cf = data
        self.is_loaded = True
        self._stop_progress()
        self.login_status.setText("Ready. Please enter User ID.")
        self.login_status.setProperty("cssClass", "text_blue")
        self.login_status.style().unpolish(self.login_status)
        self.login_status.style().polish(self.login_status)
        self.login_inp_uid.setEnabled(True)
        self.login_btn.setEnabled(True)
        self.graph_canvas._draw_empty(self.P)
        self._update_graph_nav()
        
        # Populate Stats
        if hasattr(self.graph, 'get_stats'):
            stats = self.graph.get_stats()
            self.lbl_stat_users.setText(f"Users: {stats.get('total_users', 0):,}")
            self.lbl_stat_movies.setText(f"Movies: {stats.get('total_movies', 0):,}")
            self.lbl_stat_edges.setText(f"Edges: {stats.get('total_edges', 0):,}")

    def _on_login_generate(self):
        raw = self.login_inp_uid.text().strip()
        try:
            uid = int(raw)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", f"User ID '{raw}' harus berupa angka bulat.")
            return
            
        if not self.graph.has_user(uid):
            sample = list(self.graph.users)[:5]
            QMessageBox.warning(self, "User Not Found", f"User ID {uid} tidak ditemukan di dataset.\n\nContoh User ID yang valid: {sample}")
            return
            
        self.cur_uid = uid
        self.login_inp_uid.setEnabled(False)
        self.login_btn.setEnabled(False)
        self._prog_val = 0
        self._prog_timer.start(60)
        self.login_status.setText(f"Generating recommendations for User {uid}...")
        self.login_status.setProperty("cssClass", "text_blue")
        self.login_status.style().unpolish(self.login_status)
        self.login_status.style().polish(self.login_status)
        
        self._thread_rec = RecommendThread(self.graph, self.bfs, self.cf, uid)
        self._thread_rec.status.connect(lambda s: self.login_status.setText(s))
        self._thread_rec.error.connect(lambda e: self.login_status.setText(f"Error: {e}"))
        self._thread_rec.done.connect(self._on_rec_done)
        self._thread_rec.start()

    def _on_rec_done(self, result):
        self._stop_progress()
        self.last_result = result
        uid         = result['uid']
        sim_scores  = result['sim_scores']
        recs        = result['recs']
        
        self.lbl_breadcrumb.setText(f"DASHBOARD  >  User {uid} Profiling")
        self.lbl_profile.setText(f"User {uid} Profiling")

        # Sidebar Similar Users
        for i in reversed(range(self.sim_rows_lay.count())): 
            w = self.sim_rows_lay.itemAt(i).widget()
            if w: w.setParent(None)
            
        for su, sim in sim_scores[:4]:
            self.sim_rows_lay.addWidget(SimUserRow(su, sim))
            
        # Top 5 Recs
        for i in reversed(range(self.rec_lay.count())): 
            w = self.rec_lay.itemAt(i).widget()
            if w: w.setParent(None)
            
        for rec in recs:
            card = RecCard(rec)
            card.on_click_callback = self._show_contributors_dialog
            self.rec_lay.insertWidget(self.rec_lay.count()-1, card)
            
        # Watched Movies
        for i in reversed(range(self.watched_lay.count())): 
            w = self.watched_lay.itemAt(i).widget()
            if w: w.setParent(None)
            
        top_watched = self.graph.get_user_top_movies(uid, top_n=1000) # Load all
        for mid, title, rating in top_watched:
            short = title[:35]+"..." if len(title)>35 else title
            self.watched_lay.insertWidget(self.watched_lay.count()-1, WatchedRow(short, rating))
            
        # Graph & Adj
        candidates = result['candidates']
        sim_raw = result['sim_raw']
        self.graph_canvas.load_graph(self.graph, uid, sim_scores, recs, self.P, candidates, sim_raw)
        self._update_graph_nav()
        self.txt_adj.setPlainText(self.graph.get_adjacency_list_str(uid, limit=20))

        # Reset button state safely
        self.login_inp_uid.setEnabled(True)
        self.login_btn.setEnabled(True)
        self.login_status.setText("Ready.")
        
        # Switch Screen
        self.stacked_widget.setCurrentWidget(self.main_widget)
        self._switch_tab(0)

    def _show_contributors_dialog(self, rec_data):
        contributors = rec_data.get('contributors', [])
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Rincian Rekomendasi: {rec_data.get('title')}")
        dialog.setMinimumSize(600, 400)
        dialog.setStyleSheet(f"""
            QDialog {{ background: {self.P['bg_panel']}; color: {self.P['text_primary']}; }}
            QLabel {{ color: {self.P['text_primary']}; }}
            QTableWidget {{ 
                background: {self.P['bg_app']}; 
                color: {self.P['text_primary']}; 
                border: 1px solid {self.P['border']};
                border-radius: 8px;
                gridline-color: {self.P['border']};
            }}
            QHeaderView::section {{ 
                background: {self.P['bg_panel']}; 
                color: {self.P['text_second']}; 
                border: none;
                border-bottom: 1px solid {self.P['border']};
                font-weight: bold;
                padding: 4px;
            }}
            QTableWidget::item {{ padding: 8px; }}
        """)
        
        lay = QVBoxLayout(dialog)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(15)
        
        title_lbl = QLabel(f"{rec_data.get('title')}")
        title_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lay.addWidget(title_lbl)
        
        desc_lbl = QLabel(f"Top {len(contributors)} User yang merekomendasikan film ini:")
        desc_lbl.setFont(QFont("Segoe UI", 12))
        desc_lbl.setStyleSheet(f"color: {self.P['text_second']};")
        lay.addWidget(desc_lbl)
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["User ID", "Rating", "Similarity", "Kontribusi"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        
        if self.P['is_dark']:
            table.setStyleSheet(table.styleSheet() + f"QTableWidget {{ alternate-background-color: {self.P['bg_panel']}; }}")
        else:
            table.setStyleSheet(table.styleSheet() + f"QTableWidget {{ alternate-background-color: #F1F5F9; }}")
            
        table.setRowCount(len(contributors))
        for row, c in enumerate(contributors):
            item_id = QTableWidgetItem(f"User {c['user_id']}")
            item_id.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 0, item_id)
            
            item_rat = QTableWidgetItem(f"{c['rating']} ★")
            item_rat.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 1, item_rat)
            
            item_sim = QTableWidgetItem(f"{c['similarity']:.4f}")
            item_sim.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 2, item_sim)
            
            item_cont = QTableWidgetItem(f"{c['contribution']:.4f}")
            item_cont.setTextAlignment(Qt.AlignCenter)
            item_cont.setFont(QFont("Segoe UI", 10, QFont.Bold))
            item_cont.setForeground(QColor(self.P['primary_blue']))
            table.setItem(row, 3, item_cont)
            
        lay.addWidget(table)
        
        btn_close = QPushButton("Tutup")
        btn_close.setFixedHeight(40)
        btn_close.setProperty("cssClass", "btn_primary")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(dialog.accept)
        
        btn_lay = QHBoxLayout()
        btn_lay.addStretch()
        btn_lay.addWidget(btn_close)
        btn_lay.addStretch()
        lay.addLayout(btn_lay)
        
        dialog.exec_()

    def _filter_watched_movies(self, text):
        query = text.lower()
        for i in range(self.watched_lay.count() - 1):
            item = self.watched_lay.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if hasattr(w, 'title_text'):
                    w.setVisible(query in w.title_text)

    def _on_reset(self):
        # State Safety Management
        self.cur_uid = None
        self.last_result = None
        
        # Explicitly ensure login state is safe to use again
        if self.is_loaded:
            self.login_inp_uid.setEnabled(True)
            self.login_btn.setEnabled(True)
            self.login_status.setText("Ready. Please enter User ID.")
            self.login_status.setProperty("cssClass", "text_blue")
            self.login_status.style().unpolish(self.login_status)
            self.login_status.style().polish(self.login_status)
            
        self.login_inp_uid.clear()
        self.stacked_widget.setCurrentWidget(self.login_widget)

def main():
    
    import traceback
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        with open("error_log.txt", "w") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.excepthook = global_exception_handler
    
    app = QApplication(sys.argv)
    win = MovieRecommenderApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()