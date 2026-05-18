# ============================================================================
# gui_app.py — Antarmuka GUI PyQt5 (Premium Redesign — Figma-Quality)
# ============================================================================
#
# Desain: Dark Obsidian + Cyan Neon — premium, modern, high-contrast
#
# Layout:
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │  HEADER  (logo + judul + badge teknologi + live stats)              │
#   ├──────────────────┬───────────────────────────────────────────────────┤
#   │  SIDEBAR (290px) │  MAIN CONTENT (TAB: Rekomendasi / Graf / Adj)    │
#   │  ─ User Input    │                                                   │
#   │  ─ Graph Stats   │                                                   │
#   │  ─ Film Search   │                                                   │
#   ├──────────────────┴───────────────────────────────────────────────────┤
#   │  STATUS BAR                                                          │
#   └──────────────────────────────────────────────────────────────────────┘
# ============================================================================

import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QScrollArea, QTextEdit,
    QTabWidget, QSizePolicy, QGridLayout, QGraphicsDropShadowEffect,
    QProgressBar, QSplitter, QStackedWidget, QSpacerItem
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QSize, pyqtProperty, QObject
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QLinearGradient, QPainter,
    QBrush, QPen, QPixmap, QIcon, QFontMetrics, QPainterPath,
    QRadialGradient
)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Conditional imports (ubah path sesuai project Anda) ─────────────────────
try:
    from graph_ds    import BipartiteGraph
    from algorithms  import BFSTraversal, CollaborativeFilter
    from data_loader import DataLoader
    _DEMO_MODE = False
except ImportError:
    _DEMO_MODE = True

# ============================================================================
# DEMO STUBS  (aktif jika modul project belum tersedia)
# ============================================================================
if _DEMO_MODE:
    import random, time

    class _DemoGraph:
        def __init__(self):
            self.users = set(range(1, 51))
            self.movie_info = {
                i: {'title': t, 'genres': g}
                for i, (t, g) in enumerate([
                    ("The Shawshank Redemption", "Drama"),
                    ("The Godfather", "Crime|Drama"),
                    ("The Dark Knight", "Action|Crime"),
                    ("Pulp Fiction", "Crime|Drama"),
                    ("Forrest Gump", "Drama|Romance"),
                    ("Inception", "Action|Sci-Fi"),
                    ("The Matrix", "Action|Sci-Fi"),
                    ("Goodfellas", "Biography|Crime"),
                    ("Fight Club", "Drama|Thriller"),
                    ("Interstellar", "Adventure|Sci-Fi"),
                    ("Parasite", "Drama|Thriller"),
                    ("Avengers: Endgame", "Action|Adventure"),
                    ("Your Name", "Animation|Drama"),
                    ("Whiplash", "Drama|Music"),
                    ("1917", "Drama|War"),
                ], start=1)
            }
            self._ratings = {
                u: {random.randint(1,15): round(random.uniform(1,5),1)
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
            top = sorted(movies.items(), key=lambda x: -x[1])[:top_n]
            return [(mid, self.movie_info.get(mid,{}).get('title','?'), r) for mid,r in top]
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
            return sorted(
                [(u, round(random.uniform(0.3, 0.97), 4)) for u in list(candidates)[:top_k]],
                key=lambda x: -x[1]
            )
        def get_recommendations(self, uid, sim_scores, candidates, top_n=5):
            items = sorted(candidates.items(), key=lambda x: -x[1])[:top_n]
            res = []
            for mid, score in items:
                info = self.g.movie_info.get(mid, {})
                res.append({
                    'movie_id': mid,
                    'title': info.get('title', f'Movie {mid}'),
                    'genres': info.get('genres', 'Unknown'),
                    'score': round(score, 3),
                    'rated_by': random.randint(2, 10)
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
# PALETTE  ─ Obsidian + Neon Cyan
# ============================================================================
P = {
    # Backgrounds
    "bg_deep":    "#07090F",   # Paling gelap
    "bg_base":    "#0D1117",   # Base utama
    "bg_panel":   "#161B22",   # Panel / sidebar
    "bg_card":    "#1C2330",   # Card
    "bg_card2":   "#212B3A",   # Card hover / alt

    # Accent
    "cyan":       "#00D9FF",   # Aksen utama (neon cyan)
    "cyan_dk":    "#0099BB",   # Versi gelap
    "cyan_glow":  "#00D9FF22", # Transparan untuk glow
    "purple":     "#9F7AEA",   # Secondary accent
    "pink":       "#F687B3",   # Tertiary

    # Text
    "txt_h":      "#F0F6FC",   # Heading
    "txt_b":      "#8B949E",   # Body
    "txt_m":      "#484F58",   # Muted

    # Semantic
    "green":      "#3FB950",   # Success
    "yellow":     "#D29922",   # Warning
    "red":        "#F85149",   # Error
    "orange":     "#E3B341",   # Info

    # Borders
    "border":     "#30363D",
    "border_act": "#00D9FF",

    # Table
    "row_a":      "#161B22",
    "row_b":      "#1C2330",
    "row_top":    "#1A2A3A",
}

# ============================================================================
# GLOBAL STYLESHEET
# ============================================================================
QSS = f"""
/* ── Root ───────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background: {P['bg_base']};
    color: {P['txt_h']};
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial;
    font-size: 13px;
}}

/* ── QScrollBar ─────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {P['bg_base']};
    width: 6px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {P['border']};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {P['cyan_dk']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {P['bg_base']};
    height: 6px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {P['border']};
    border-radius: 3px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {P['cyan_dk']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── QLineEdit ──────────────────────────────────────────────── */
QLineEdit {{
    background: {P['bg_card']};
    border: 1.5px solid {P['border']};
    border-radius: 8px;
    color: {P['txt_h']};
    padding: 8px 14px;
    font-size: 14px;
    selection-background-color: {P['cyan_dk']};
}}
QLineEdit:focus {{
    border: 1.5px solid {P['cyan']};
    background: {P['bg_card2']};
}}
QLineEdit::placeholder {{
    color: {P['txt_m']};
}}

/* ── QPushButton ────────────────────────────────────────────── */
QPushButton {{
    border-radius: 8px;
    padding: 9px 20px;
    font-size: 13px;
    font-weight: 600;
    border: none;
    cursor: pointer;
}}
QPushButton#btn_generate {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #00D9FF, stop:1 #0099BB);
    color: #07090F;
    font-size: 14px;
    font-weight: 700;
    padding: 12px 24px;
}}
QPushButton#btn_generate:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #33E5FF, stop:1 #00AACC);
}}
QPushButton#btn_generate:pressed {{
    background: {P['cyan_dk']};
}}
QPushButton#btn_reset {{
    background: {P['bg_card2']};
    color: {P['txt_b']};
    border: 1px solid {P['border']};
}}
QPushButton#btn_reset:hover {{
    background: {P['bg_card']};
    color: {P['txt_h']};
    border-color: {P['border_act']};
}}
QPushButton#btn_search {{
    background: {P['purple']};
    color: white;
    font-weight: 600;
    padding: 9px 16px;
}}
QPushButton#btn_search:hover {{
    background: #B794F4;
}}

/* ── QTabWidget ─────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {P['border']};
    border-top: none;
    background: {P['bg_panel']};
    border-radius: 0 0 10px 10px;
}}
QTabBar::tab {{
    background: {P['bg_card']};
    color: {P['txt_b']};
    padding: 10px 24px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid {P['border']};
    border-bottom: none;
    min-width: 120px;
}}
QTabBar::tab:selected {{
    background: {P['bg_panel']};
    color: {P['cyan']};
    font-weight: 700;
    border-color: {P['border']};
    border-bottom: 2px solid {P['cyan']};
}}
QTabBar::tab:hover:!selected {{
    background: {P['bg_card2']};
    color: {P['txt_h']};
}}

/* ── QTextEdit ──────────────────────────────────────────────── */
QTextEdit {{
    background: {P['bg_card']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    color: {P['txt_b']};
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 10px;
    selection-background-color: {P['cyan_dk']};
}}

/* ── QProgressBar ───────────────────────────────────────────── */
QProgressBar {{
    background: {P['bg_card']};
    border: none;
    border-radius: 4px;
    height: 4px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {P['cyan']}, stop:1 {P['purple']});
    border-radius: 4px;
}}

/* ── QLabel tooltips ────────────────────────────────────────── */
QToolTip {{
    background: {P['bg_card2']};
    color: {P['txt_h']};
    border: 1px solid {P['border']};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
"""


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================

class GlowLabel(QLabel):
    """Label dengan efek glow cyan di belakangnya."""
    def __init__(self, text, parent=None, glow_color=None):
        super().__init__(text, parent)
        self._glow = glow_color or QColor(P['cyan'])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Tidak perlu custom paint — biarkan QLabel handle text,
        # drop shadow diset via QGraphicsDropShadowEffect
        super().paintEvent(event)


def make_shadow(color=None, blur=20, x_off=0, y_off=4):
    eff = QGraphicsDropShadowEffect()
    eff.setColor(QColor(color or P['cyan']))
    eff.setBlurRadius(blur)
    eff.setOffset(x_off, y_off)
    return eff


class StatCard(QFrame):
    """Kartu statistik kecil dengan icon + nilai + label."""
    def __init__(self, icon, label, value="—", accent=None, parent=None):
        super().__init__(parent)
        self._accent = accent or P['cyan']
        self.setFixedHeight(78)
        self.setStyleSheet(f"""
            QFrame {{
                background: {P['bg_card']};
                border: 1px solid {P['border']};
                border-left: 3px solid {self._accent};
                border-radius: 10px;
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)

        icon_lbl = QLabel(icon)
        icon_lbl.setFont(QFont("Segoe UI Emoji", 22))
        icon_lbl.setFixedWidth(40)
        lay.addWidget(icon_lbl)

        vlay = QVBoxLayout()
        vlay.setSpacing(2)
        self.val_lbl = QLabel(value)
        self.val_lbl.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.val_lbl.setStyleSheet(f"color: {self._accent}; border: none;")
        vlay.addWidget(self.val_lbl)

        self.key_lbl = QLabel(label)
        self.key_lbl.setFont(QFont("Segoe UI", 9))
        self.key_lbl.setStyleSheet(f"color: {P['txt_b']}; border: none;")
        vlay.addWidget(self.key_lbl)

        lay.addLayout(vlay)
        lay.addStretch()

    def set_value(self, v):
        self.val_lbl.setText(str(v))


class SectionHeader(QWidget):
    """Section header dengan garis aksen vertikal + teks."""
    def __init__(self, text, parent=None, accent=None):
        super().__init__(parent)
        accent = accent or P['cyan']
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 4)
        lay.setSpacing(10)

        bar = QFrame()
        bar.setFixedWidth(3)
        bar.setStyleSheet(f"background: {accent}; border-radius: 2px;")
        lay.addWidget(bar)

        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl.setStyleSheet(f"color: {P['txt_h']}; letter-spacing: 0.5px;")
        lay.addWidget(lbl)
        lay.addStretch()


class Badge(QLabel):
    """Badge kecil untuk header."""
    def __init__(self, text, bg, fg="#F0F6FC", parent=None):
        super().__init__(f"  {text}  ", parent)
        self.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                color: {fg};
                border-radius: 5px;
                padding: 3px 2px;
            }}
        """)
        self.setFixedHeight(22)


class SimBar(QWidget):
    """Bar similarity bergaya dengan gradient fill."""
    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self.setFixedHeight(12)
        self.setMinimumWidth(120)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.setBrush(QBrush(QColor(P['bg_deep'])))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, w, h, 4, 4)

        # Fill dengan gradient
        fill_w = int(w * self._value)
        if fill_w > 0:
            if self._value >= 0.75:
                c1, c2 = QColor(P['cyan']), QColor(P['green'])
            elif self._value >= 0.45:
                c1, c2 = QColor(P['purple']), QColor(P['cyan'])
            else:
                c1, c2 = QColor(P['txt_m']), QColor(P['purple'])
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, c1)
            grad.setColorAt(1, c2)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(0, 0, fill_w, h, 4, 4)


class RecCard(QFrame):
    """Kartu rekomendasi film premium."""
    RANK_COLORS = [P['yellow'], "#C0C0C0", "#CD7F32", P['txt_b'], P['txt_b']]
    RANK_ICONS  = ["🥇", "🥈", "🥉", "④", "⑤"]

    def __init__(self, rank, title, genres, score, rated_by, parent=None):
        super().__init__(parent)
        is_top = (rank == 1)
        bg = P['row_top'] if is_top else (P['row_a'] if rank % 2 == 1 else P['row_b'])
        border_l = P['yellow'] if is_top else P['border']
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: 1px solid {P['border']};
                border-left: 3px solid {border_l};
                border-radius: 8px;
                margin: 2px 0;
            }}
            QLabel {{ border: none; background: transparent; }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 16, 10)
        lay.setSpacing(14)

        # Rank icon
        rank_lbl = QLabel(self.RANK_ICONS[min(rank-1, 4)])
        rank_lbl.setFont(QFont("Segoe UI Emoji", 20 if is_top else 16))
        rank_lbl.setFixedWidth(32)
        rank_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(rank_lbl)

        # Film info
        info_lay = QVBoxLayout()
        info_lay.setSpacing(2)
        title_lbl = QLabel(title[:52] + ("…" if len(title) > 52 else ""))
        title_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold if is_top else QFont.Medium))
        title_lbl.setStyleSheet(f"color: {P['txt_h']};")
        info_lay.addWidget(title_lbl)
        genre_lbl = QLabel(genres[:40] + ("…" if len(genres) > 40 else ""))
        genre_lbl.setFont(QFont("Segoe UI", 12))
        genre_lbl.setStyleSheet(f"color: {P['txt_b']};")
        info_lay.addWidget(genre_lbl)
        lay.addLayout(info_lay)
        lay.addStretch()

        # Score + rated_by
        right_lay = QVBoxLayout()
        right_lay.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        score_lbl = QLabel(f"{score:.3f}")
        score_lbl.setFont(QFont("Segoe UI", 15, QFont.Bold))
        score_lbl.setStyleSheet(f"color: {P['green']};")
        score_lbl.setAlignment(Qt.AlignRight)
        right_lay.addWidget(score_lbl)
        rated_lbl = QLabel(f"{rated_by} user")
        rated_lbl.setFont(QFont("Segoe UI", 9))
        rated_lbl.setStyleSheet(f"color: {P['txt_m']};")
        rated_lbl.setAlignment(Qt.AlignRight)
        right_lay.addWidget(rated_lbl)
        lay.addLayout(right_lay)


class SimUserRow(QFrame):
    """Baris satu similar user."""
    def __init__(self, rank, uid, sim, n_films, parent=None):
        super().__init__(parent)
        bg = P['row_a'] if rank % 2 == 1 else P['row_b']
        sim_color = (P['cyan'] if sim >= 0.75 else
                     P['purple'] if sim >= 0.45 else P['txt_b'])
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: none;
                border-radius: 6px;
                margin: 1px 0;
            }}
            QLabel {{ border: none; background: transparent; }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 14, 6)
        lay.setSpacing(12)

        r = QLabel(str(rank))
        r.setFixedWidth(22)
        r.setFont(QFont("Segoe UI", 12))
        r.setStyleSheet(f"color: {P['txt_m']};")
        lay.addWidget(r)

        u = QLabel(f"User {uid}")
        u.setFixedWidth(70)
        u.setFont(QFont("Segoe UI", 13, QFont.Medium))
        u.setStyleSheet(f"color: {P['txt_h']};")
        lay.addWidget(u)

        sv = QLabel(f"{sim:.4f}")
        sv.setFixedWidth(65)
        sv.setFont(QFont("Cascadia Code", 12, QFont.Bold))
        sv.setStyleSheet(f"color: {sim_color};")
        lay.addWidget(sv)

        bar = SimBar(sim)
        lay.addWidget(bar, 1)

        nf = QLabel(f"{n_films} film")
        nf.setFont(QFont("Segoe UI", 12))
        nf.setStyleSheet(f"color: {P['txt_m']};")
        nf.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        nf.setFixedWidth(55)
        lay.addWidget(nf)


# ============================================================================
# DATA LOADER THREAD
# ============================================================================
class LoaderThread(QThread):
    done    = pyqtSignal(object)
    error   = pyqtSignal(str)
    status  = pyqtSignal(str)

    def run(self):
        try:
            self.status.emit("Memuat dataset MovieLens…")
            loader = DataLoader()
            loader.load_data()
            self.status.emit("Membangun Weighted Bipartite Graph…")
            graph = BipartiteGraph()
            loader.build_graph(graph, max_users=20, max_movies=None)
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
            self.status.emit(f"BFS traversal untuk User {uid}…")
            sim_raw, visited = self.bfs.find_similar_users(uid, max_depth=2)
            self.status.emit("Menghitung cosine similarity…")
            sim_scores = self.cf.get_top_similar_users(uid, sim_raw.keys(), top_k=10)
            self.status.emit("Mengumpulkan kandidat film…")
            candidates = self.bfs.get_candidate_movies(uid, dict(sim_scores))
            self.status.emit("Menghitung recommendation score…")
            recs = self.cf.get_recommendations(uid, sim_scores, candidates, top_n=5)
            self.done.emit({
                'uid': uid, 'sim_scores': sim_scores,
                'recs': recs, 'visited': visited,
                'sim_raw': sim_raw, 'candidates': candidates
            })
        except Exception as e:
            self.error.emit(str(e))


# ============================================================================
# MATPLOTLIB GRAPH CANVAS
# ============================================================================
class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.fig.patch.set_facecolor(P['bg_panel'])
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._draw_empty()

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_facecolor(P['bg_card'])
        self.fig.patch.set_facecolor(P['bg_panel'])
        # Decorative placeholder
        theta = [i * 2 * math.pi / 6 for i in range(6)]
        xs = [0.5 + 0.3 * math.cos(t) for t in theta]
        ys = [0.5 + 0.3 * math.sin(t) for t in theta]
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            self.ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                         color=P['border'], lw=1, alpha=0.5)
            self.ax.plot([0.5, xs[i]], [0.5, ys[i]],
                         color=P['border'], lw=0.8, alpha=0.3, ls='--')
        self.ax.scatter(xs, ys, s=80, c=P['border'], zorder=5)
        self.ax.scatter([0.5], [0.5], s=150, c=P['cyan'], zorder=6,
                        edgecolors=P['bg_panel'], linewidths=2)
        self.ax.text(0.5, 0.5, "●", ha='center', va='center',
                     fontsize=18, color=P['cyan'], zorder=7,
                     transform=self.ax.transAxes)
        self.ax.text(0.5, 0.12,
                     "Generate rekomendasi\nuntuk melihat visualisasi graph",
                     ha='center', va='center', fontsize=11,
                     color=P['txt_m'], transform=self.ax.transAxes,
                     linespacing=1.7)
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.draw()

    def draw_graph(self, graph, target_uid, sim_scores, recs):
        self.ax.clear()
        self.ax.set_facecolor(P['bg_card'])
        self.fig.patch.set_facecolor(P['bg_panel'])

        C_TARGET  = P['yellow']
        C_SIMILAR = P['pink']
        C_REC     = P['cyan']
        C_WATCHED = P['txt_m']
        C_EDGE_R  = P['cyan']
        C_EDGE_O  = "#2A3A4A"
        C_EDGE_W  = "#2D3748"
        C_LABEL   = P['txt_b']

        rec_ids    = [r['movie_id'] for r in recs]
        rec_set    = set(rec_ids)
        rec_score  = {r['movie_id']: r['score'] for r in recs}
        target_w   = list(graph.get_user_movies(target_uid).keys())
        ctx_ids    = [m for m in target_w if m not in rec_set][:4]

        # Pastikan semua film rekomendasi ADA di movie_list meski similar user
        # belum tentu punya koneksi ke ctx_ids — rec_ids selalu masuk duluan.
        movie_list = rec_ids + ctx_ids

        display_u  = [target_uid] + [u for u, _ in sim_scores[:5]]
        n_u, n_m   = len(display_u), len(movie_list)
        if n_m == 0:
            self._draw_empty(); return

        XU, XM = 0.2, 0.58
        PAD = 0.08
        def _y(i, n): return (PAD + (1-2*PAD)/(max(n-1,1))*i) if n>1 else 0.5

        upos = {u: (XU, _y(i, n_u)) for i,u in enumerate(display_u)}
        mpos = {m: (XM, _y(i, n_m)) for i,m in enumerate(movie_list)}

        # ── Draw edges ────────────────────────────────────────────────────────
        # Iterasi per-user, bukan per-movie_list, agar edge similar→rec_movie
        # selalu digambar meskipun film itu tidak ada di ctx_ids target user.
        for uid in display_u:
            ux, uy = upos[uid]
            um     = graph.get_user_movies(uid)
            is_t   = (uid == target_uid)

            for mid, r in um.items():
                if mid not in mpos:
                    # Film ini belum ada di movie_list —
                    # masukkan HANYA jika similar user punya koneksi ke rec film
                    # (bukti kenapa film itu direkomendasikan)
                    if mid in rec_set:
                        # Sudah ada di mpos karena rec_ids selalu dimasukkan,
                        # jadi cabang ini sebenarnya tidak akan terpicu.
                        # Guard ini hanya pengaman.
                        pass
                    # Untuk film non-rec yang tidak ada di movie_list, skip
                    continue

                mx, my   = mpos[mid]
                is_rec   = mid in rec_set

                if is_t:
                    # Target user → film yang sudah ditonton (dashed)
                    self.ax.plot([ux,mx],[uy,my],'--',
                                 color=C_EDGE_W, lw=1.0, alpha=0.45, zorder=2)
                elif is_rec:
                    # Similar user → film rekomendasi (TEAL TEBAL — alasan utama)
                    # Edge ini adalah "bukti" mengapa film direkomendasikan
                    lw = 1.0 + (r/5)*2.5
                    self.ax.plot([ux,mx],[uy,my],'-',
                                 color=C_EDGE_R, lw=lw, alpha=0.75, zorder=3)
                    self.ax.text((ux+mx)/2+0.01, (uy+my)/2+0.01,
                                 f"{r:.1f}★", fontsize=6.5,
                                 color=P['txt_b'], ha='center', va='center', zorder=4,
                                 bbox=dict(boxstyle='round,pad=0.1',
                                           fc=P['bg_card'], ec='none', alpha=0.7))
                else:
                    # Similar user → film konteks (abu tipis)
                    self.ax.plot([ux,mx],[uy,my],'-',
                                 color=C_EDGE_O, lw=0.7, alpha=0.3, zorder=1)

        # Draw user nodes
        for uid, (x, y) in upos.items():
            is_t = uid == target_uid
            c = C_TARGET if is_t else C_SIMILAR
            sz = 260 if is_t else 140
            self.ax.scatter(x, y, s=sz*2.2, c=c, alpha=0.08, zorder=5, linewidths=0)
            self.ax.scatter(x, y, s=sz, c=c, zorder=6,
                            edgecolors=P['bg_panel'], linewidths=1.5)
            sv = next((s for u, s in sim_scores if u == uid), None)
            lbl = f"U{uid}\n(target)" if is_t else (f"U{uid}\n{sv:.2f}" if sv else f"U{uid}")
            self.ax.text(x-0.05, y, lbl, fontsize=7.5, color=c,
                         ha='right', va='center', fontweight='bold' if is_t else 'normal',
                         zorder=7, linespacing=1.4)

        # Draw movie nodes
        for i, mid in enumerate(movie_list):
            x, y = mpos[mid]
            info  = graph.movie_info.get(mid, {})
            title = info.get('title', f'Movie {mid}')
            is_rec = mid in rec_set
            if is_rec:
                rank = rec_ids.index(mid)+1
                c = C_REC
                self.ax.scatter(x, y, s=240, c=c, alpha=0.12,
                                marker='*', zorder=5, linewidths=0)
                self.ax.scatter(x, y, s=150, c=c, marker='*',
                                zorder=6, edgecolors=P['bg_panel'], linewidths=0.8)
                short = title[:24]+"…" if len(title)>24 else title
                self.ax.text(x+0.04, y+0.022, short,
                             fontsize=7, color=c, ha='left', va='center',
                             fontweight='bold', zorder=7, clip_on=False)
                self.ax.text(x+0.04, y-0.022,
                             f"#{rank}  {rec_score[mid]:.2f}",
                             fontsize=6, color=P['yellow'],
                             ha='left', va='center', zorder=7, clip_on=False)
            else:
                c = C_WATCHED
                self.ax.scatter(x, y, s=100, c=c, alpha=0.12,
                                marker='s', zorder=5, linewidths=0)
                self.ax.scatter(x, y, s=60, c=c, marker='s',
                                zorder=6, edgecolors=P['bg_panel'], linewidths=0.7)
                short = title[:24]+"…" if len(title)>24 else title
                self.ax.text(x+0.04, y, short, fontsize=6.5, color=c,
                             ha='left', va='center', zorder=7, clip_on=False)

        # Divider + column labels
        mid_x = (XU+XM)/2
        self.ax.axvline(x=mid_x, color=P['border'], ls='--', alpha=0.35, lw=0.8, zorder=0)
        self.ax.text(XU, 1.04, "USERS", fontsize=8.5, color=C_TARGET,
                     ha='center', fontweight='bold', transform=self.ax.transAxes)
        self.ax.text(0.73, 1.04, "MOVIES", fontsize=8.5, color=C_REC,
                     ha='center', fontweight='bold', transform=self.ax.transAxes)

        # Legend
        legend_items = [
            mpatches.Patch(color=C_EDGE_R, label="Similar → Rekomendasi"),
            mpatches.Patch(color=C_EDGE_W, label="Target → Ditonton"),
            mpatches.Patch(color=C_EDGE_O, label="Koneksi lain"),
        ]
        self.ax.legend(handles=legend_items, loc='lower left', fontsize=6,
                       facecolor=P['bg_card'], edgecolor=P['border'],
                       labelcolor=C_LABEL, framealpha=0.9, handlelength=1.2)

        self.ax.set_xlim(-0.05, 1.08)
        self.ax.set_ylim(-0.04, 1.08)
        self.ax.axis('off')
        self.ax.set_title(
            f"User {target_uid}  —  ★ {len(rec_ids)} rekomendasi  ·  □ {len(ctx_ids)} ditonton",
            fontsize=9, color=C_LABEL, pad=12
        )
        self.fig.tight_layout(pad=0.4)
        self.draw()


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================
class MovieRecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AbiyuRecommender — Sistem Rekomendasi Film Berbasis Graph")
        self.setMinimumSize(1200, 780)
        self.resize(1380, 860)
        self.setStyleSheet(QSS)

        self.graph = None
        self.bfs   = None
        self.cf    = None
        self.is_loaded = False
        self.cur_uid   = None
        self.cur_recs  = []
        self.cur_sims  = []

        central = QWidget()
        self.setCentralWidget(central)
        self._root_lay = QVBoxLayout(central)
        self._root_lay.setContentsMargins(0, 0, 0, 0)
        self._root_lay.setSpacing(0)

        self._build_header()
        self._build_body()
        self._build_statusbar()

        # Animasi loading bar
        self._prog_timer = QTimer()
        self._prog_timer.timeout.connect(self._tick_progress)
        self._prog_val = 0

        # Load data
        QTimer.singleShot(300, self._load_data)

    # ── Header ──────────────────────────────────────────────────────────────

    def _build_header(self):
        # Top accent line (cyan gradient)
        accent_line = QFrame()
        accent_line.setFixedHeight(4)
        accent_line.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {P['cyan']}, stop:0.5 {P['purple']},
                    stop:1 {P['pink']});
                border: none;
            }}
        """)
        self._root_lay.addWidget(accent_line)

        # Header bar
        hdr = QFrame()
        hdr.setFixedHeight(72)
        hdr.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #0D1621, stop:1 #0A1218);
                border-bottom: 1px solid {P['border']};
            }}
        """)
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(24, 0, 24, 0)
        lay.setSpacing(16)

        # Logo placeholder + title
        logo = QLabel("⬡")
        logo.setFont(QFont("Segoe UI", 28))
        logo.setStyleSheet(f"color: {P['cyan']}; border: none;")
        logo.setGraphicsEffect(make_shadow(P['cyan'], blur=18, y_off=0))
        lay.addWidget(logo)

        title_lay = QVBoxLayout()
        title_lay.setSpacing(1)
        t1 = QLabel("Abiyu Recommend")
        t1.setFont(QFont("Segoe UI", 20, QFont.Bold))
        t1.setStyleSheet(f"color: {P['txt_h']}; letter-spacing: -0.5px; border: none;")
        title_lay.addWidget(t1)
        t2 = QLabel("Movie Recommendation System")
        t2.setFont(QFont("Segoe UI", 9))
        t2.setStyleSheet(f"color: {P['txt_b']}; border: none;")
        title_lay.addWidget(t2)
        lay.addLayout(title_lay)
        lay.addStretch()

        # Badges
        for txt, bg, fg in [
            ("Abiyu",          "#0D2A3A", P['cyan']),
            ("Khansa",   "#1A1A3A", P['purple']),
            ("Zaki",    "#0D1A2A", P['txt_h']),
        ]:
            b = Badge(txt, bg, fg)
            lay.addWidget(b)

        self._root_lay.addWidget(hdr)

    # ── Body ─────────────────────────────────────────────────────────────────

    def _build_body(self):
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(16, 14, 16, 8)
        body_lay.setSpacing(14)

        self._build_sidebar(body_lay)
        self._build_main(body_lay)

        self._root_lay.addWidget(body, 1)

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent_lay):
        sidebar = QFrame()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background: {P['bg_panel']};
                border: 1px solid {P['border']};
                border-radius: 12px;
            }}
        """)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(14)

        # ── User Input ─────────────────────────────────────────────────────
        lay.addWidget(SectionHeader("👤  INSERT USER ID"))

        self.inp_uid = QLineEdit("1")
        self.inp_uid.setPlaceholderText("Contoh: 1, 5, 12, …")
        self.inp_uid.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.inp_uid.setAlignment(Qt.AlignCenter)
        self.inp_uid.returnPressed.connect(self._on_generate)
        lay.addWidget(self.inp_uid)

        self.lbl_hint = QLabel("ID EXAMPLE: loading…")
        self.lbl_hint.setFont(QFont("Segoe UI", 9))
        self.lbl_hint.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        lay.addWidget(self.lbl_hint)

        self.btn_generate = QPushButton("Find Recommendation")
        self.btn_generate.setObjectName("btn_generate")
        self.btn_generate.setCursor(Qt.PointingHandCursor)
        self.btn_generate.setGraphicsEffect(make_shadow(P['cyan'], blur=24, y_off=4))
        self.btn_generate.clicked.connect(self._on_generate)
        lay.addWidget(self.btn_generate)

        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.clicked.connect(self._on_reset)
        lay.addWidget(self.btn_reset)

        # Divider
        div = QFrame(); div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"color: {P['border']}; border: none; background: {P['border']}; max-height: 1px;")
        lay.addWidget(div)

        # ── Graph Stats ─────────────────────────────────────────────────────
        lay.addWidget(SectionHeader("📊  GRAPH STATISTIC", accent=P['purple']))

        stat_grid = QGridLayout()
        stat_grid.setSpacing(8)
        self.stat_users  = StatCard("👥", "Total Users",  "—", P['cyan'])
        self.stat_movies = StatCard("🎬", "Total Movies", "—", P['purple'])
        self.stat_edges  = StatCard("🔗", "Total Edges",  "—", P['pink'])
        self.stat_avg    = StatCard("⭐", "Avg Rating/U", "—", P['green'])
        stat_grid.addWidget(self.stat_users,  0, 0)
        stat_grid.addWidget(self.stat_movies, 0, 1)
        stat_grid.addWidget(self.stat_edges,  1, 0)
        stat_grid.addWidget(self.stat_avg,    1, 1)
        lay.addLayout(stat_grid)

        # Divider
        div2 = QFrame(); div2.setFrameShape(QFrame.HLine)
        div2.setStyleSheet(f"color: {P['border']}; border: none; background: {P['border']}; max-height: 1px;")
        lay.addWidget(div2)

        # ── Film Search ─────────────────────────────────────────────────────
        lay.addWidget(SectionHeader("🔍  FIND USER'S FAVORITE MOVIE", accent=P['yellow']))

        search_row = QHBoxLayout()
        self.inp_search = QLineEdit()
        self.inp_search.setPlaceholderText("Type movie title…")
        self.inp_search.returnPressed.connect(self._on_search)
        search_row.addWidget(self.inp_search)
        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("btn_search")
        self.btn_search.setFixedWidth(60)
        self.btn_search.setCursor(Qt.PointingHandCursor)
        self.btn_search.clicked.connect(self._on_search)
        search_row.addWidget(self.btn_search)
        lay.addLayout(search_row)

        self.txt_search = QTextEdit()
        self.txt_search.setReadOnly(True)
        self.txt_search.setFixedHeight(110)
        self.txt_search.setPlaceholderText("Search result shows here…")
        lay.addWidget(self.txt_search)

        lay.addStretch()
        parent_lay.addWidget(sidebar)

    # ── Main Content (Tabs) ──────────────────────────────────────────────────

    def _build_main(self, parent_lay):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("")  # use global QSS
        self.tabs.setDocumentMode(False)

        self._build_tab_recs()
        self._build_tab_graph()
        self._build_tab_adj()

        parent_lay.addWidget(self.tabs, 1)

    def _build_tab_recs(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {P['bg_panel']};")
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(14)

        # BFS info bar
        self.lbl_bfs = QLabel("")
        self.lbl_bfs.setFont(QFont("Segoe UI", 10))
        self.lbl_bfs.setStyleSheet(f"""
            color: {P['orange']};
            background: #1A1400;
            border: 1px solid #3A2A00;
            border-radius: 7px;
            padding: 7px 14px;
        """)
        self.lbl_bfs.setVisible(False)
        lay.addWidget(self.lbl_bfs)

        # Similar Users
        lay.addWidget(SectionHeader("👥  SIMILAR USERS  "))

        # Table header
        sim_hdr = QFrame()
        sim_hdr.setStyleSheet(f"background: {P['bg_deep']}; border-radius: 6px;")
        sim_hdr_lay = QHBoxLayout(sim_hdr)
        sim_hdr_lay.setContentsMargins(12, 7, 14, 7)
        sim_hdr_lay.setSpacing(12)
        for txt, w in [("Rank",22), ("User ID",70), ("Cosine Sim",65), ("Visualisasi",0), ("# Film",55)]:
            l = QLabel(txt)
            l.setFont(QFont("Segoe UI", 9, QFont.Bold))
            l.setStyleSheet(f"color: {P['cyan']}; border: none;")
            if w: l.setFixedWidth(w)
            else: l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            if txt == "# Film": l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sim_hdr_lay.addWidget(l)
        lay.addWidget(sim_hdr)

        # Similar users scrollable area
        sim_scroll = QScrollArea()
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setFrameShape(QFrame.NoFrame)
        sim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sim_scroll.setStyleSheet(f"background: transparent;")
        self.sim_container = QWidget()
        self.sim_container.setStyleSheet(f"background: transparent;")
        self.sim_lay = QVBoxLayout(self.sim_container)
        self.sim_lay.setContentsMargins(0, 0, 0, 0)
        self.sim_lay.setSpacing(2)
        self.sim_lay.addStretch()
        sim_scroll.setWidget(self.sim_container)
        sim_scroll.setFixedHeight(220)
        lay.addWidget(sim_scroll)

        # Recommendations
        lay.addWidget(SectionHeader("🎬  TOP MOVIE RECOMMENDATION  ", accent=P['yellow']))

        rec_scroll = QScrollArea()
        rec_scroll.setWidgetResizable(True)
        rec_scroll.setFrameShape(QFrame.NoFrame)
        rec_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rec_scroll.setStyleSheet("background: transparent;")
        self.rec_container = QWidget()
        self.rec_container.setStyleSheet("background: transparent;")
        self.rec_lay = QVBoxLayout(self.rec_container)
        self.rec_lay.setContentsMargins(0, 0, 0, 0)
        self.rec_lay.setSpacing(4)
        self.rec_lay.addStretch()
        rec_scroll.setWidget(self.rec_container)
        lay.addWidget(rec_scroll, 1)

        # Placeholder texts
        self._sim_placeholder = QLabel("Press Find Recommendation To Start.")
        self._sim_placeholder.setFont(QFont("Segoe UI", 11))
        self._sim_placeholder.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        self._sim_placeholder.setAlignment(Qt.AlignCenter)
        self.sim_lay.insertWidget(0, self._sim_placeholder)

        self._rec_placeholder = QLabel("Daftar rekomendasi akan tampil di sini setelah generate.")
        self._rec_placeholder.setFont(QFont("Segoe UI", 11))
        self._rec_placeholder.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        self._rec_placeholder.setAlignment(Qt.AlignCenter)
        self.rec_lay.insertWidget(0, self._rec_placeholder)

        self.tabs.addTab(tab, "📋  Rekomendasi")

    def _build_tab_graph(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {P['bg_panel']};")
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        lay.addWidget(SectionHeader("🕸️  GRAPH VISUALIZATION  —  User-Movie"))

        info = QLabel("Users (kiri)  ↔  Movies (kanan)  ·  Ketebalan edge = nilai rating  ·  Bintang = direkomendasikan")
        info.setFont(QFont("Segoe UI", 9))
        info.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        lay.addWidget(info)

        self.graph_canvas = GraphCanvas()
        self.graph_canvas.setStyleSheet(f"border-radius: 10px; border: 1px solid {P['border']};")
        lay.addWidget(self.graph_canvas, 1)

        # Legend chips
        leg_row = QHBoxLayout()
        for txt, clr in [
            ("● Target User",     P['yellow']),
            ("● Similar User",    P['pink']),
            ("★ Direkomendasikan", P['cyan']),
            ("■ Sudah Ditonton",  P['txt_m']),
            ("── Rating (bobot)", P['border']),
        ]:
            l = QLabel(txt)
            l.setFont(QFont("Segoe UI", 9))
            l.setStyleSheet(f"color: {clr}; border: none;")
            leg_row.addWidget(l)
        leg_row.addStretch()
        lay.addLayout(leg_row)

        self.tabs.addTab(tab, "🕸️  Graf Visualisasi")

    def _build_tab_adj(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {P['bg_panel']};")
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)

        lay.addWidget(SectionHeader("🔗  ADJACENCY LIST  (User → Movie Ratings)"))

        info = QLabel("Menampilkan daftar koneksi user target beserta bobot rating dan similar users terdekat.")
        info.setFont(QFont("Segoe UI", 9))
        info.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        lay.addWidget(info)

        self.txt_adj = QTextEdit()
        self.txt_adj.setReadOnly(True)
        self.txt_adj.setFont(QFont("Cascadia Code", 11))
        self.txt_adj.setStyleSheet(f"""
            QTextEdit {{
                background: {P['bg_card']};
                border: 1px solid {P['border']};
                border-radius: 10px;
                color: {P['cyan']};
                font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
                font-size: 12px;
                padding: 14px;
                line-height: 1.6;
            }}
        """)
        self.txt_adj.setPlaceholderText(
            "Adjacency list akan ditampilkan setelah Generate Recommendation…\n\n"
            "Format:\n  User X:\n  ├── Movie Y  [rating★]  Judul Film\n  └── …"
        )
        lay.addWidget(self.txt_adj, 1)

        self.tabs.addTab(tab, "🔗  Adjacency List")

    # ── Status Bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = QFrame()
        bar.setFixedHeight(32)
        bar.setStyleSheet(f"""
            QFrame {{
                background: {P['bg_deep']};
                border-top: 1px solid {P['border']};
            }}
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        self.dot = QLabel("●")
        self.dot.setFont(QFont("Segoe UI", 10))
        self.dot.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        lay.addWidget(self.dot)

        self.lbl_status = QLabel("Memuat data…")
        self.lbl_status.setFont(QFont("Segoe UI", 10))
        self.lbl_status.setStyleSheet(f"color: {P['txt_b']}; border: none;")
        lay.addWidget(self.lbl_status, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setFixedHeight(5)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        lay.addWidget(self.progress)

        ver = QLabel("v2.0  |  PyQt5")
        ver.setFont(QFont("Segoe UI", 9))
        ver.setStyleSheet(f"color: {P['txt_m']}; border: none;")
        lay.addWidget(ver)

        self._root_lay.addWidget(bar)

    # ── Progress Animations ───────────────────────────────────────────────────

    def _start_progress(self):
        self._prog_val = 0
        self._prog_timer.start(60)

    def _stop_progress(self):
        self._prog_timer.stop()
        self.progress.setValue(100)
        QTimer.singleShot(800, lambda: self.progress.setValue(0))

    def _tick_progress(self):
        self._prog_val = min(self._prog_val + 3, 90)
        self.progress.setValue(self._prog_val)

    def _set_status(self, text, dot_color=None):
        self.lbl_status.setText(text)
        if dot_color:
            self.dot.setStyleSheet(f"color: {dot_color}; border: none;")

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        self._start_progress()
        self._set_status("Memuat dataset MovieLens…", P['yellow'])
        self._thread_load = LoaderThread()
        self._thread_load.status.connect(lambda s: self._set_status(s, P['yellow']))
        self._thread_load.done.connect(self._on_load_done)
        self._thread_load.error.connect(self._on_error)
        self._thread_load.start()

    def _on_load_done(self, data):
        self.graph, self.bfs, self.cf = data
        self.is_loaded = True
        self._stop_progress()
        s = self.graph.get_stats()
        self.stat_users.set_value(s['total_users'])
        self.stat_movies.set_value(s['total_movies'])
        self.stat_edges.set_value(s['total_edges'])
        self.stat_avg.set_value(f"{s['avg_ratings_per_user']:.1f}")
        sample = sorted(list(self.graph.users))[:8]
        self.lbl_hint.setText(f"Contoh ID: {', '.join(map(str, sample))}…")
        self._set_status(
            f"✅  Graph siap  —  {s['total_users']} users  ·  {s['total_movies']} movies  ·  {s['total_edges']} edges",
            P['green']
        )

    # ── Generate ──────────────────────────────────────────────────────────────

    def _on_generate(self):
        if not self.is_loaded:
            self._set_status("⚠  Data masih dimuat, tunggu sebentar.", P['yellow'])
            return
        raw = self.inp_uid.text().strip()
        try:
            uid = int(raw)
        except ValueError:
            self._set_status(f"❌  '{raw}' bukan angka yang valid.", P['red'])
            return
        if not self.graph.has_user(uid):
            avail = sorted(list(self.graph.users))[:10]
            self._set_status(f"❌  User {uid} tidak ada di graph. Coba: {avail}…", P['red'])
            return

        self.cur_uid = uid
        self.btn_generate.setEnabled(False)
        self._start_progress()
        self._set_status(f"Memproses rekomendasi untuk User {uid}…", P['yellow'])

        self._thread_rec = RecommendThread(self.graph, self.bfs, self.cf, uid)
        self._thread_rec.status.connect(lambda s: self._set_status(s, P['yellow']))
        self._thread_rec.done.connect(self._on_rec_done)
        self._thread_rec.error.connect(self._on_error)
        self._thread_rec.start()

    def _on_rec_done(self, result):
        self.btn_generate.setEnabled(True)
        self._stop_progress()

        uid         = result['uid']
        sim_scores  = result['sim_scores']
        recs        = result['recs']
        visited     = result['visited']
        candidates  = result['candidates']
        self.cur_recs = recs
        self.cur_sims = sim_scores

        # ── Render Similar Users tab ────────────────────────────────────────
        for w in [self._sim_placeholder] + [self.sim_lay.itemAt(i).widget()
                   for i in range(self.sim_lay.count())
                   if self.sim_lay.itemAt(i) and self.sim_lay.itemAt(i).widget()]:
            if w: w.setParent(None)

        for i, (su, sim) in enumerate(sim_scores[:8]):
            n_films = len(self.graph.get_user_movies(su))
            row = SimUserRow(i+1, su, sim, n_films)
            self.sim_lay.insertWidget(i, row)

        # ── Render Recommendations ──────────────────────────────────────────
        for w in [self._rec_placeholder] + [self.rec_lay.itemAt(i).widget()
                   for i in range(self.rec_lay.count())
                   if self.rec_lay.itemAt(i) and self.rec_lay.itemAt(i).widget()]:
            if w: w.setParent(None)

        for i, rec in enumerate(recs):
            card = RecCard(i+1, rec['title'], rec['genres'],
                           rec['score'], rec['rated_by'])
            self.rec_lay.insertWidget(i, card)

        # ── BFS Info banner ─────────────────────────────────────────────────
        n_rated = len(self.graph.get_user_movies(uid))
        self.lbl_bfs.setText(
            f"📡  BFS: {len(visited)} node dikunjungi  |  "
            f"{len(result['sim_raw'])} similar users  |  "
            f"{len(candidates)} kandidat film  |  "
            f"User {uid} menilai {n_rated} film"
        )
        self.lbl_bfs.setVisible(True)

        # ── Adjacency List ──────────────────────────────────────────────────
        adj = self.graph.get_adjacency_list_str(uid, limit=15)
        if sim_scores:
            adj += "\n\n── Similar Users (BFS Level-2) ─────────────────────\n"
            for su, sim in sim_scores[:6]:
                adj += f"  User {su:<6}  similarity = {sim:.4f}\n"
        self.txt_adj.setPlainText(adj)

        # ── Graph Visualization ─────────────────────────────────────────────
        self.graph_canvas.draw_graph(self.graph, uid, sim_scores, recs)

        self._set_status(
            f"✅  User {uid}  —  {len(sim_scores)} similar users  ·  {len(recs)} rekomendasi dihasilkan",
            P['green']
        )

        # Switch ke tab rekomendasi
        self.tabs.setCurrentIndex(0)

    def _on_error(self, msg):
        self.btn_generate.setEnabled(True)
        self._stop_progress()
        self._set_status(f"❌  Error: {msg}", P['red'])

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _on_reset(self):
        self.cur_uid = None
        self.cur_recs = []
        self.cur_sims = []

        # Clear sim rows
        for i in range(self.sim_lay.count()):
            item = self.sim_lay.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self.sim_lay.addWidget(self._sim_placeholder)
        self._sim_placeholder.setVisible(True)

        # Clear rec rows
        for i in range(self.rec_lay.count()):
            item = self.rec_lay.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self.rec_lay.addWidget(self._rec_placeholder)
        self._rec_placeholder.setVisible(True)

        self.lbl_bfs.setVisible(False)
        self.txt_adj.clear()
        self.txt_search.clear()
        self.inp_uid.setText("1")
        self.inp_search.clear()
        self.graph_canvas._draw_empty()
        self._set_status("Reset selesai — masukkan User ID untuk memulai ulang.", P['cyan'])

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self):
        if not self.is_loaded:
            return
        raw = self.inp_uid.text().strip()
        try:
            uid = int(raw)
        except ValueError:
            self.txt_search.setPlainText("User ID harus angka.")
            return

        kw = self.inp_search.text().strip().lower()
        if not kw:
            top = self.graph.get_user_top_movies(uid, top_n=10)
            lines = [f"Top film User {uid}:\n"]
            for _, title, rating in top:
                stars = "★" * int(rating) + "☆" * (5 - int(rating))
                lines.append(f"  {stars}  ({rating:.1f})  {title}")
            self.txt_search.setPlainText("\n".join(lines))
            return

        user_movies = self.graph.get_user_movies(uid)
        results = []
        for mid, rating in user_movies.items():
            title = self.graph.movie_info.get(mid, {}).get('title', '')
            if kw in title.lower():
                stars = "★" * int(rating) + "☆" * (5 - int(rating))
                results.append(f"  {stars}  ({rating:.1f})  {title}")
        if results:
            self.txt_search.setPlainText(f"Ditemukan {len(results)} film:\n\n" + "\n".join(results))
        else:
            self.txt_search.setPlainText(f"Film '{kw}' tidak ditemukan untuk User {uid}.")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Abiyu Recommend")
    app.setApplicationDisplayName("Abiyu Recommend — Movie Recommender")

    # Set palette (dark base)
    pal = app.palette()
    pal.setColor(QPalette.Window,          QColor(P['bg_base']))
    pal.setColor(QPalette.WindowText,      QColor(P['txt_h']))
    pal.setColor(QPalette.Base,            QColor(P['bg_card']))
    pal.setColor(QPalette.AlternateBase,   QColor(P['bg_card2']))
    pal.setColor(QPalette.Text,            QColor(P['txt_h']))
    pal.setColor(QPalette.Button,          QColor(P['bg_card']))
    pal.setColor(QPalette.ButtonText,      QColor(P['txt_h']))
    pal.setColor(QPalette.Highlight,       QColor(P['cyan_dk']))
    pal.setColor(QPalette.HighlightedText, QColor("#07090F"))
    app.setPalette(pal)

    win = MovieRecommenderApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()