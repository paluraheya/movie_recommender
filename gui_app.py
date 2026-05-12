# ============================================================================
# gui_app.py — Antarmuka GUI Tkinter
# ============================================================================
#
# Layout tiga kolom:
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  HEADER: Judul Aplikasi                                             │
#   ├──────────────────┬──────────────────────────┬───────────────────────┤
#   │  LEFT (280px)    │  CENTER (expand)         │  RIGHT (350px)        │
#   │  ─ Stats Graph   │  ─ Tabel Similar Users   │  ─ Visualisasi Graph  │
#   │  ─ Input User ID │  ─ BFS Info              │    (Matplotlib)       │
#   │  ─ Tombol Aksi   │  ─ Tabel Top-5 Rekom.   │                       │
#   │  ─ Cari Film     │                          │                       │
#   │  ─ Adj. List     │                          │                       │
#   ├──────────────────┴──────────────────────────┴───────────────────────┤
#   │  STATUS BAR                                                         │
#   └─────────────────────────────────────────────────────────────────────┘
#
# ============================================================================
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PANDUAN KUSTOMISASI TAMPILAN                                          ║
# ║  Cari komentar "=== STYLING ===" untuk menemukan bagian yang bisa      ║
# ║  diubah tampilannya.                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from graph_ds    import BipartiteGraph
from algorithms  import BFSTraversal, CollaborativeFilter
from data_loader import DataLoader


class MovieRecommenderApp:
    """
    Kelas utama GUI Tkinter untuk Sistem Rekomendasi Film.

    Mengintegrasikan BipartiteGraph, BFSTraversal, dan CollaborativeFilter
    ke dalam antarmuka grafis yang interaktif.
    """

    # ════════════════════════════════════════════════════════════════════
    # === STYLING: KONFIGURASI WARNA =====================================
    # Ubah nilai hex di bawah untuk mengganti skema warna aplikasi.
    # ════════════════════════════════════════════════════════════════════

    # Background & Panel
    BG_COLOR      = "#1E1E2E"   # Background window utama
    FRAME_COLOR   = "#2A2A3E"   # Background panel/frame
    ENTRY_BG      = "#313244"   # Background input & tabel alternatif

    # Teks
    TEXT_COLOR    = "#CDD6F4"   # Teks utama
    TEXT_MUTED    = "#6C7086"   # Teks sekunder / redup

    # Aksen
    ACCENT        = "#CBA6F7"   # Ungu (Mauve) — header, border aktif
    ACCENT_DARK   = "#7C3AED"   # Ungu gelap — header background
    SUCCESS       = "#A6E3A1"   # Hijau — nilai baik / sukses
    WARNING       = "#FAB387"   # Oranye — informasi BFS
    GOLD          = "#F9E2AF"   # Kuning — rank 1

    # Tombol
    BTN_PRIMARY   = "#CBA6F7"   # Tombol utama (Generate)
    BTN_RESET     = "#45475A"   # Tombol reset
    BTN_SEARCH    = "#89B4FA"   # Tombol cari
    BTN_TEXT      = "#1E1E2E"   # Teks di atas tombol (gelap agar kontras)

    # Header tabel
    TBL_HEADER    = "#181825"   # Background baris header tabel

    # Highlight row
    ROW_ODD       = "#2A2A3E"
    ROW_EVEN      = "#313244"
    ROW_TOP       = "#3D2B5E"   # Highlight baris rekomendasi #1

    # ════════════════════════════════════════════════════════════════════
    # === STYLING: KONFIGURASI FONT ======================================
    # Format: ("Nama Font", ukuran, "bold"/"italic"/... [opsional])
    # ════════════════════════════════════════════════════════════════════

    FONT_APP_TITLE = ("Segoe UI", 16, "bold")   # Judul di header
    FONT_SUBTITLE  = ("Segoe UI",  9)            # Subtitle di header
    FONT_SECTION   = ("Segoe UI", 11, "bold")    # Judul setiap section
    FONT_NORMAL    = ("Segoe UI", 10)            # Teks normal
    FONT_SMALL     = ("Segoe UI",  9)            # Teks kecil
    FONT_MONO      = ("Consolas",  9)            # Adjacency list (monospace)
    FONT_INPUT     = ("Segoe UI", 13, "bold")    # Input user ID
    FONT_BTN_MAIN  = ("Segoe UI", 11, "bold")    # Tombol utama

    # ════════════════════════════════════════════════════════════════════
    # === STYLING: UKURAN WINDOW =========================================
    # ════════════════════════════════════════════════════════════════════

    WIN_W = 1250   # Lebar window (piksel)
    WIN_H =  780   # Tinggi window (piksel)

    # ════════════════════════════════════════════════════════════════════
    # === STYLING: UKURAN PANEL ==========================================
    # ════════════════════════════════════════════════════════════════════

    LEFT_W  = 285   # Lebar panel kiri (piksel)
    RIGHT_W = 370   # Lebar panel kanan / visualisasi (piksel)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistem Rekomendasi Film — Weighted Bipartite Graph")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.configure(bg=self.BG_COLOR)
        self.root.minsize(950, 620)

        # ── Komponen sistem ──────────────────────────────────────────────
        self.graph       = BipartiteGraph()
        self.bfs         = None   # inisialisasi setelah data dimuat
        self.cf          = None
        self.loader      = DataLoader()
        self.is_loaded   = False

        # ── State saat ini ───────────────────────────────────────────────
        self.cur_user_id    = None
        self.cur_recs       = []
        self.cur_sim_users  = []

        # ── Build UI ─────────────────────────────────────────────────────
        self._build_ui()

        # ── Muat data setelah UI siap ────────────────────────────────────
        self.root.after(400, self._load_data)

    # ════════════════════════════════════════════════════════════════════
    # UI BUILDER
    # ════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        """Membangun layout utama aplikasi."""
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

    def _build_header(self):
        # ════════════════════════════════════════════════════════════════
        # === STYLING: HEADER ============================================
        # ════════════════════════════════════════════════════════════════
        hdr = tk.Frame(self.root, bg=self.ACCENT_DARK, height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="🎬  SISTEM REKOMENDASI FILM BERBASIS GRAPH",
            font=self.FONT_APP_TITLE,
            bg=self.ACCENT_DARK, fg="#FFFFFF"
        ).pack(side=tk.LEFT, padx=18, pady=10)

        tk.Label(
            hdr,
            text="Weighted Bipartite Graph  •  BFS Traversal  •  Cosine Similarity  •  Collaborative Filtering",
            font=self.FONT_SUBTITLE,
            bg=self.ACCENT_DARK, fg="#DDD6FE"
        ).pack(side=tk.RIGHT, padx=18)

    def _build_main_area(self):
        """Tiga kolom utama."""
        container = tk.Frame(self.root, bg=self.BG_COLOR)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # ── Panel kiri ───────────────────────────────────────────────────
        self.pnl_left = tk.Frame(container, bg=self.FRAME_COLOR,
                                  width=self.LEFT_W)
        self.pnl_left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.pnl_left.pack_propagate(False)

        # ── Panel tengah ─────────────────────────────────────────────────
        self.pnl_center = tk.Frame(container, bg=self.FRAME_COLOR)
        self.pnl_center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                              padx=(0, 8))

        # ── Panel kanan ──────────────────────────────────────────────────
        self.pnl_right = tk.Frame(container, bg=self.FRAME_COLOR,
                                   width=self.RIGHT_W)
        self.pnl_right.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.pnl_right.pack_propagate(False)

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    # ────────────────────────────────────────────────────────────────────
    # Panel Kiri
    # ────────────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        p = tk.Frame(self.pnl_left, bg=self.FRAME_COLOR)
        p.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # ── Graph Statistics ─────────────────────────────────────────────
        self._section_label(p, "📊  GRAPH STATISTICS")

        stats_box = tk.Frame(p, bg=self.ENTRY_BG)
        stats_box.pack(fill=tk.X, pady=(4, 14))

        self.lbl_stats = tk.Label(
            stats_box, text="Memuat data…",
            font=self.FONT_SMALL,
            bg=self.ENTRY_BG, fg=self.TEXT_MUTED,
            justify=tk.LEFT, padx=10, pady=6
        )
        self.lbl_stats.pack(anchor=tk.W)

        # ── Input User ID ────────────────────────────────────────────────
        self._section_label(p, "👤  INPUT USER ID")

        # ════════════════════════════════════════════════════════════════
        # === STYLING: INPUT FIELD USER ID ===============================
        # ════════════════════════════════════════════════════════════════
        self.ent_uid = tk.Entry(
            p,
            font=self.FONT_INPUT,
            bg=self.ENTRY_BG, fg=self.TEXT_COLOR,
            insertbackground=self.TEXT_COLOR,
            relief=tk.FLAT, width=12,
            justify=tk.CENTER
        )
        self.ent_uid.pack(fill=tk.X, ipady=7, pady=(4, 2))
        self.ent_uid.insert(0, "1")

        self.lbl_uid_hint = tk.Label(
            p, text="ID tersedia: memuat…",
            font=self.FONT_SMALL,
            bg=self.FRAME_COLOR, fg=self.TEXT_MUTED
        )
        self.lbl_uid_hint.pack(anchor=tk.W, pady=(0, 10))

        # ── Tombol Generate ──────────────────────────────────────────────
        # ════════════════════════════════════════════════════════════════
        # === STYLING: TOMBOL GENERATE ===================================
        # ════════════════════════════════════════════════════════════════
        self.btn_gen = tk.Button(
            p, text="🚀  Generate Recommendation",
            font=self.FONT_BTN_MAIN,
            bg=self.BTN_PRIMARY, fg=self.BTN_TEXT,
            relief=tk.FLAT, padx=8, pady=9,
            cursor="hand2",
            activebackground=self.ACCENT, activeforeground=self.BTN_TEXT,
            command=self._on_generate
        )
        self.btn_gen.pack(fill=tk.X, pady=(0, 6))

        # ── Tombol Reset ─────────────────────────────────────────────────
        # ════════════════════════════════════════════════════════════════
        # === STYLING: TOMBOL RESET ======================================
        # ════════════════════════════════════════════════════════════════
        self.btn_reset = tk.Button(
            p, text="🔄  Reset",
            font=self.FONT_NORMAL,
            bg=self.BTN_RESET, fg=self.TEXT_COLOR,
            relief=tk.FLAT, padx=8, pady=6,
            cursor="hand2",
            activebackground="#585B70", activeforeground=self.TEXT_COLOR,
            command=self._on_reset
        )
        self.btn_reset.pack(fill=tk.X, pady=(0, 16))

        # ── Pencarian Film Favorit ───────────────────────────────────────
        self._section_label(p, "🔍  CARI FILM FAVORIT USER")

        search_row = tk.Frame(p, bg=self.FRAME_COLOR)
        search_row.pack(fill=tk.X, pady=(4, 4))

        # ════════════════════════════════════════════════════════════════
        # === STYLING: SEARCH INPUT ======================================
        # ════════════════════════════════════════════════════════════════
        self.ent_search = tk.Entry(
            search_row,
            font=self.FONT_NORMAL,
            bg=self.ENTRY_BG, fg=self.TEXT_MUTED,
            insertbackground=self.TEXT_COLOR,
            relief=tk.FLAT, width=17
        )
        self.ent_search.pack(side=tk.LEFT, ipady=5, padx=(0, 5))
        self.ent_search.insert(0, "ketik judul film…")
        self.ent_search.bind("<FocusIn>",  self._search_focus_in)
        self.ent_search.bind("<FocusOut>", self._search_focus_out)
        self.ent_search.bind("<Return>",   lambda _: self._on_search())

        # ════════════════════════════════════════════════════════════════
        # === STYLING: TOMBOL SEARCH =====================================
        # ════════════════════════════════════════════════════════════════
        tk.Button(
            search_row, text="Cari",
            font=self.FONT_SMALL,
            bg=self.BTN_SEARCH, fg=self.BTN_TEXT,
            relief=tk.FLAT, padx=8, pady=4,
            cursor="hand2",
            command=self._on_search
        ).pack(side=tk.LEFT)

        self.txt_search = scrolledtext.ScrolledText(
            p, height=5,
            font=self.FONT_SMALL,
            bg=self.ENTRY_BG, fg=self.TEXT_COLOR,
            relief=tk.FLAT, wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.txt_search.pack(fill=tk.X, pady=(2, 14))

        # ── Adjacency List ───────────────────────────────────────────────
        self._section_label(p, "🔗  ADJACENCY LIST (User → Movie)")

        self.txt_adj = scrolledtext.ScrolledText(
            p, height=8,
            font=self.FONT_MONO,
            bg=self.ENTRY_BG, fg=self.SUCCESS,
            relief=tk.FLAT, wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.txt_adj.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    # ────────────────────────────────────────────────────────────────────
    # Panel Tengah
    # ────────────────────────────────────────────────────────────────────

    def _build_center_panel(self):
        p = tk.Frame(self.pnl_center, bg=self.FRAME_COLOR)
        p.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # ── Similar Users ────────────────────────────────────────────────
        self._section_label(p, "👥  SIMILAR USERS  (BFS Depth-2 + Cosine Similarity)")

        # Header tabel
        sim_hdr_cols = [("Rank", 5), ("User ID", 8),
                        ("Cosine Sim", 14), ("Similarity Bar", 18), ("# Film Bersama", 14)]
        sim_hdr = tk.Frame(p, bg=self.TBL_HEADER)
        sim_hdr.pack(fill=tk.X, pady=(4, 0))
        for txt, w in sim_hdr_cols:
            tk.Label(sim_hdr, text=txt, font=self.FONT_SMALL,
                     bg=self.TBL_HEADER, fg=self.ACCENT,
                     width=w, anchor=tk.W, padx=6, pady=4).pack(side=tk.LEFT)

        # Body frame — akan diisi saat generate
        self.frm_sim = tk.Frame(p, bg=self.FRAME_COLOR)
        self.frm_sim.pack(fill=tk.X)
        self._placeholder(self.frm_sim, "Tekan 'Generate Recommendation' untuk memulai.")

        # ── BFS Info Label ───────────────────────────────────────────────
        self.lbl_bfs = tk.Label(
            p, text="",
            font=self.FONT_SMALL,
            bg=self.FRAME_COLOR, fg=self.WARNING,
            justify=tk.LEFT
        )
        self.lbl_bfs.pack(anchor=tk.W, pady=(6, 8))

        # ── Top-5 Rekomendasi ────────────────────────────────────────────
        self._section_label(p, "🎬  TOP 5 REKOMENDASI FILM  (Collaborative Filtering)")

        rec_hdr_cols = [("#", 3), ("Judul Film", 30), ("Genre", 22), ("Rec Score", 10), ("Dinilai", 7)]
        rec_hdr = tk.Frame(p, bg=self.TBL_HEADER)
        rec_hdr.pack(fill=tk.X, pady=(4, 0))
        for txt, w in rec_hdr_cols:
            tk.Label(rec_hdr, text=txt, font=self.FONT_SMALL,
                     bg=self.TBL_HEADER, fg=self.ACCENT,
                     width=w, anchor=tk.W, padx=6, pady=4).pack(side=tk.LEFT)

        # Body frame rekomendasi
        self.frm_rec = tk.Frame(p, bg=self.FRAME_COLOR)
        self.frm_rec.pack(fill=tk.BOTH, expand=True)
        self._placeholder(self.frm_rec, "Daftar rekomendasi akan muncul di sini.")

    # ────────────────────────────────────────────────────────────────────
    # Panel Kanan — Visualisasi Graph
    # ────────────────────────────────────────────────────────────────────

    def _build_right_panel(self):
        p = tk.Frame(self.pnl_right, bg=self.FRAME_COLOR)
        p.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self._section_label(p, "🕸️  GRAPH VISUALIZATION")
        tk.Label(
            p, text="Bipartite: User (kiri) ↔ Movie (kanan) | Ketebalan edge = rating",
            font=self.FONT_SMALL,
            bg=self.FRAME_COLOR, fg=self.TEXT_MUTED
        ).pack(anchor=tk.W, pady=(0, 6))

        # ════════════════════════════════════════════════════════════════
        # === STYLING: MATPLOTLIB FIGURE =================================
        # Ubah figsize untuk mengatur ukuran area grafik
        # ════════════════════════════════════════════════════════════════
        self.fig, self.ax = plt.subplots(figsize=(4.2, 5.2))
        self.fig.patch.set_facecolor(self.BG_COLOR)
        self.ax.set_facecolor("#181825")

        self.canvas = FigureCanvasTkAgg(self.fig, master=p)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_graph()

        # Legend
        legend = tk.Frame(p, bg=self.FRAME_COLOR)
        legend.pack(fill=tk.X, pady=(5, 0))
        for txt, clr in [("● Target User", "#F9E2AF"),
                         ("● Similar User", "#F38BA8"),
                         ("■ Movie", "#A6E3A1"),
                         ("─ Rating (bobot)", "#7F849C")]:
            tk.Label(legend, text=txt, font=self.FONT_SMALL,
                     bg=self.FRAME_COLOR, fg=clr).pack(side=tk.LEFT, padx=5)

    # ────────────────────────────────────────────────────────────────────
    # Status Bar
    # ────────────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        # ════════════════════════════════════════════════════════════════
        # === STYLING: STATUS BAR ========================================
        # ════════════════════════════════════════════════════════════════
        bar = tk.Frame(self.root, bg="#11111B", height=24)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self.lbl_status = tk.Label(
            bar, text="⏳ Memuat data…",
            font=self.FONT_SMALL,
            bg="#11111B", fg=self.TEXT_MUTED,
            anchor=tk.W, padx=10
        )
        self.lbl_status.pack(side=tk.LEFT, fill=tk.Y)

    # ════════════════════════════════════════════════════════════════════
    # HELPER UI
    # ════════════════════════════════════════════════════════════════════

    def _section_label(self, parent, text: str):
        # ════════════════════════════════════════════════════════════════
        # === STYLING: SECTION HEADER ====================================
        # ════════════════════════════════════════════════════════════════
        tk.Label(
            parent, text=text,
            font=self.FONT_SECTION,
            bg=self.FRAME_COLOR, fg=self.ACCENT
        ).pack(anchor=tk.W, pady=(0, 0))

    def _placeholder(self, parent, text: str):
        for w in parent.winfo_children():
            w.destroy()
        tk.Label(
            parent, text=text,
            font=self.FONT_SMALL,
            bg=self.FRAME_COLOR, fg=self.TEXT_MUTED,
            pady=10
        ).pack()

    def _set_text(self, widget, text: str):
        """Helper menulis ke ScrolledText."""
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    def _set_status(self, text: str):
        self.lbl_status.config(text=text)

    # ════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ════════════════════════════════════════════════════════════════════

    def _load_data(self):
        """Memuat dataset dan membangun graph (dipanggil setelah UI siap)."""
        try:
            self._set_status("⏳ Memuat dataset MovieLens…")
            self.root.update()

            self.loader.load_data()

            self._set_status("⏳ Membangun Weighted Bipartite Graph…")
            self.root.update()

            self.loader.build_graph(self.graph, max_users=20)
            self.bfs = BFSTraversal(self.graph)
            self.cf  = CollaborativeFilter(self.graph)

            # Update stats label
            s = self.graph.get_stats()
            self.lbl_stats.config(text=(
                f"  Total Users  : {s['total_users']}\n"
                f"  Total Movies : {s['total_movies']}\n"
                f"  Total Edges  : {s['total_edges']}\n"
                f"  Avg Rating/U : {s['avg_ratings_per_user']:.1f}\n"
                f"  Max Rating/U : {s['max_ratings_per_user']}"
            ), fg=self.TEXT_COLOR)

            # Tunjukkan beberapa ID user yang tersedia
            sample_ids = sorted(list(self.graph.users))[:8]
            self.lbl_uid_hint.config(
                text=f"Contoh ID: {', '.join(map(str, sample_ids))}…"
            )

            self.is_loaded = True
            self._set_status(
                f"✅ Graph siap — {s['total_users']} users, "
                f"{s['total_movies']} movies, {s['total_edges']} edges"
            )

        except Exception as exc:
            self._set_status(f"❌ Error: {exc}")
            messagebox.showerror("Error Memuat Data", str(exc))

    # ════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ════════════════════════════════════════════════════════════════════

    def _on_generate(self):
        """Handler tombol Generate Recommendation."""
        if not self.is_loaded:
            messagebox.showwarning("Tunggu", "Data masih dimuat, harap tunggu sebentar.")
            return

        raw = self.ent_uid.get().strip()
        try:
            user_id = int(raw)
        except ValueError:
            messagebox.showerror("Input Salah", f"'{raw}' bukan angka yang valid.")
            return

        if not self.graph.has_user(user_id):
            available = sorted(list(self.graph.users))[:10]
            messagebox.showerror(
                "User Tidak Ditemukan",
                f"User {user_id} tidak ada di graph.\nContoh ID yang tersedia: {available}…"
            )
            return

        self.cur_user_id = user_id
        self._set_status(f"⏳ Memproses rekomendasi untuk User {user_id}…")
        self.root.update()

        try:
            # ── Step 1: BFS Traversal ────────────────────────────────────
            similar_raw, visited = self.bfs.find_similar_users(user_id, max_depth=2)

            # ── Step 2: Cosine Similarity ────────────────────────────────
            sim_scores = self.cf.get_top_similar_users(
                user_id, similar_raw.keys(), top_k=10
            )
            self.cur_sim_users = sim_scores

            # ── Step 3: Candidate Movies via BFS ─────────────────────────
            candidates = self.bfs.get_candidate_movies(user_id, dict(sim_scores))

            # ── Step 4: Recommendation Score & Sort ──────────────────────
            recs = self.cf.get_recommendations(user_id, sim_scores, candidates, top_n=5)
            self.cur_recs = recs

            # ── Update UI ────────────────────────────────────────────────
            self._render_similar_users(sim_scores)
            self._render_recommendations(recs)
            self._update_adj_list(user_id)
            self._draw_graph(user_id, sim_scores[:3])

            # BFS summary
            n_rated = len(self.graph.get_user_movies(user_id))
            self.lbl_bfs.config(text=(
                f"📡 BFS Info: {len(visited)} node dikunjungi  |  "
                f"{len(similar_raw)} similar users ditemukan  |  "
                f"{len(candidates)} candidate movies  |  "
                f"User {user_id} telah menilai {n_rated} film"
            ))

            self._set_status(
                f"✅ Selesai — User {user_id}: "
                f"{len(sim_scores)} similar users, {len(recs)} rekomendasi"
            )

        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", str(exc))
            self._set_status(f"❌ Error: {exc}")

    def _on_reset(self):
        """Handler tombol Reset — kembalikan semua ke kondisi awal."""
        self._placeholder(self.frm_sim, "Tekan 'Generate Recommendation' untuk memulai.")
        self._placeholder(self.frm_rec, "Daftar rekomendasi akan muncul di sini.")
        self.lbl_bfs.config(text="")
        self._set_text(self.txt_adj, "")
        self._set_text(self.txt_search, "")
        self._draw_empty_graph()

        self.cur_user_id   = None
        self.cur_recs      = []
        self.cur_sim_users = []

        self.ent_uid.delete(0, tk.END)
        self.ent_uid.insert(0, "1")
        self.ent_search.delete(0, tk.END)
        self.ent_search.insert(0, "ketik judul film…")
        self.ent_search.config(fg=self.TEXT_MUTED)

        self._set_status("🔄 Reset selesai — masukkan User ID untuk memulai ulang.")

    def _on_search(self):
        """Handler pencarian film favorit user."""
        if not self.is_loaded:
            return

        raw = self.ent_uid.get().strip()
        try:
            user_id = int(raw)
        except ValueError:
            self._set_text(self.txt_search, "User ID harus angka.")
            return

        keyword = self.ent_search.get().strip().lower()
        if not keyword or keyword == "ketik judul film…":
            # Tampilkan semua film favorit jika tidak ada keyword
            top_movies = self.graph.get_user_top_movies(user_id, top_n=10)
            lines = [f"Top film User {user_id}:"]
            for _, title, rating in top_movies:
                stars = '★' * int(rating)
                lines.append(f"  {'★'*int(rating)}{'☆'*(5-int(rating))}  {title}")
            self._set_text(self.txt_search, "\n".join(lines))
            return

        # Cari film yang cocok
        user_movies = self.graph.get_user_movies(user_id)
        results = []
        for movie_id, rating in user_movies.items():
            title = self.graph.movie_info.get(movie_id, {}).get('title', '')
            if keyword in title.lower():
                stars = '★' * int(rating) + '☆' * (5 - int(rating))
                results.append(f"{stars} ({rating})  {title}")

        if results:
            self._set_text(self.txt_search,
                           f"Ditemukan {len(results)} film:\n" + "\n".join(results))
        else:
            self._set_text(self.txt_search,
                           f"Tidak ada film '{keyword}' untuk User {user_id}.")

    def _search_focus_in(self, _event):
        if self.ent_search.get() == "ketik judul film…":
            self.ent_search.delete(0, tk.END)
            self.ent_search.config(fg=self.TEXT_COLOR)

    def _search_focus_out(self, _event):
        if not self.ent_search.get():
            self.ent_search.insert(0, "ketik judul film…")
            self.ent_search.config(fg=self.TEXT_MUTED)

    # ════════════════════════════════════════════════════════════════════
    # RENDER TABEL
    # ════════════════════════════════════════════════════════════════════

    def _render_similar_users(self, sim_scores: list):
        """Render tabel similar users berdasarkan hasil BFS + cosine sim."""
        for w in self.frm_sim.winfo_children():
            w.destroy()

        if not sim_scores:
            self._placeholder(self.frm_sim, "Tidak ada similar users ditemukan.")
            return

        # ════════════════════════════════════════════════════════════════
        # === STYLING: BARIS TABEL SIMILAR USERS =========================
        # ════════════════════════════════════════════════════════════════
        for i, (uid, sim) in enumerate(sim_scores[:8]):
            bg = self.ROW_ODD if i % 2 == 0 else self.ROW_EVEN
            row = tk.Frame(self.frm_sim, bg=bg)
            row.pack(fill=tk.X)

            # Bar visual similarity (10 karakter)
            filled = int(sim * 10)
            bar = "█" * filled + "░" * (10 - filled)

            # Warna berdasarkan nilai similarity
            if sim >= 0.8:
                sim_clr = self.SUCCESS
            elif sim >= 0.5:
                sim_clr = self.WARNING
            else:
                sim_clr = self.TEXT_MUTED

            n_movies = len(self.graph.get_user_movies(uid))
            row_data = [
                (f"{i+1}", 5, self.TEXT_MUTED),
                (f"User {uid}", 8, self.TEXT_COLOR),
                (f"{sim:.4f}", 14, sim_clr),
                (bar, 18, sim_clr),
                (str(n_movies), 14, self.TEXT_MUTED),
            ]
            for txt, w, fg in row_data:
                tk.Label(row, text=txt, font=self.FONT_SMALL,
                         bg=bg, fg=fg, width=w,
                         anchor=tk.W, padx=6, pady=3).pack(side=tk.LEFT)

    def _render_recommendations(self, recs: list):
        """Render tabel top-5 rekomendasi."""
        for w in self.frm_rec.winfo_children():
            w.destroy()

        if not recs:
            self._placeholder(self.frm_rec, "Tidak ada rekomendasi — coba user lain.")
            return

        # ════════════════════════════════════════════════════════════════
        # === STYLING: BARIS TABEL REKOMENDASI ===========================
        # ════════════════════════════════════════════════════════════════
        RANK_ICONS = ["🥇", "🥈", "🥉", "4.", "5."]
        RANK_COLS  = [self.GOLD, "#C0C0C0", "#CD7F32",
                      self.TEXT_COLOR, self.TEXT_COLOR]

        for i, rec in enumerate(recs):
            bg = self.ROW_TOP if i == 0 else (
                self.ROW_ODD if i % 2 == 0 else self.ROW_EVEN
            )
            row = tk.Frame(self.frm_rec, bg=bg)
            row.pack(fill=tk.X, pady=1)

            title  = rec['title'][:36]  + ("…" if len(rec['title'])  > 36 else "")
            genres = rec['genres'][:24] + ("…" if len(rec['genres']) > 24 else "")

            row_data = [
                (RANK_ICONS[i] if i < 5 else f"{i+1}.", 3,  RANK_COLS[i]),
                (title,                                  30, self.TEXT_COLOR),
                (genres,                                 22, self.TEXT_MUTED),
                (f"{rec['score']:.3f}",                  10, self.SUCCESS),
                (f"{rec['rated_by']} user",               7, self.TEXT_MUTED),
            ]
            for txt, w, fg in row_data:
                tk.Label(row, text=txt, font=self.FONT_SMALL,
                         bg=bg, fg=fg, width=w,
                         anchor=tk.W, padx=6, pady=5).pack(side=tk.LEFT)

    def _update_adj_list(self, user_id: int):
        """Perbarui tampilan adjacency list."""
        adj_str = self.graph.get_adjacency_list_str(user_id, limit=10)

        if self.cur_sim_users:
            adj_str += "\n\n── Similar Users (BFS Level-2) ────"
            for uid, sim in self.cur_sim_users[:5]:
                adj_str += f"\n  User {uid:<4}  sim = {sim:.4f}"

        self._set_text(self.txt_adj, adj_str)

    # ════════════════════════════════════════════════════════════════════
    # GRAPH VISUALIZATION (Matplotlib)
    # ════════════════════════════════════════════════════════════════════

    def _draw_empty_graph(self):
        """Gambar placeholder saat belum ada data."""
        self.ax.clear()
        self.ax.set_facecolor("#181825")
        self.ax.text(0.5, 0.5,
                     "Graph akan muncul\nsetelah generate\nrekomendasi",
                     ha='center', va='center',
                     fontsize=11, color=self.TEXT_MUTED,
                     transform=self.ax.transAxes)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()

    def _draw_graph(self, target_uid: int, sim_scores: list):
        """
        Menggambar visualisasi Bipartite Graph secara manual
        menggunakan Matplotlib (tanpa library NetworkX).

        Layout:
          - User nodes  : sumbu X = 0.18  (sisi kiri)
          - Movie nodes : sumbu X = 0.82  (sisi kanan)
          - Edge        : garis dengan ketebalan proporsional terhadap rating

        Kompleksitas render: O(U × M)
        """
        self.ax.clear()
        # ════════════════════════════════════════════════════════════════
        # === STYLING: GRAPH VISUALIZATION ===============================
        # Ubah warna node, edge, dan background di bawah ini.
        # ════════════════════════════════════════════════════════════════

        # Warna
        C_BG      = "#181825"   # Background area grafik
        C_TARGET  = "#F9E2AF"   # Warna node target user (kuning)
        C_SIMILAR = "#F38BA8"   # Warna node similar user (merah muda)
        C_MOVIE   = "#A6E3A1"   # Warna node movie (hijau)
        C_EDGE_HI = "#CBA6F7"   # Warna edge rating tinggi (≥4)
        C_EDGE_LO = "#45475A"   # Warna edge rating rendah
        C_LABEL   = "#BAC2DE"   # Warna label teks

        self.ax.set_facecolor(C_BG)
        self.fig.patch.set_facecolor(self.BG_COLOR)

        # ── Kumpulkan nodes yang akan ditampilkan ────────────────────────
        display_users = [target_uid] + [uid for uid, _ in sim_scores]
        all_movies = set()
        for uid in display_users:
            for mid in list(self.graph.get_user_movies(uid).keys())[:6]:
                all_movies.add(mid)
        movie_list = list(all_movies)[:14]   # Max 14 movie

        # ── Hitung posisi node ───────────────────────────────────────────
        n_u = len(display_users)
        n_m = len(movie_list)
        X_USER  = 0.18
        X_MOVIE = 0.82

        user_pos  = {}
        for i, uid in enumerate(display_users):
            y = (0.9 / max(n_u - 1, 1)) * i + 0.05 if n_u > 1 else 0.5
            user_pos[uid] = (X_USER, y)

        movie_pos = {}
        for i, mid in enumerate(movie_list):
            y = (0.9 / max(n_m - 1, 1)) * i + 0.05 if n_m > 1 else 0.5
            movie_pos[mid] = (X_MOVIE, y)

        # ── Gambar edges ─────────────────────────────────────────────────
        for uid in display_users:
            ux, uy = user_pos[uid]
            u_movies = self.graph.get_user_movies(uid)
            for mid in movie_list:
                if mid in u_movies:
                    mx, my  = movie_pos[mid]
                    rating  = u_movies[mid]
                    lw      = 0.4 + (rating / 5) * 2.2
                    ec      = C_EDGE_HI if rating >= 4 else C_EDGE_LO
                    alpha   = 0.35 + (rating / 5) * 0.45
                    self.ax.plot([ux, mx], [uy, my], '-',
                                 color=ec, linewidth=lw, alpha=alpha, zorder=1)
                    # Label bobot di tengah edge
                    self.ax.text((ux + mx) / 2, (uy + my) / 2,
                                 f"{rating:.1f}",
                                 fontsize=5.5, color="#6C7086",
                                 ha='center', va='center', zorder=3)

        # ── Gambar node User ─────────────────────────────────────────────
        for uid, (x, y) in user_pos.items():
            is_target = (uid == target_uid)
            color  = C_TARGET  if is_target else C_SIMILAR
            size   = 240       if is_target else 160
            sim_lbl = ""
            if not is_target:
                sim_val = next((s for u, s in sim_scores if u == uid), None)
                sim_lbl = f"\n{sim_val:.2f}" if sim_val is not None else ""

            self.ax.scatter(x, y, s=size, c=color, zorder=5,
                            edgecolors='white', linewidths=0.6)
            self.ax.text(x - 0.09, y,
                         f"U{uid}" + ("\n(target)" if is_target else sim_lbl),
                         fontsize=7, color=color,
                         ha='center', va='center', zorder=6,
                         fontweight='bold' if is_target else 'normal')

        # ── Gambar node Movie ─────────────────────────────────────────────
        for mid, (x, y) in movie_pos.items():
            title = self.graph.movie_info.get(mid, {}).get('title', f'M{mid}')
            label = title[:18] + "…" if len(title) > 18 else title
            self.ax.scatter(x, y, s=70, c=C_MOVIE, zorder=5,
                            marker='s', edgecolors='white', linewidths=0.3)
            self.ax.text(x + 0.05, y, label,
                         fontsize=5.8, color=C_MOVIE,
                         ha='left', va='center', zorder=6)

        # ── Label kolom ───────────────────────────────────────────────────
        self.ax.text(X_USER,  1.01, "USERS",  fontsize=8, color=C_TARGET,
                     ha='center', fontweight='bold', transform=self.ax.transAxes)
        self.ax.text(X_MOVIE, 1.01, "MOVIES", fontsize=8, color=C_MOVIE,
                     ha='center', fontweight='bold', transform=self.ax.transAxes)

        # Garis pemisah tengah
        self.ax.axvline(x=0.5, color='#313244', linestyle='--', alpha=0.4, lw=0.8)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.ax.set_title(
            f"Bipartite Graph — User {target_uid}  ({n_u} users, {n_m} movies)",
            fontsize=8, color=C_LABEL, pad=6
        )
        self.canvas.draw()
