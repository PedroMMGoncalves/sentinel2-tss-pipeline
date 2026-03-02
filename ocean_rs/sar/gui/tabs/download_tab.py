"""
Download & Credentials Tab for SAR Bathymetry Toolkit GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import logging

logger = logging.getLogger('ocean_rs')


def create_download_tab(gui, notebook):
    """Create the Download & Credentials tab."""
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Download & Credentials")

    # --- Credentials Section ---
    cred_frame = ttk.LabelFrame(frame, text="NASA Earthdata Login", padding="10")
    cred_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    cred_row1 = ttk.Frame(cred_frame)
    cred_row1.pack(fill=tk.X, pady=2)
    ttk.Label(cred_row1, text="Username:").pack(side=tk.LEFT)
    ttk.Entry(cred_row1, textvariable=gui.username_var, width=30).pack(side=tk.LEFT, padx=5)

    cred_row2 = ttk.Frame(cred_frame)
    cred_row2.pack(fill=tk.X, pady=2)
    ttk.Label(cred_row2, text="Password:").pack(side=tk.LEFT)
    ttk.Entry(cred_row2, textvariable=gui.password_var, width=30,
              show="*").pack(side=tk.LEFT, padx=5)

    cred_btn_frame = ttk.Frame(cred_frame)
    cred_btn_frame.pack(fill=tk.X, pady=5)
    ttk.Button(cred_btn_frame, text="Test Connection",
               command=lambda: _test_connection(gui)).pack(side=tk.LEFT, padx=2)
    ttk.Button(cred_btn_frame, text="Save to .env",
               command=lambda: _save_to_env(gui)).pack(side=tk.LEFT, padx=2)
    gui.connection_status = ttk.Label(cred_btn_frame, text="", style='Status.TLabel')
    gui.connection_status.pack(side=tk.LEFT, padx=10)

    # --- Download Directory ---
    dir_frame = ttk.LabelFrame(frame, text="Download Directory", padding="10")
    dir_frame.pack(fill=tk.X, padx=10, pady=5)

    dir_row = ttk.Frame(dir_frame)
    dir_row.pack(fill=tk.X)
    ttk.Entry(dir_row, textvariable=gui.download_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(dir_row, text="Browse...",
               command=lambda: _browse_download_dir(gui)).pack(side=tk.LEFT, padx=5)

    # --- Download Controls ---
    ctrl_frame = ttk.Frame(frame)
    ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
    gui.download_start_btn = ttk.Button(ctrl_frame, text="Start Download",
                                         style='Primary.TButton',
                                         command=lambda: _start_download(gui))
    gui.download_start_btn.pack(side=tk.LEFT, padx=2)
    gui.download_stop_btn = ttk.Button(ctrl_frame, text="Stop",
                                        style='Danger.TButton',
                                        state=tk.DISABLED,
                                        command=lambda: _stop_download(gui))
    gui.download_stop_btn.pack(side=tk.LEFT, padx=2)

    # --- Download Progress ---
    progress_frame = ttk.LabelFrame(frame, text="Download Progress", padding="10")
    progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    gui.download_progress_var = tk.DoubleVar(value=0)
    gui.download_progress_bar = ttk.Progressbar(progress_frame,
                                                  variable=gui.download_progress_var,
                                                  maximum=100)
    gui.download_progress_bar.pack(fill=tk.X, pady=5)
    gui.download_status_label = ttk.Label(progress_frame, text="Ready",
                                           style='Status.TLabel')
    gui.download_status_label.pack(anchor=tk.W)

    # Download log
    gui.download_log = tk.Text(progress_frame, height=8, font=('Consolas', 9),
                                state=tk.DISABLED)
    gui.download_log.pack(fill=tk.BOTH, expand=True, pady=5)

    return tab_index


def _test_connection(gui):
    """Test Earthdata credentials."""
    from ocean_rs.sar.download import CredentialManager
    creds = CredentialManager()
    username = gui.username_var.get().strip()
    password = gui.password_var.get().strip()
    if username and password:
        creds.set_credentials(username, password)

    success, msg = creds.test_connection()
    if success:
        gui.connection_status.config(text="\u2713 Connected", foreground='green')
    else:
        gui.connection_status.config(text="\u2717 Failed", foreground='red')
        messagebox.showwarning("Connection Failed", msg, parent=gui.root)


def _save_to_env(gui):
    """Save credentials to .env file."""
    username = gui.username_var.get().strip()
    password = gui.password_var.get().strip()
    if not username or not password:
        messagebox.showerror("Error", "Enter username and password first.",
                            parent=gui.root)
        return
    from ocean_rs.sar.download import CredentialManager
    creds = CredentialManager()
    creds.save_to_dotenv(username, password)
    messagebox.showinfo("Saved", "Credentials saved to .env file (gitignored).",
                       parent=gui.root)


def _browse_download_dir(gui):
    """Browse for download directory."""
    directory = filedialog.askdirectory(title="Select Download Directory")
    if directory:
        gui.download_dir_var.set(directory)


def _start_download(gui):
    """Start downloading selected scenes."""
    if not hasattr(gui, 'search_results') or not gui.search_results:
        messagebox.showerror("Error", "No search results. Run a search first.",
                            parent=gui.root)
        return

    selected_items = gui.results_tree.selection()
    if not selected_items:
        messagebox.showerror("Error", "No scenes selected.", parent=gui.root)
        return

    download_dir = gui.download_dir_var.get().strip()
    if not download_dir:
        messagebox.showerror("Error", "Select a download directory.", parent=gui.root)
        return

    # Get selected scenes
    selected_scenes = []
    for item in selected_items:
        idx = gui.results_tree.index(item)
        if idx < len(gui.search_results):
            selected_scenes.append(gui.search_results[idx])

    gui.selected_scenes = selected_scenes

    # Setup credentials
    from ocean_rs.sar.download import CredentialManager, BatchDownloader
    creds = CredentialManager()
    username = gui.username_var.get().strip()
    password = gui.password_var.get().strip()
    if username and password:
        creds.set_credentials(username, password)

    gui.downloader = BatchDownloader(creds)

    gui.download_active = True
    gui.download_start_btn.config(state=tk.DISABLED)
    gui.download_stop_btn.config(state=tk.NORMAL)

    def progress_callback(idx, total, msg):
        pct = (idx / total * 100) if total > 0 else 0
        gui.download_progress_var.set(pct)
        gui.download_status_label.config(text=msg)
        _log_download(gui, msg)

    def download_thread():
        try:
            paths = gui.downloader.download_scenes(
                selected_scenes, download_dir, progress_callback
            )
            gui.downloaded_paths = paths
            _log_download(gui, f"Download complete: {len(paths)} files")
        except Exception as e:
            _log_download(gui, f"Download error: {e}")
        finally:
            gui.download_active = False
            gui.root.after(0, lambda: gui.download_start_btn.config(state=tk.NORMAL))
            gui.root.after(0, lambda: gui.download_stop_btn.config(state=tk.DISABLED))

    gui.download_thread = threading.Thread(target=download_thread, daemon=True)
    gui.download_thread.start()


def _stop_download(gui):
    """Stop downloading."""
    if hasattr(gui, 'downloader'):
        gui.downloader.cancel()
    gui.download_status_label.config(text="Cancelling...")


def _log_download(gui, msg):
    """Append message to download log."""
    def _append():
        gui.download_log.config(state=tk.NORMAL)
        gui.download_log.insert(tk.END, msg + "\n")
        gui.download_log.see(tk.END)
        gui.download_log.config(state=tk.DISABLED)
    gui.root.after(0, _append)
