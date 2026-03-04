"""
Theme Manager for SAR Bathymetry Toolkit GUI.

Ocean/teal palette distinct from optical's blue.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('ocean_rs')


class ThemeManager:
    """Manages SAR GUI theming and styling."""

    COLORS = {
        'primary': '#0e7490',
        'primary_hover': '#0c6478',
        'primary_light': '#cffafe',
        'success': '#16a34a',
        'success_light': '#dcfce7',
        'warning': '#d97706',
        'warning_light': '#fef3c7',
        'error': '#dc2626',
        'error_light': '#fee2e2',
        'info': '#0891b2',
        'info_light': '#cffafe',
        'bg_main': '#f0fdfa',
        'bg_card': '#ffffff',
        'bg_hover': '#f0f9ff',
        'bg_active': '#e0f2fe',
        'text_primary': '#1e293b',
        'text_secondary': '#64748b',
        'text_muted': '#94a3b8',
        'text_inverse': '#ffffff',
        'border': '#e2e8f0',
        'border_focus': '#0e7490',
        'border_error': '#dc2626',
        'section_search': '#ecfeff',
        'section_download': '#f0f9ff',
        'section_processing': '#f0fdf4',
        'section_results': '#fefce8',
    }

    FONTS = {
        'title': ('Calibri', 18, 'bold'),
        'subtitle': ('Calibri', 13, 'bold'),
        'heading': ('Calibri', 12, 'bold'),
        'body': ('Calibri', 11),
        'small': ('Calibri', 10),
        'mono': ('Consolas', 10),
    }

    ICONS = {
        'expand': '\u25BC',
        'collapse': '\u25B6',
        'check': '\u2713',
        'cross': '\u2717',
        'search': '\u2315',
        'download': '\u2B07',
        'process': '\u2699',
        'map': '\u2316',
    }

    def __init__(self, root):
        self.root = root
        self._configure_styles()

    def _configure_styles(self):
        style = ttk.Style()

        # Try modern Sun Valley (Azure) theme first
        try:
            import sv_ttk
            sv_ttk.set_theme("light")
            logger.info("Applied Sun Valley (Azure) light theme")
        except ImportError:
            style.theme_use('clam')
            self.root.configure(bg=self.COLORS['bg_main'])
            logger.info("sv-ttk not installed, using fallback theme")

        style.configure('Primary.TButton',
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['text_inverse'],
                        font=self.FONTS['body'],
                        padding=(15, 8))
        style.map('Primary.TButton',
                  background=[('active', self.COLORS['primary_hover'])])

        style.configure('Success.TButton',
                        background=self.COLORS['success'],
                        foreground=self.COLORS['text_inverse'],
                        font=self.FONTS['body'],
                        padding=(15, 8))

        style.configure('Danger.TButton',
                        background=self.COLORS['error'],
                        foreground=self.COLORS['text_inverse'],
                        font=self.FONTS['body'],
                        padding=(15, 8))

        style.configure('Card.TFrame', background=self.COLORS['bg_card'])
        style.configure('Main.TFrame', background=self.COLORS['bg_main'])

        style.configure('Title.TLabel',
                        background=self.COLORS['bg_main'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['title'])
        style.configure('Heading.TLabel',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['heading'])
        style.configure('Body.TLabel',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['body'])
        style.configure('Status.TLabel',
                        background=self.COLORS['bg_main'],
                        foreground=self.COLORS['text_secondary'],
                        font=self.FONTS['small'])

        style.configure('TLabelframe', background=self.COLORS['bg_card'])
        style.configure('TLabelframe.Label',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['heading'])

        style.configure('TNotebook', background=self.COLORS['bg_main'])
        style.configure('TNotebook.Tab',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['body'],
                        padding=(12, 6))
        style.map('TNotebook.Tab',
                  background=[('selected', self.COLORS['primary_light'])],
                  foreground=[('selected', self.COLORS['primary'])])

        style.configure('TProgressbar',
                        background=self.COLORS['primary'],
                        troughcolor=self.COLORS['bg_main'])

        style.configure('Treeview',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['body'],
                        rowheight=25)
        style.configure('Treeview.Heading',
                        background=self.COLORS['primary_light'],
                        foreground=self.COLORS['primary'],
                        font=self.FONTS['heading'])
