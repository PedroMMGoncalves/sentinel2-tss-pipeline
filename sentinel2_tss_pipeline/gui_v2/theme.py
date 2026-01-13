"""
Theme Manager for GUI v2.

Provides centralized styling with a professional color palette
and custom ttk styles for all widgets.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


class ThemeManager:
    """
    Manages GUI theming and styling.

    Provides a consistent, professional look across all widgets
    with a scientific/research-oriented color scheme.
    """

    # Color palette - professional scientific theme
    COLORS = {
        # Primary colors
        'primary': '#2563eb',           # Blue - main actions
        'primary_hover': '#1d4ed8',     # Darker blue
        'primary_light': '#dbeafe',     # Light blue background

        # Status colors
        'success': '#16a34a',           # Green - success/valid
        'success_light': '#dcfce7',     # Light green background
        'warning': '#d97706',           # Orange - warnings
        'warning_light': '#fef3c7',     # Light orange background
        'error': '#dc2626',             # Red - errors
        'error_light': '#fee2e2',       # Light red background
        'info': '#0891b2',              # Cyan - information
        'info_light': '#cffafe',        # Light cyan background

        # Neutral colors
        'bg_main': '#f8fafc',           # Main background
        'bg_card': '#ffffff',           # Card/section background
        'bg_hover': '#f1f5f9',          # Hover state
        'bg_active': '#e2e8f0',         # Active/pressed state

        # Text colors
        'text_primary': '#1e293b',      # Main text
        'text_secondary': '#64748b',    # Secondary text
        'text_muted': '#94a3b8',        # Muted/disabled text
        'text_inverse': '#ffffff',      # Text on dark backgrounds

        # Border colors
        'border': '#e2e8f0',            # Default border
        'border_focus': '#2563eb',      # Focused border
        'border_error': '#dc2626',      # Error border

        # Section colors (for visual grouping)
        'section_processing': '#eff6ff',    # Blue tint
        'section_parameters': '#f0fdf4',    # Green tint
        'section_output': '#fefce8',        # Yellow tint
        'section_advanced': '#faf5ff',      # Purple tint
    }

    # Font configuration
    FONTS = {
        'title': ('Segoe UI', 16, 'bold'),
        'subtitle': ('Segoe UI', 12, 'bold'),
        'heading': ('Segoe UI', 11, 'bold'),
        'body': ('Segoe UI', 10),
        'small': ('Segoe UI', 9),
        'mono': ('Consolas', 10),
    }

    # Icons (Unicode)
    ICONS = {
        'expand': '\u25BC',         # ▼ Down arrow
        'collapse': '\u25B6',       # ▶ Right arrow
        'check': '\u2713',          # ✓ Check mark
        'cross': '\u2717',          # ✗ X mark
        'warning': '\u26A0',        # ⚠ Warning
        'info': '\u2139',           # ℹ Info
        'folder': '\U0001F4C1',     # 📁 Folder
        'file': '\U0001F4C4',       # 📄 File
        'gear': '\u2699',           # ⚙ Gear
        'play': '\u25B6',           # ▶ Play
        'stop': '\u25A0',           # ■ Stop
        'success': '\u2705',        # ✅ Green check
        'error': '\u274C',          # ❌ Red X
    }

    def __init__(self, root=None):
        """
        Initialize theme manager.

        Args:
            root: Optional tk root window. If provided, styles are applied immediately.
        """
        self.style = None
        if root:
            self.apply(root)

    def apply(self, root):
        """
        Apply theme to the application.

        Args:
            root: tk root window
        """
        self.style = ttk.Style(root)

        # Try to use a modern base theme
        available_themes = self.style.theme_names()
        if 'clam' in available_themes:
            self.style.theme_use('clam')
        elif 'vista' in available_themes:
            self.style.theme_use('vista')

        self._configure_frame_styles()
        self._configure_label_styles()
        self._configure_button_styles()
        self._configure_entry_styles()
        self._configure_checkbox_styles()
        self._configure_radiobutton_styles()
        self._configure_progressbar_styles()
        self._configure_notebook_styles()
        self._configure_labelframe_styles()
        self._configure_scrollbar_styles()

        logger.info("Theme applied successfully")

    def _configure_frame_styles(self):
        """Configure frame styles."""
        self.style.configure(
            'TFrame',
            background=self.COLORS['bg_main']
        )
        self.style.configure(
            'Card.TFrame',
            background=self.COLORS['bg_card'],
            relief='flat'
        )
        # Section-specific frames
        for section in ['processing', 'parameters', 'output', 'advanced']:
            self.style.configure(
                f'Section.{section.capitalize()}.TFrame',
                background=self.COLORS[f'section_{section}']
            )

    def _configure_label_styles(self):
        """Configure label styles."""
        self.style.configure(
            'TLabel',
            background=self.COLORS['bg_main'],
            foreground=self.COLORS['text_primary'],
            font=self.FONTS['body']
        )
        self.style.configure(
            'Title.TLabel',
            font=self.FONTS['title'],
            foreground=self.COLORS['text_primary']
        )
        self.style.configure(
            'Subtitle.TLabel',
            font=self.FONTS['subtitle'],
            foreground=self.COLORS['text_primary']
        )
        self.style.configure(
            'Heading.TLabel',
            font=self.FONTS['heading'],
            foreground=self.COLORS['text_primary']
        )
        self.style.configure(
            'Muted.TLabel',
            foreground=self.COLORS['text_muted'],
            font=self.FONTS['small']
        )
        # Status labels
        self.style.configure(
            'Success.TLabel',
            foreground=self.COLORS['success']
        )
        self.style.configure(
            'Warning.TLabel',
            foreground=self.COLORS['warning']
        )
        self.style.configure(
            'Error.TLabel',
            foreground=self.COLORS['error']
        )
        self.style.configure(
            'Info.TLabel',
            foreground=self.COLORS['info']
        )

    def _configure_button_styles(self):
        """Configure button styles."""
        # Default button
        self.style.configure(
            'TButton',
            font=self.FONTS['body'],
            padding=(10, 5)
        )
        # Primary button (main actions)
        self.style.configure(
            'Primary.TButton',
            font=self.FONTS['body'],
            padding=(12, 6)
        )
        self.style.map(
            'Primary.TButton',
            background=[
                ('active', self.COLORS['primary_hover']),
                ('!disabled', self.COLORS['primary'])
            ],
            foreground=[
                ('!disabled', self.COLORS['text_inverse'])
            ]
        )
        # Success button
        self.style.configure(
            'Success.TButton',
            padding=(12, 6)
        )
        self.style.map(
            'Success.TButton',
            background=[('!disabled', self.COLORS['success'])]
        )
        # Danger button
        self.style.configure(
            'Danger.TButton',
            padding=(12, 6)
        )
        self.style.map(
            'Danger.TButton',
            background=[('!disabled', self.COLORS['error'])]
        )

    def _configure_entry_styles(self):
        """Configure entry styles."""
        self.style.configure(
            'TEntry',
            font=self.FONTS['body'],
            padding=5
        )
        self.style.map(
            'TEntry',
            bordercolor=[
                ('focus', self.COLORS['border_focus']),
                ('!focus', self.COLORS['border'])
            ]
        )
        # Validation states
        self.style.configure(
            'Valid.TEntry',
            bordercolor=self.COLORS['success']
        )
        self.style.configure(
            'Invalid.TEntry',
            bordercolor=self.COLORS['error']
        )

    def _configure_checkbox_styles(self):
        """Configure checkbox styles."""
        self.style.configure(
            'TCheckbutton',
            font=self.FONTS['body'],
            background=self.COLORS['bg_main']
        )
        self.style.map(
            'TCheckbutton',
            background=[('active', self.COLORS['bg_hover'])]
        )

    def _configure_radiobutton_styles(self):
        """Configure radiobutton styles."""
        self.style.configure(
            'TRadiobutton',
            font=self.FONTS['body'],
            background=self.COLORS['bg_main']
        )
        self.style.map(
            'TRadiobutton',
            background=[('active', self.COLORS['bg_hover'])]
        )

    def _configure_progressbar_styles(self):
        """Configure progressbar styles."""
        self.style.configure(
            'TProgressbar',
            thickness=20,
            troughcolor=self.COLORS['bg_active'],
            background=self.COLORS['primary']
        )
        # Stage-specific progress bars
        self.style.configure(
            'Resampling.Horizontal.TProgressbar',
            background=self.COLORS['info']
        )
        self.style.configure(
            'C2RCC.Horizontal.TProgressbar',
            background=self.COLORS['primary']
        )
        self.style.configure(
            'TSS.Horizontal.TProgressbar',
            background=self.COLORS['success']
        )

    def _configure_notebook_styles(self):
        """Configure notebook (tab) styles."""
        self.style.configure(
            'TNotebook',
            background=self.COLORS['bg_main'],
            tabmargins=[2, 5, 2, 0]
        )
        self.style.configure(
            'TNotebook.Tab',
            font=self.FONTS['body'],
            padding=[12, 6],
            background=self.COLORS['bg_card']
        )
        self.style.map(
            'TNotebook.Tab',
            background=[
                ('selected', self.COLORS['bg_card']),
                ('!selected', self.COLORS['bg_active'])
            ],
            expand=[('selected', [1, 1, 1, 0])]
        )

    def _configure_labelframe_styles(self):
        """Configure labelframe styles."""
        self.style.configure(
            'TLabelframe',
            background=self.COLORS['bg_card'],
            relief='groove',
            borderwidth=1
        )
        self.style.configure(
            'TLabelframe.Label',
            font=self.FONTS['heading'],
            foreground=self.COLORS['text_primary'],
            background=self.COLORS['bg_card']
        )
        # Collapsible frame header
        self.style.configure(
            'Collapsible.TFrame',
            background=self.COLORS['bg_hover']
        )

    def _configure_scrollbar_styles(self):
        """Configure scrollbar styles."""
        self.style.configure(
            'TScrollbar',
            background=self.COLORS['bg_active'],
            troughcolor=self.COLORS['bg_main'],
            borderwidth=0,
            arrowsize=14
        )

    def get_color(self, name):
        """Get a color by name."""
        return self.COLORS.get(name, '#000000')

    def get_font(self, name):
        """Get a font by name."""
        return self.FONTS.get(name, self.FONTS['body'])

    def get_icon(self, name):
        """Get an icon by name."""
        return self.ICONS.get(name, '')
