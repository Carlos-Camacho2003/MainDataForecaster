"""
UI Styles and Theme Configuration

Defines colors, fonts, and styling constants for a modern look.
Riopaila theme: White background with red accents.
"""

# Color Palette - Riopaila Theme (White + Red)
COLORS = {
    # Primary colors - White theme
    "bg_dark": "#ffffff",       # Main background - white
    "bg_medium": "#f5f5f5",     # Header/sections - light gray
    "bg_light": "#e8e8e8",      # Selected tabs - lighter gray
    "accent": "#c41e3a",        # Riopaila red
    "accent_hover": "#a01830",  # Darker red on hover
    
    # Text colors
    "text_primary": "#1a1a1a",      # Dark text for readability
    "text_secondary": "#4a4a4a",    # Secondary text - dark gray
    "text_muted": "#7a7a7a",        # Muted text - medium gray
    
    # Status colors
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8",
    
    # Button colors
    "btn_primary": "#c41e3a",       # Riopaila red
    "btn_primary_hover": "#a01830", # Darker red
    "btn_secondary": "#6c757d",     # Gray
    "btn_secondary_hover": "#545b62",
    "btn_disabled": "#cccccc",
    
    # Input colors
    "input_bg": "#ffffff",
    "input_border": "#ced4da",
    "input_focus": "#c41e3a",
    
    # Progress bar
    "progress_bg": "#e9ecef",
    "progress_fill": "#c41e3a",
}

# Font Configuration
FONTS = {
    "title": ("Segoe UI", 24, "bold"),
    "heading": ("Segoe UI", 16, "bold"),
    "subheading": ("Segoe UI", 12, "bold"),
    "body": ("Segoe UI", 10),
    "body_bold": ("Segoe UI", 10, "bold"),
    "small": ("Segoe UI", 9),
    "monospace": ("Consolas", 9),
    "button": ("Segoe UI", 10, "bold"),
}

# Padding and Spacing
PADDING = {
    "small": 5,
    "medium": 10,
    "large": 20,
    "xlarge": 30,
}

# Widget Dimensions
DIMENSIONS = {
    "button_width": 20,
    "button_height": 2,
    "entry_width": 30,
    "combobox_width": 25,
    "log_height": 15,
    "log_width": 80,
}
