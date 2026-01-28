"""
Custom Styled Widgets for IDC_RIOP UI

Provides themed widgets with a modern industrial look.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

from ui.styles import COLORS, FONTS, PADDING, DIMENSIONS


class StyledButton(tk.Button):
    """A modern styled button with hover effects."""
    
    def __init__(
        self,
        parent,
        text: str,
        command: Optional[Callable] = None,
        style: str = "primary",
        width: int = None,
        **kwargs
    ):
        # Determine colors based on style
        if style == "primary":
            bg = COLORS["btn_primary"]
            hover_bg = COLORS["btn_primary_hover"]
            fg = "#ffffff"  # White text on red button
        elif style == "secondary":
            bg = COLORS["btn_secondary"]
            hover_bg = COLORS["btn_secondary_hover"]
            fg = "#ffffff"  # White text on gray button
        else:
            bg = COLORS["btn_primary"]
            hover_bg = COLORS["btn_primary_hover"]
            fg = "#ffffff"
        
        super().__init__(
            parent,
            text=text,
            command=command,
            font=FONTS["button"],
            bg=bg,
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief=tk.FLAT,
            cursor="hand2",
            width=width or DIMENSIONS["button_width"],
            pady=8,
            **kwargs
        )
        
        self._bg = bg
        self._hover_bg = hover_bg
        self._fg = fg
        self._enabled = True
        
        # Bind hover events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, event):
        if self._enabled:
            self.configure(bg=self._hover_bg)
    
    def _on_leave(self, event):
        if self._enabled:
            self.configure(bg=self._bg)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the button."""
        self._enabled = enabled
        if enabled:
            self.configure(
                state=tk.NORMAL,
                bg=self._bg,
                fg=self._fg,
                cursor="hand2"
            )
        else:
            self.configure(
                state=tk.DISABLED,
                bg=COLORS["btn_disabled"],
                fg="#888888",
                cursor="arrow"
            )


class StyledLabel(tk.Label):
    """A styled label with consistent theming."""
    
    def __init__(
        self,
        parent,
        text: str,
        style: str = "body",
        **kwargs
    ):
        font_map = {
            "title": FONTS["title"],
            "heading": FONTS["heading"],
            "subheading": FONTS["subheading"],
            "body": FONTS["body"],
            "body_bold": FONTS["body_bold"],
            "small": FONTS["small"],
        }
        
        color_map = {
            "title": COLORS["text_primary"],
            "heading": COLORS["text_primary"],
            "subheading": COLORS["text_primary"],
            "body": COLORS["text_secondary"],
            "body_bold": COLORS["text_primary"],
            "small": COLORS["text_muted"],
        }
        
        super().__init__(
            parent,
            text=text,
            font=font_map.get(style, FONTS["body"]),
            bg=kwargs.pop("bg", COLORS["bg_dark"]),
            fg=kwargs.pop("fg", color_map.get(style, COLORS["text_secondary"])),
            **kwargs
        )


class StyledFrame(tk.Frame):
    """A styled frame with consistent theming."""
    
    def __init__(self, parent, style: str = "dark", **kwargs):
        bg_map = {
            "dark": COLORS["bg_dark"],
            "medium": COLORS["bg_medium"],
            "light": COLORS["bg_light"],
        }
        
        super().__init__(
            parent,
            bg=bg_map.get(style, COLORS["bg_dark"]),
            **kwargs
        )


class StyledLabelFrame(tk.LabelFrame):
    """A styled label frame for grouping related controls."""
    
    def __init__(self, parent, text: str, **kwargs):
        super().__init__(
            parent,
            text=text,
            font=FONTS["subheading"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            padx=PADDING["medium"],
            pady=PADDING["medium"],
            **kwargs
        )


class StyledCombobox(ttk.Combobox):
    """A styled combobox (dropdown) widget."""
    
    def __init__(self, parent, values: list, **kwargs):
        super().__init__(
            parent,
            values=values,
            state="readonly",
            width=kwargs.pop("width", DIMENSIONS["combobox_width"]),
            **kwargs
        )
        if values:
            self.current(0)


class StyledCheckbutton(tk.Checkbutton):
    """A styled checkbutton widget."""
    
    def __init__(self, parent, text: str, variable: tk.BooleanVar, **kwargs):
        super().__init__(
            parent,
            text=text,
            variable=variable,
            font=FONTS["body"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_secondary"],
            selectcolor=COLORS["bg_medium"],
            activebackground=COLORS["bg_dark"],
            activeforeground=COLORS["text_primary"],
            cursor="hand2",
            **kwargs
        )


class ScrollableFrame(tk.Frame):
    """A scrollable frame container for content that may overflow.
    
    Use this when you have content that might not fit in the available space,
    especially on smaller screens or laptops.
    
    Usage:
        scrollable = ScrollableFrame(parent)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        # Add content to scrollable.interior (not to scrollable directly)
        label = tk.Label(scrollable.interior, text="Content here")
        label.pack()
    """
    
    def __init__(self, parent, bg=None, **kwargs):
        if bg is None:
            bg = COLORS["bg_dark"]
        
        super().__init__(parent, bg=bg, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        
        # Configure canvas to use scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create interior frame inside canvas
        self.interior = tk.Frame(self.canvas, bg=bg)
        self.interior_id = self.canvas.create_window((0, 0), window=self.interior, anchor=tk.NW)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind events for resizing and scrolling
        self.interior.bind("<Configure>", self._on_interior_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel for scrolling
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
    
    def _on_interior_configure(self, event):
        """Update scroll region when interior frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Show/hide scrollbar based on content height
        canvas_height = self.canvas.winfo_height()
        content_height = self.interior.winfo_reqheight()
        
        if content_height <= canvas_height:
            self.scrollbar.pack_forget()
        else:
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _on_canvas_configure(self, event):
        """Resize interior frame width to match canvas."""
        self.canvas.itemconfig(self.interior_id, width=event.width)
    
    def _bind_mousewheel(self, event):
        """Bind mousewheel when cursor enters canvas."""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel when cursor leaves canvas."""
        self.canvas.unbind_all("<MouseWheel>")
    
    def _on_mousewheel(self, event):
        """Scroll on mousewheel."""
        # Only scroll if content is larger than canvas
        if self.interior.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class StyledSpinbox(tk.Spinbox):
    """A styled spinbox for numeric input."""
    
    def __init__(
        self,
        parent,
        from_: int,
        to: int,
        value: int = None,
        width: int = 10,
        **kwargs
    ):
        super().__init__(
            parent,
            from_=from_,
            to=to,
            font=FONTS["body"],
            bg=COLORS["input_bg"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            buttonbackground=COLORS["bg_medium"],
            width=width,
            **kwargs
        )
        if value is not None:
            self.delete(0, tk.END)
            self.insert(0, str(value))


class LogConsole(tk.Frame):
    """A scrollable log console for displaying output."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_dark"])
        
        # Create text widget with scrollbar
        self.text = tk.Text(
            self,
            font=FONTS["monospace"],
            bg=COLORS["input_bg"],
            fg=COLORS["text_secondary"],
            insertbackground=COLORS["text_primary"],
            height=kwargs.get("height", DIMENSIONS["log_height"]),
            width=kwargs.get("width", DIMENSIONS["log_width"]),
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        
        scrollbar = tk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            command=self.text.yview
        )
        self.text.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tags for colored output
        self.text.tag_configure("info", foreground=COLORS["info"])
        self.text.tag_configure("success", foreground=COLORS["success"])
        self.text.tag_configure("warning", foreground=COLORS["warning"])
        self.text.tag_configure("error", foreground=COLORS["error"])
        self.text.tag_configure("heading", foreground=COLORS["accent"], font=FONTS["body_bold"])
    
    def log(self, message: str, level: str = "info"):
        """Add a log message with the specified level."""
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, message + "\n", level)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)
        self.update_idletasks()
    
    def clear(self):
        """Clear all log messages."""
        self.text.configure(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.configure(state=tk.DISABLED)


class ProgressIndicator(tk.Frame):
    """A styled progress indicator with label."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_dark"])
        
        self.label = StyledLabel(self, text="Ready", style="small")
        self.label.pack(anchor=tk.W)
        
        # Create progress bar style
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=COLORS["progress_bg"],
            background=COLORS["progress_fill"],
            thickness=8
        )
        
        self.progress = ttk.Progressbar(
            self,
            style="Custom.Horizontal.TProgressbar",
            orient=tk.HORIZONTAL,
            length=300,
            mode="determinate"
        )
        self.progress.pack(fill=tk.X, pady=(5, 0))
    
    def set_progress(self, value: int, text: str = None):
        """Set progress value (0-100) and optional label text."""
        self.progress["value"] = value
        if text:
            self.label.configure(text=text)
        self.update_idletasks()
    
    def set_indeterminate(self, active: bool, text: str = None):
        """Set indeterminate mode for unknown-length operations."""
        if active:
            self.progress.configure(mode="indeterminate")
            self.progress.start(10)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress["value"] = 0
        
        if text:
            self.label.configure(text=text)
        self.update_idletasks()
    
    def reset(self):
        """Reset progress indicator to initial state."""
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        self.label.configure(text="Ready")


class StatusBar(tk.Frame):
    """A status bar for displaying current status."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_medium"], **kwargs)
        
        self.status_label = tk.Label(
            self,
            text="Ready",
            font=FONTS["small"],
            bg=COLORS["bg_medium"],
            fg=COLORS["text_secondary"],
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.version_label = tk.Label(
            self,
            text="IDC_RIOP v1.0",
            font=FONTS["small"],
            bg=COLORS["bg_medium"],
            fg=COLORS["text_muted"],
            padx=10
        )
        self.version_label.pack(side=tk.RIGHT)
    
    def set_status(self, text: str, level: str = "info"):
        """Set status text with optional level coloring."""
        color_map = {
            "info": COLORS["text_secondary"],
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "error": COLORS["error"],
        }
        self.status_label.configure(
            text=text,
            fg=color_map.get(level, COLORS["text_secondary"])
        )
