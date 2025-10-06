"""Logging and status display utilities."""

import sys
from datetime import datetime
from typing import Optional


class Logger:
    """Simple logger for consistent output formatting."""
    
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "gray": "\033[90m",
    }
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def _color(self, text: str, color: str) -> str:
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text
    
    def info(self, message: str):
        """Log info message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = self._color(f"[{timestamp}] INFO:", "blue")
        print(f"{prefix} {message}")
    
    def success(self, message: str):
        """Log success message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = self._color(f"[{timestamp}] SUCCESS:", "green")
        print(f"{prefix} {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = self._color(f"[{timestamp}] WARNING:", "yellow")
        print(f"{prefix} {message}", file=sys.stderr)
    
    def error(self, message: str):
        """Log error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = self._color(f"[{timestamp}] ERROR:", "red")
        print(f"{prefix} {message}", file=sys.stderr)
    
    def progress(self, current: int, total: int, message: str = ""):
        """Show progress indicator."""
        pct = (current / total * 100) if total > 0 else 0
        bar_length = 30
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        status = f"\r[{bar}] {pct:.1f}% ({current}/{total})"
        if message:
            status += f" - {message}"
        
        sys.stdout.write(status)
        if current >= total:
            sys.stdout.write("\n")
        sys.stdout.flush()


# Global logger instance
logger = Logger()
