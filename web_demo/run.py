"""Launch the offline Streamlit demo.

This file is also the PyInstaller entry point.  The bundled application is
started through Streamlit's Python API rather than a child ``python`` process,
so the executable does not require Python on the target machine.
"""

import os
import sys


def _bundle_root() -> str:
    """Return the directory containing bundled application resources."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_ROOT = _bundle_root()
_APP = os.path.join(_ROOT, "web_demo", "app.py")

if not os.path.isfile(_APP):
    raise FileNotFoundError(f"Streamlit application was not bundled: {_APP}")

print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║          Cognitive-Affective Constraint Framework            ║")
print("║                  Streamlit Demo Launcher                     ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()
print("  Starting Streamlit server …")
print("  Open your browser at:  http://localhost:8501")
print()
print("  Press Ctrl-C to stop the server.")
print()

try:
    # Streamlit resolves its config relative to the working directory.  The
    # spec file places .streamlit/config.toml below this bundled root.
    os.chdir(_ROOT)
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        _APP,
        "--global.developmentMode", "false",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    try:
        import webbrowser
        webbrowser.open("http://localhost:8501", new=2)
    except Exception:
        pass
    cli.main()
except KeyboardInterrupt:
    print("\n  Demo stopped.")
