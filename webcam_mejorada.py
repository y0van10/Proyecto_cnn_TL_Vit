"""Compatibilidad con la versión anterior del repositorio: ejecuta app_webcam.py con los mismos argumentos."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "app_webcam.py"), run_name="__main__")
