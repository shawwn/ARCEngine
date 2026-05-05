"""
Standalone Cython build script.

Why not pyproject.toml: the project's existing build backend is uv-build, which
doesn't compile Cython. Switching backends would conflict with the recent pypi
release pipeline (b495c6a). This script compiles the .pyx in place so the
speedup is opt-in at build time without disturbing the package build config.

Usage:
    uv run python build_cython.py build_ext --inplace

After this runs, arcengine/_raw_render_cython.<plat>.so exists and Python's
import of it from camera.py will succeed.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "arcengine._raw_render_cython",
        ["arcengine/_raw_render_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="arcengine_cython_ext",
    ext_modules=cythonize(extensions, language_level=3),
)
