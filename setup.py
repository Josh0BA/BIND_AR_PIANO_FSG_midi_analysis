from pathlib import Path
from setuptools import setup

README = Path(__file__).with_name("README.md")

setup(
    name="bind-ar-piano-midi-analysis",
    version="1.0.1",
    description="MIDI-based piano motor-learning analysis scripts",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    # This repository is script-first (no installable Python package directory yet).
    packages=[],
    install_requires=[
        "numpy",
        "pandas",
        "pretty_midi",
        "matplotlib",
        "seaborn",
        "scipy",
        "statsmodels",
    ],
    extras_require={
        "excel": ["openpyxl", "python-calamine"],
    },
)
