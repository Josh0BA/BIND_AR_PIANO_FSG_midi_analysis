from setuptools import setup, find_packages

setup(
    name="midi_state_analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["pandas", "pretty_midi", "scipy", "statsmodels", "seaborn", "matplotlib"],
    entry_points={"console_scripts": ["midi-analysis=midi_state_analysis.cli:main"]},
    author="Your Name",
    description="Analyse von State-Transitionen in Klavier-MIDI-Daten",
)
