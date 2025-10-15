from setuptools import setup, find_packages

setup(
    name="guitar_map",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "librosa>=0.11",
        "mido",
        "pydantic>=2.0",
        "demucs",  # if using separation
        "torch",  # for Demucs
        "torchaudio",  # for audio processing with torch
        "resampy",  # required by librosa
    ],
    entry_points={
        "console_scripts": [
            "guitar-map=guitar_map.cli:main",
        ],
    },
    python_requires=">=3.9",
    author="Your Name",
    description="Guitar tab extraction pipeline from audio",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/guitar_map",
)
