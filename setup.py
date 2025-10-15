from setuptools import setup, find_packages

setup(
    name="guitar_map",
    version="0.1.0",
    packages=find_packages(),  # Now just find packages in current dir
    install_requires=[
        "numpy",
        "librosa",
        "mido",
        "pydantic",
        "torch",
        "torchaudio",
        "demucs",
    ],
    entry_points={
        "console_scripts": [
            "guitar-map = app.cli:main",
        ],
    },
    python_requires=">=3.10",
)
