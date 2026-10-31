from setuptools import setup, find_packages

setup(
    name="ez-cz",
    version="0.1.0",
    description="A standalone package for complex impedance measurement and analysis of TES devices.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "lmfit",
        "pandas",
        "PyQt5",
        "python-usbtmc",
        "python-vxi11",
        "pyusb",
        "pyvisa",
        "pyvisa-py",
        "pyserial",
    ],
    entry_points={
        "console_scripts": [
            "ez-cz-measure = ez_cz.measure:main",
            "ez-cz-analyze = ez_cz.analyze:main",
            "ez-cz-gui = ez_cz.gui:main",
        ],
    },
)
