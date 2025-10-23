import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

def load_cpp_files(folder):
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".cpp")]
    return files


__version__ = "1.0.4"

ext_modules = [
    Pybind11Extension(
        "miner_cpp",
        load_cpp_files("./src"),
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="rlcomp_cpp",
    version=__version__,
    author="VietBT",
    author_email="tvbui@smu.edu.sg",
    description="RLComp C++ Wrapper",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=True,
    python_requires=">=3.6",
)