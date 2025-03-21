from setuptools import setup, find_namespace_packages


requirements_package = [
    "python_ex @ git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git",
    "scipy"]

setup(
    name="gui_toolbox",
    version="0.0.1",
    description="",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_vision_toolbox.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    zip_safe=False,
    python_requires=">= 3.10.0",
    install_requires=requirements_package
)