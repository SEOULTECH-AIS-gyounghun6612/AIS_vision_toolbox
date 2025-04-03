from setuptools import setup, find_namespace_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, encoding="utf-8") as f:
        return [_l for _l in f.readlines() if _l and not _l[0] != "#"]


setup(
    name="vision_toolbox",
    description="",
    url=(
        "https://github.com/"
        "/".join(
            "SEOULTECH-AIS-gyounghun6612"  # github user name
            "AIS_vision_toolbox.git"  # repository name
        )),
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    zip_safe=False,
    python_requires=">= 3.10.0",
    install_requires=load_requirements()
)