import setuptools

setuptools.setup(
    name="analysis"
    version="1.0.0.009",
    author="The Titan Scouting Team",
    author_email="titanscout2022@gmail.com",
    description="analysis package developed by Titan Scouting for The Red Alliance",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/titanscout2022/tr2022-strategy",
    packages=setuptools.find_packages(),
    install_requires=[
        "numba",
        "numpy",
        "scipy",
        "scikit-learn",
        "six",
        "matplotlib"
    ],
    license = "GNU General Public License v3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)