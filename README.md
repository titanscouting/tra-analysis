# Red Alliance Analysis &middot; ![GitHub release (latest by date)](https://img.shields.io/github/v/release/titanscout2022/red-alliance-analysis)

Titan Robotics 2022 Strategy Team Repository for Data Analysis Tools. Included with these tools are the backend data analysis engine formatted as a python package, associated binaries for the analysis package, and premade scripts that can be pulled directly from this repository and will integrate with other Red Alliance applications to quickly deploy FRC scouting tools.

---

# `tra-analysis`

`tra-analysis` is a higher level package for data processing and analysis. It is a python library that combines popular data science tools like numpy, scipy, and sklearn along with other tools to create an easy-to-use data analysis engine. tra-analysis includes analysis in all ranges of complexity from basic statistics like mean, median, mode to complex kernel based classifiers and allows user to more quickly deploy these algorithms. The package also includes performance metrics for score based applications including elo, glicko2, and trueskill ranking systems.

At the core of the tra-analysis package is the modularity of each analytical tool. The package encapsulates the setup code for the included data science tools. For example, there are many packages that allow users to generate many different types of regressions. With the tra-analysis package, one function can be called to generate many regressions and sort them by accuracy.

## Prerequisites
---

* Python >= 3.6
* Pip which can be installed by running\
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`\
`python get-pip.py`\
after installing python, or with a package manager on linux. Refer to the [pip installation instructions](https://pip.pypa.io/en/stable/installing/) for more information.

## Installing
---

#### Standard Platforms

For the latest version of tra-analysis, run `pip install tra-analysis` or `pip install tra_analysis`. The requirements for tra-analysis should be automatically installed.

#### Exotic Platforms (Android)

[Termux](https://termux.com/) is recommended for a linux environemnt on Android. Consult the [documentation](https://titanscouting.github.io/analysis/general/installation#exotic-platforms-android) for advice on installing the prerequisites. After installing the prerequisites, the package should be installed normally with `pip install tra-analysis` or `pip install tra_analysis`. 

## Use

---

tra-analysis operates like any other python package. Consult the [documentation](https://titanscouting.github.io/analysis/tra_analysis/) for more information.

## Supported Platforms

---

Although any modern 64 bit platform should be supported, the following platforms have been tested to be working:
* AMD64 (Tested on Zen, Zen+, and Zen 2)
* Intel 64/x86_64/x64 (Tested on Kaby Lake, Ice Lake)
* ARM64 (Tested on Broadcom BCM2836 SoC, Broadcom BCM2711 SoC)

The following OSes have been tested to be working:
* Linux Kernel 3.16, 4.4, 4.15, 4.19, 5.4
	* Ubuntu 16.04, 18.04, 20.04
	* Debian (and Debian derivaives) Jessie, Buster
* Windows 7, 10

The following python versions are supported:
* python 3.6 (not tested)
* python 3.7
* python 3.8

---

# `data-analysis`

To facilitate data analysis of collected scouting data in a user firendly tool, we created the data-analysis application. At its core it uses the tra-analysis package to conduct any number of user selected tests on data collected from the TRA scouting app. It uploads these tests back to MongoDB where it can be viewed from the app at any time.  

The data-analysis application also uses the TRA API to interface with MongoDB and uses the TBA API to collect additional data (match win/loss).

The application can be configured with a configuration tool or by editing the config.json directly.

## Prerequisites

---

Before installing and using data-analysis, make sure that you have installed the folowing prerequisites:
- A common operating system like **Windows** or (*most*) distributions of **Linux**. BSD may work but has not been tested nor is it reccomended.
- [Python](https://www.python.org/) version **3.6** or higher
- [Pip](https://pip.pypa.io/en/stable/) (installation instructions [here](https://pip.pypa.io/en/stable/installing/))

## Installing Requirements

---

Once navigated to the data-analysis folder run `pip install -r requirements.txt` to install all of the required python libraries.

## Scripts

---

The data-analysis application is a collection of various scripts and one config file. For users, only the main application `superscript.py` and the config file `config.json` are important. 

To run the data-analysis application, navigate to the data-analysis folder once all requirements have been installed and run `python superscript.py`. If you encounter the error:

`pymongo.errors.ConfigurationError: Empty host (or extra comma in host list).`

don't worry, you may have just not configured the application correctly, but would otherwise work. Refer to [the documentation](https://titanscouting.github.io/analysis/data_analysis/Config) to learn how to configure data-analysis.

# Contributing

Read our included contributing guidelines (`CONTRIBUTING.md`) for more information and feel free to reach out to any current maintainer for more information. 

# Build Statuses
![Analysis Unit Tests](https://github.com/titanscout2022/red-alliance-analysis/workflows/Analysis%20Unit%20Tests/badge.svg)
![Superscript Unit Tests](https://github.com/titanscout2022/red-alliance-analysis/workflows/Superscript%20Unit%20Tests/badge.svg?branch=master)
