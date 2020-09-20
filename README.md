# Red Alliance Analysis &middot; ![GitHub release (latest by date)](https://img.shields.io/github/v/release/titanscout2022/red-alliance-analysis)
Titan Robotics 2022 Strategy Team Repository for Data Analysis Tools. Included with these tools are the backend data analysis engine formatted as a python package, associated binaries for the analysis package, and premade scripts that can be pulled directly from this repository and will integrate with other Red Alliance applications to quickly deploy FRC scouting tools.
# Getting Started
## Prerequisites
* Python >= 3.6
* Pip which can be installed by running\
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`\
`python get-pip.py`\
after installing python, or with a package manager on linux. Refer to the [pip installation instructions](https://pip.pypa.io/en/stable/installing/) for more information.
## Installing
### Standard Platforms
For the latest version of tra-analysis, run `pip install tra-analysis` or `pip install tra_analysis`. The requirements for tra-analysis should be automatically installed.
### Exotic Platforms (Android)
[Termux](https://termux.com/) is recommended for a linux environemnt on Android. Consult the [documentation](https://titanscouting.github.io/analysis/general/installation#exotic-platforms-android) for advice on installing the prerequisites. After installing the prerequisites, the package should be installed normally with `pip install tra-analysis` or `pip install tra_analysis`. 
## Use
tra-analysis operates like any other python package. Consult the [documentation](https://titanscouting.github.io/analysis/tra_analysis/) for more information.
# Supported Platforms
Although any modern 64 bit platform should be supported, the following platforms have been tested to be working:
* AMD64 (Tested on Zen, Zen+, and Zen 2)
* Intel 64/x86_64/x64 (Tested on Kaby Lake)
* ARM64 (Tested on Broadcom BCM2836 SoC, Broadcom BCM2711 SoC)
### 
The following OSes have been tested to be working:
* Linux Kernel 3.16, 4.4, 4.15, 4.19, 5.4
	* Ubuntu 16.04, 18.04, 20.04
	* Debian (and Debian derivaives) Jessie, Buster
* Windows 7, 10
### 
The following python versions are supported:
* python 3.6 (not tested)
* python 3.7
* python 3.8
# Contributing
Read our included contributing guidelines (`CONTRIBUTING.md`) for more information and feel free to reach out to any current maintainer for more information. 
# Build Statuses
![Analysis Unit Tests](https://github.com/titanscout2022/red-alliance-analysis/workflows/Analysis%20Unit%20Tests/badge.svg)
![Superscript Unit Tests](https://github.com/titanscout2022/red-alliance-analysis/workflows/Superscript%20Unit%20Tests/badge.svg?branch=master)
