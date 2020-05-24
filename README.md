# Red Alliance Analysis &middot; ![GitHub release (latest by date)](https://img.shields.io/github/v/release/titanscout2022/red-alliance-analysis)
Titan Robotics 2022 Strategy Team Repository for Data Analysis Tools. Included with these tools are the backend data analysis engine formatted as a python package, associated binaries for the analysis package, and premade scripts that can be pulled directly from this repository and will integrate with other Red Alliance applications to quickly deploy FRC scouting tools.
# Getting Started
## Prerequisites
* Python >= 3.6
* Pip which can be installed by running `python -m pip install -U pip` after installing python
## Installing
There are a few different services/packages that can be installed. Installation instructions are seperated by scenario and are ordered from least complex to most. 
### Linux (Most Distros)/Windows ｜ TRA Service plus GUI, Analysis Package
Portable application files are avaliable on the releases page. 
* Download the `.zip` or `.tar.gz` files from the releases page. If you want the GUI version, be sure to download the one marked with GUI.
* Unzip the files and save the folder somewhere safe.
* Navigate to the unzipped folder and run:
	* Windows: `TRA-GUI.bat` .
	* Linux: `TRA-GUI.sh`.
### Linux (Most Distros)/Windows ｜ TRA Service, Analysis Package
Service application files are avaliable on the releases page.
* Download the `.zip` or `tar.gz` files from from the releases page. If you want the Service version, be sure to download the one that is not marked with GUI. The GUI version will also contain the service application.
* Unzip the files and save the folder somewhere safe.
* Navigate to the unzipped folder and run:
	* Windows: `TRA.bat`.
	* Linux: `TRA.sh`.
### Linux (Most Distros) CLI-Only ｜ TRA Service, Analysis Package
Service application files are avaliable on the releases page.
* Download the `.zip` or `tar.gz` files from from the releases page with `wget https://github.com/titanscout2022/red-alliance-analysis/releases/download/latest/TRA.tar.gz`. If you want the Service version, be sure to download the one that is not marked with GUI. The GUI version will also contain the service application.
* Unzip the files with `tar -xzf TRA.tar.gz` .
* Navigate to the unzipped folder with `cd TRA/` and run `./TRA.sh`.
### Installing Only the Analysis Package
Make sure that python an pip are already installed. 
Download the `.whl` file from the releases page. Then install the wheel file by running `pip install [name of the .whl file]`.
### Installing Only the Analysis Package From Source
Navigating to `analysis-master/`, running `./build.sh` will build the analysis package from `analysis-master/analysis/`. The built wheel files are stored in `analysis-master/dist/`.
## Config
# Supported Platforms
Although any modern 64 bit platform should be supported, the following platforms have been tested to be working:
* AMD64 (Tested on Zen 1, Zen +, and Zen 2)
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
# Build Statuses
![Analysis Unit Tests](https://github.com/titanscout2022/red-alliance-analysis/workflows/Analysis%20Unit%20Tests/badge.svg)
![Superscript Unit Tests](https://github.com/titanscout2022/red-alliance-analysis/workflows/Superscript%20Unit%20Tests/badge.svg?branch=master)