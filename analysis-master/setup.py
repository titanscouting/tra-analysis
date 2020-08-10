import setuptools

requirements = []

with open("requirements.txt", 'r') as file:
	for line in file:
		requirements.append(line)

setuptools.setup(
	name="tra_analysis",
	version="2.0.3",
	author="The Titan Scouting Team",
	author_email="titanscout2022@gmail.com",
	description="Analysis package developed by Titan Scouting for The Red Alliance",
	long_description="",
	long_description_content_type="text/markdown",
	url="https://github.com/titanscout2022/tr2022-strategy",
	packages=setuptools.find_packages(),
	install_requires=requirements,
	license = "GNU General Public License v3.0",
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.6',
)