import setuptools

requirements = []

with open("requirements.txt", 'r') as file:
	for line in file:
		requirements.append(line)

setuptools.setup(
	name="analysis",
	version="1.12",
	author="The Titan Scouting Team",
	author_email="titanscout2022@gmail.com",
	description="analysis package developed by Titan Scouting for The Red Alliance",
	long_description="",
	long_description_content_type="text/markdown",
	url="https://github.com/titanscout2022/tr2022-strategy",
	packages=setuptools.find_packages(),
	install_requires=requirements,
	license = "BSD 3-Clause License",
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.6',
)