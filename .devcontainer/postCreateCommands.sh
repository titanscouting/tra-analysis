source ~/.bashrc
shopt -s expand_aliases
apt-get install software-properties-common
apt-add-repository universe
apt update -y
apt upgrade -y
apt-get install -y python3.7 python3.7-dev git python3-pip python3-kivy libc6 build-essential libgl1-mesa-glx libglib2.0-0 libgstreamer1.0-0 libsdl2-2.0.0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0
pip install -r /workspaces/red-allsiance-analysis/data-analysis/requirements.txt
pip install -r /workspaces/red-alliance-analysis/analysis-master/requirements.txt
pip install --no-cache-dir pylint