sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock

sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

export LC_ALL=C

git clone https://github.com/openai/gym.git
cd gym
sudo pip3 install -e '.[all]'

cd ..