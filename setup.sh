sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

git clone https://github.com/openai/gym.git
cd gym
sudo pip3 install -e '.[all]'

cd ..