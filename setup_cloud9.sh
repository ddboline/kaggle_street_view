#!/bin/bash

sudo apt-get update
sudo apt-get install -y python-pip
sudo apt-get install -y gcc g++ gfortran build-essential
sudo apt-get install -y git wget linux-image-generic libopenblas-dev
sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy
sudo apt-get install -y cython linux-image-extra-virtual htop
sudo apt-get install -y python-matplotlib python-pandas python-virtualenv

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda unzip

cat >> ~/.theanorc << EOL
[global]
floatX=float32
device=gpu
[mode]=FAST_RUN

[nvcc]
fastmath=True

[cuda]
root=/usr/local/cuda
EOL

cat >> ~/.bashrc << EOL
export PATH="/usr/local/cuda-6.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-6.5/lib64"
EOL

cat > blacklist-nouveau.conf << EOL
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOL

sudo mv blacklist-nouveau.conf /etc/modprobe.d/

echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf

sudo update-initramfs -u

rm /home/ubuntu/.ssh/known_hosts
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_STEP1"

sudo reboot

# cd /usr/local/cuda/samples/1_Utilities/deviceQuery
# sudo make
# sudo ./deviceQuery
