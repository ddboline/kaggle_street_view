#!/bin/bash

export PATH="/usr/local/cuda-6.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-6.5/lib64"

sudo apt-get install -y liblapack-dev
sudo apt-get remove -y linux-headers-3.13.0-36 linux-headers-3.13.0-36-generic linux-image-3.13.0-36-generic
sudo apt-get install -y --reinstall nvidia-340 nvidia-340-dev nvidia-340-uvm

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

### changing a line in theano/sandbox/cuda/cuda_ndarray.cuh (more likely the former than the latter)
sudo pip install --upgrade scipy 2>&1 > scipy_build.log
sudo pip install --upgrade git+https://github.com/Theano/Theano.git 2>&1 > theano_build.log
sudo pip install --upgrade git+https://github.com/benanne/Lasagne.git 2>&1 > lasagne_build.log
sudo pip install --upgrade git+https://github.com/dnouri/nolearn.git 2>&1 > nolearn_build.log

cd $HOME
git clone https://github.com/lisa-lab/pylearn2.git
### this doesn't really work any other way
cd pylearn2
echo "You will need to build pylearn2 manually (setup.py written by idiot...)"
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_STEP2"

