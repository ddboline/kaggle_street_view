#!/bin/bash

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

sudo apt-get install -y python-skimage

export PATH="/usr/local/cuda-6.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-6.5/lib64"

rm *.pickle *.png

for F in testResized.zip test.zip trainResized.zip train.zip;
do
    scp ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_street_view/$F .
    unzip -x $F
done

touch output.out
touch output.err
# ./test.py >> output.out 2>> output.err
# ./my_model.py $1 >> output.out 2>> output.err
./my_model.py

# if [ -z $1 ]; then
#     D=`date +%Y%m%d%H%M%S`
# else
#     D=$1_`date +%Y%m%d%H%M%S`
# fi
# ssh ddboline@ddbolineathome.mooo.com "mkdir -p ~/setup_files/build/kaggle_facial_keypoints/output_${D}"
# scp output.out output.err *.png *.pickle submission*.csv ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/output_${D}
# ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_${D}"
# sudo shutdown now
