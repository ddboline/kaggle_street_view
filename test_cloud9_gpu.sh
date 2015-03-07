#!/bin/bash

sudo modprobe nvidia-340
sudo modprobe nvidia-340-uvm

sudo apt-get install -y python-pandas python-sklearn python-skimage unzip ipython

export PATH="/usr/local/cuda-6.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-6.5/lib64"

rm *.pickle *.png
rm /home/ubuntu/.ssh/known_hosts

for F in sampleSubmission.csv submission.csv trainLabels.csv testResized.zip test.zip trainResized.zip train.zip;
do
    scp ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_street_view/$F .
done

for F in testResized.zip test.zip trainResized.zip train.zip;
do
    unzip -x $F
done

touch output.out
touch output.err
# ./test.py >> output.out 2>> output.err
./my_model.py $1 >> output.out 2>> output.err

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_STEP3"

# if [ -z $1 ]; then
#     D=`date +%Y%m%d%H%M%S`
# else
#     D=$1_`date +%Y%m%d%H%M%S`
# fi
# ssh ddboline@ddbolineathome.mooo.com "mkdir -p ~/setup_files/build/kaggle_facial_keypoints/output_${D}"
# scp output.out output.err *.png *.pickle submission*.csv ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_facial_keypoints/output_${D}
# ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_${D}"
# sudo shutdown now
