
#
# Tensorflow-gpu + Keras
# This docker will also include the BraTS challenge code.
#


FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Mariano Cabezas <mcabezas@eia.udg.edu>


# We add all the necessary files to the image
ADD get-pip.py /usr/local/
ADD test_brats2018.py /bin/
ADD nets.py /bin/
ADD data_creation.py /bin/
ADD data_manipulation /bin/data_manipulation
ADD layers.py /bin/
ADD utils.py /bin/
ADD __init__.py /bin/
ADD brats18-unet.hdf5 /usr/local/models/
ADD brats18-ensemble.hdf5 /usr/local/models/
ADD brats18-nets.hdf5 /usr/local/models/
ADD requirements.txt /usr/local/

# We install all the necessary commands
RUN apt-get update
RUN apt-get install -y python2.7 python-dev libpng-dev libfreetype6-dev git pkg-config g++ libjpeg-dev
RUN python /usr/local/get-pip.py
RUN pip install -r /usr/local/requirements.txt


CMD test_brats2018.py