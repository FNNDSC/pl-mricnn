pl-mricnn
================================

.. image:: https://badge.fury.io/py/mricnn.svg
    :target: https://badge.fury.io/py/mricnn

.. image:: https://travis-ci.org/FNNDSC/mricnn.svg?branch=master
    :target: https://travis-ci.org/FNNDSC/mricnn

.. image:: https://img.shields.io/badge/python-3.5%2B-blue.svg
    :target: https://badge.fury.io/py/pl-mricnn

.. contents:: Table of Contents


Abstract
--------

An app to train and infer MRI data using a CNN model


Synopsis
--------

.. code::

    python mricnn.py                                           \
        [-v <level>] [--verbosity <level>]                          \
        [--version]                                                 \
        [--man]                                                     \
        [--meta]                                                    \
        [--mode <mode>]                                             \
        [--epochs <epochs>]                                         \
        <inputDir>
        <outputDir> 

Description
-----------

``mricnn.py`` is a ChRIS-based application that uses a 3D Convolutional Neural Network to train on low contrast brain MRI images and predict a masked/segmented image of the same. I have implemented a 3D dense U-net as a training/inference model.
For efficient learning, I am using a batch size of 16 for training. The batch overlaps with one another for a continued learning and minimal loss. A good data set size would be around 20-30 subjects with an epoch number of 50. If the dataset is 
more than 200, a much lower epoch size(>=5 && <=10) is recommended. For efficient functioning of this application, the input and output directory structure is necessary. The input folder MUST have valid train and mask .NPY files. This can be achieved by using a plugin fnndsc/pl-mgz_converter. The output of the mentioned plugin is most suitable for the current plugin to work.
A typical input directory will have 2 subdirectories called 'train' & 'masks'; alongwith 2 .npy files called imgs_train.npy and imgs_mask_train.npy

Arguments
---------

.. code::

    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.

    [--version]
    If specified, print version number. 
    
    [--man]
    If specified, print (this) man page.

    [--meta]
    If specified, print plugin meta data.
    
    [--mode]
    Required: 1: Training 2: Prediction/inference
    
    [--epochs]
    Optional: Default epoch number is 5


Run
----

This ``plugin`` can be run in two modes: natively as a python package or as a containerized docker image.

Using PyPI
~~~~~~~~~~

To run from PyPI, simply do a 

.. code:: bash

    pip install mricnn

and run with

.. code:: bash

    mricnn.py --man --mode <mode> --epochs <no. of epochs> /tmp /tmp

to get inline help. The app should also understand being called with only two positional arguments

.. code:: bash

    mricnn.py --mode <mode> --epochs <no. of epochs> /some/input/directory /destination/directory


Using ``docker run``
~~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Now, prefix all calls with 

.. code:: bash

    docker run --rm -v $(pwd)/out:/outgoing                             \
            fnndsc/pl-mricnn mricnn.py  --mode <1 or 2>                      \

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-mricnn mricnn.py                        \
            --man                                                       \
            --mode 1                                                    \
            --epochs 10                                                 \
            /incoming /outgoing

Examples
--------

.. code:: bash

 mkdir in out && chmod 777 out                                \
 docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-mricnn mricnn.py                        \
            --man                                                       \
            --mode 1                                                    \
            --epochs 10                                                 \
            /incoming /outgoing






