#!/usr/bin/env python                                            
#
# mricnn ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

np.random.seed(256)
import tensorflow as tf

from keras.models import Model,load_model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
K.set_image_data_format('channels_last')
import cv2
import sys
sys.path.append(os.path.dirname(__file__))

print ("*************TEST FOR GPUs*****************")
print(tf.test.gpu_device_name())

# import the Chris app superclass
from chrisapp.base import ChrisApp
from data3D import load_train_data, load_test_data, preprocess_squeeze,create_test_data

K.set_image_data_format('channels_last')

project_name = '3D-Dense-Unet'
img_rows = 256
img_cols = 256
img_depth = 16
smooth = 1.


Gstr_title = """

Generate a title from 
http://patorjk.com/software/taag/#p=display&f=Doom&t=mricnn

"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the 
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       mricnn.py 

    SYNOPSIS

        python mricnn.py                                         \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python mricnn.py   \\
                                in    out

    DESCRIPTION

        `mricnn.py` ...

    ARGS

        [-h] [--help]
        If specified, show help message and exit.
        
        [--json]
        If specified, show json representation of app and exit.
        
        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.
        
        [--savejson <DIR>] 
        If specified, save json representation file to DIR and exit. 
        
        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.
        
        [--version]
        If specified, print version number and exit.

        [--mode <mode>]
        Should be specified,
        If the <mode> is train, model will be trained,
        if the <mide> is infer, model will infer

"""


class Mricnn(ChrisApp):
    """
    An app to train and infer MRI data using a CNN model.
    """
    AUTHORS                 = 'Sandip Samal (sandip.samal@childrens.harvard.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A 3D CNN for MRI Segmentation'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to train and infer MRI data using a CNN model'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument('--mode',dest='mode',type=str,optional=False,
                          help='What do you want to do? 1. Train 2. Infer')
        self.add_argument('--epochs',dest='epochs',type=int,default=5,optional=True,
                          help='Specify the number of epochs. Default epochs is 5')

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    def get_unet(self):
        inputs = Input((img_depth, img_rows, img_cols, 1))
        conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conc11 = concatenate([inputs, conv11], axis=4)
        conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc11)
        conc12 = concatenate([inputs, conv12], axis=4)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

        conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conc21 = concatenate([pool1, conv21], axis=4)
        conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc21)
        conc22 = concatenate([pool1, conv22], axis=4)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

        conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conc31 = concatenate([pool2, conv31], axis=4)
        conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc31)
        conc32 = concatenate([pool2, conv32], axis=4)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

        conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conc41 = concatenate([pool3, conv41], axis=4)
        conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc41)
        conc42 = concatenate([pool3, conv42], axis=4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

        conv51 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conc51 = concatenate([pool4, conv51], axis=4)
        conv52 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conc51)
        conc52 = concatenate([pool4, conv52], axis=4)

        up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
        conv61 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        conc61 = concatenate([up6, conv61], axis=4)
        conv62 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc61)
        conc62 = concatenate([up6, conv62], axis=4)

        up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
        conv71 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        conc71 = concatenate([up7, conv71], axis=4)
        conv72 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc71)
        conc72 = concatenate([up7, conv72], axis=4)

        up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
        conv81 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        conc81 = concatenate([up8, conv81], axis=4)
        conv82 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc81)
        conc82 = concatenate([up8, conv82], axis=4)

        up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
        conv91 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        conc91 = concatenate([up9, conv91], axis=4)
        conv92 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc91)
        conc92 = concatenate([up9, conv92], axis=4)

        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.summary()
        #plot_model(model, to_file='model.png')

        model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

        return model



    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)



    def train(self, options):
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30)


        imgs_train, imgs_mask_train = load_train_data(options)
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train /= 255.  # scale masks to [0, 1]
        imgs_train /= 255.  # scale masks to [0, 1]


        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)

        model = self.get_unet()
        weight_dir =options.outputdir +'/weights'
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + '.h5'), monitor='val_loss', save_best_only=True)

        log_dir = options.outputdir+'/logs'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        csv_logger = CSVLogger(os.path.join(log_dir,  project_name + '.txt'), separator=',', append=False)

        print('-'*30)
        print('Fitting model...')
        print('-'*30)


        model.fit_generator(imgs_train, imgs_mask_train, batch_size=1, epochs=options.epochs, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])





        print('-'*30)
        print('Training finished')
        print('-'*30)


    def predict(self,options):



        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        create_test_data(options)

        imgs_test = load_test_data(options)
        imgs_test = imgs_test.astype('float32')


        imgs_test /= 255.  # scale masks to [0, 1]


        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)

        model = self.get_unet()
        weight_dir =options.inputdir+ '/weights'
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)

        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

        npy_mask_dir = 'test_mask_npy'
        if not os.path.exists(npy_mask_dir):
            os.mkdir(npy_mask_dir)

        np.save(os.path.join(npy_mask_dir, project_name + '_mask.npy'), imgs_mask_test)

        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)

        imgs_mask_test = preprocess_squeeze(imgs_mask_test)
        # imgs_mask_test /= 1.7
        #imgs_mask_test = np.around(imgs_mask_test, decimals=0)
        #info = np.iinfo(np.uint16) # Get the information of the incoming image type
        #imgs_mask_test = imgs_mask_test.astype(np.uint16)
        #imgs_mask_test=imgs_mask_test* info.max # convert back to original class/labels
        imgs_mask_test = (imgs_mask_test*255.).astype(np.uint16)
        count_visualize = 1
        count_processed = 0
        pred_dir = 'preds'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        pred_dir = os.path.join(options.outputdir, project_name)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs_mask_test.shape[0]):
            for y in range(0, imgs_mask_test.shape[1]):
                if (count_visualize > 1) and (count_visualize < 16):
                    save_img=imgs_mask_test[x][y].astype(np.uint16)
                    imsave(os.path.join(pred_dir, 'pred_' +str( count_processed )+ '.png'), save_img)
                    count_processed += 1

                count_visualize += 1
                if count_visualize == 17:
                    count_visualize = 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs_mask_test.shape[0]*14))

        print('-'*30)
        print('Prediction finished')
        print('-'*30)

    def run(self, options):
    
        if options.mode == "1":
            self.train(options)
        elif options.mode == "2":
            self.predict(options)
        else:
            print("You have selected invalid option for conversion")

                                                                


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Mricnn()
    chris_app.launch()
