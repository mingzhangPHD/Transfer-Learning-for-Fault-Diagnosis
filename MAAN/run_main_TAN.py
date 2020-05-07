import os
import scipy.misc
import numpy as np

from FD_DCTAN_model import *
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf
from multiprocessing import Process

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20000, "Epoch to train ")
flags.DEFINE_integer("epoch_pre", 2000, "Epoch to pre_train ")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("y_dim", 3, "The kind of fault") #Attention Please : 10Classes in fault categories
flags.DEFINE_integer("cuda", 0, "The number of cuda")

flags.DEFINE_string("dataset_name_s", "PHM2009_dataset_30Hz_L_helical_C3_1_out", "The name of source dataset []")
flags.DEFINE_string("dataset_name_t", "PHM2009_dataset_40Hz_H_helical_C3_1_out", "The name of target dataset []")
flags.DEFINE_string("model_name", "MAAN", "The name of model")
flags.DEFINE_string("x_type", "X_fft", "X_w, X_fft")

flags.DEFINE_string("input_fname_pattern", ".h5", "Glob pattern of filename of input wave")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("data_dir", "./datasets", "Root directory of dataset [data]")
flags.DEFINE_string("data_dir", "E:/Algorithm_developing/github_zm/Datasets/Data4MAAN/datasets", "Root directory of dataset [data]")
# flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("run_all", False, "True for visualizing, False for nothing [False]")


global FLAGS
FLAGS = flags.FLAGS


def run(data_names):
    # global FLAGS
    print(FLAGS.dataset_name_s)
    print(FLAGS.dataset_name_t)
    FLAGS.dataset_name_s = data_names[0]
    FLAGS.dataset_name_t = data_names[1]
    print(FLAGS.dataset_name_s)
    print(FLAGS.dataset_name_t)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True


    with tf.Session(config=run_config) as sess:

        if FLAGS.model_name == "CNN":
            model = CNN(
                sess,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=FLAGS.y_dim,
                dataset_name_s=FLAGS.dataset_name_s,
                dataset_name_t=FLAGS.dataset_name_t,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                data_dir=FLAGS.data_dir,
                x_type=FLAGS.x_type)
        if FLAGS.model_name == "MAAN":
            model = MAAN(
                sess,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=FLAGS.y_dim,
                dataset_name_s=FLAGS.dataset_name_s,
                dataset_name_t=FLAGS.dataset_name_t,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                data_dir=FLAGS.data_dir,
                x_type=FLAGS.x_type)

        show_all_variables()

        if FLAGS.train:
            model.train(FLAGS)
        else:
            if not model.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")



if __name__ == '__main__':



    sign_run_all = FLAGS.run_all

    if not sign_run_all:
        # tf.app.run()
        # p = Process(target=run, args=(FLAGS,))
        data_names = [FLAGS.dataset_name_s, FLAGS.dataset_name_t]
        p = Process(target=run, args=(data_names,))
        p.start()
        # end for the sub program, GPU will release
        p.join()
    else:
        # The first experiment scenario in MAAN paper

        # domain_name_s = ['PHM2009_dataset_30Hz_L_helical_C3_1_out','PHM2009_dataset_35Hz_L_helical_C3_1_out','PHM2009_dataset_40Hz_L_helical_C3_1_out',
        #                  'PHM2009_dataset_45Hz_L_helical_C3_1_out','PHM2009_dataset_50Hz_L_helical_C3_1_out']
        # domain_name_t = ['PHM2009_dataset_30Hz_H_helical_C3_1_out','PHM2009_dataset_35Hz_H_helical_C3_1_out','PHM2009_dataset_40Hz_H_helical_C3_1_out',
        #                  'PHM2009_dataset_45Hz_H_helical_C3_1_out','PHM2009_dataset_50Hz_H_helical_C3_1_out']

        # The second experiment scenario in MAAN paper
        domain_name_s = ['PHM2009_dataset_45Hz_L_helical_C3_1_out','PHM2009_dataset_45Hz_L_helical_C3_1_out','PHM2009_dataset_45Hz_L_helical_C3_1_out',
                         'PHM2009_dataset_45Hz_L_helical_C3_1_out','PHM2009_dataset_45Hz_L_helical_C3_1_out']
        domain_name_t = ['PHM2009_dataset_30Hz_H_helical_C3_1_out','PHM2009_dataset_35Hz_H_helical_C3_1_out','PHM2009_dataset_40Hz_H_helical_C3_1_out',
                         'PHM2009_dataset_45Hz_H_helical_C3_1_out','PHM2009_dataset_50Hz_H_helical_C3_1_out']



        for d_n_s, d_n_t in zip(domain_name_s,domain_name_t):
            # global FLAGS
            print(FLAGS.dataset_name_s)
            print(FLAGS.dataset_name_t)
            FLAGS.dataset_name_s = d_n_s
            FLAGS.dataset_name_t = d_n_t
            print(FLAGS.dataset_name_s)
            print(FLAGS.dataset_name_t)
            # p = Process(target=run, args=(FLAGS,))
            data_names = [d_n_s, d_n_t]
            p = Process(target=run, args=(data_names,))
            p.start()
            # end for the sub program, GPU will release
            p.join()
            # run(FLAGs=FLAGS)

