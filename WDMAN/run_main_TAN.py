
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
flags.DEFINE_integer("y_dim", 10, "The kind of fault") #Attention Please : 10Classes for CASEDATASET 4 for BUCTDATASET
flags.DEFINE_integer("cuda", 0, "The number of cuda")

flags.DEFINE_string("dataset_name_s", "CaseDE12K_dataset_0_nx1200_class10", "The name of source dataset []")
flags.DEFINE_string("dataset_name_t", "CaseDE12K_dataset_1_nx1200_class10", "The name of target dataset []")
flags.DEFINE_string("model_name", "HANs", "The name of model")
flags.DEFINE_string("x_type", "X", "X")

flags.DEFINE_string("input_fname_pattern", ".h5", "Glob pattern of filename of input wave")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./datasets", "Root directory of dataset [data]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("sign_run_all", False, "True for visualizing, False for nothing [False]")

global FLAGS
FLAGS = flags.FLAGS

def run():

    print(FLAGS.dataset_name_s)
    print(FLAGS.dataset_name_t)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)


    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True


    with tf.Session(config=run_config) as sess:

        model = WDMAN(
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

    sign_run_all = FLAGS.sign_run_all

    if not sign_run_all:

        p = Process(target=run)
        p.start()
        # end for the sub program, GPU will release
        p.join()
    else:

        domain_name_s = ['CaseDE12K_dataset_0_nx1200_class10','CaseDE12K_dataset_0_nx1200_class10','CaseDE12K_dataset_0_nx1200_class10']
        domain_name_t = ['CaseDE12K_dataset_1_nx1200_class10','CaseDE12K_dataset_2_nx1200_class10','CaseDE12K_dataset_3_nx1200_class10']

        # domain_name_s = ['BuctDseDE12kHz_dateset_30Hz_nx1200_Class4','BuctDseDE12kHz_dateset_50Hz_nx1200_Class4']
        # domain_name_t = ['BuctDseDE12kHz_dateset_50Hz_nx1200_Class4','BuctDseDE12kHz_dateset_30Hz_nx1200_Class4']

        for d_n_s, d_n_t in zip(domain_name_s,domain_name_t):
            print(FLAGS.dataset_name_s)
            print(FLAGS.dataset_name_t)
            FLAGS.dataset_name_s = d_n_s
            FLAGS.dataset_name_t = d_n_t
            print(FLAGS.dataset_name_s)
            print(FLAGS.dataset_name_t)
            p = Process(target=run, args=(FLAGS,))
            p.start()
            # end for the sub program, GPU will release
            p.join()
            # run(FLAGs=FLAGS)


