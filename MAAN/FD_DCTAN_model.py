# -*- coding: utf-8 -*-
import os
import time
import math

# import tensorflow as tf
from tf_ops_zm import *
from tf_utils_zm import *

import pandas as pd
import numpy as np
# import h5py
# import mat4py
from sklearn.model_selection import train_test_split  
# from sklearn import preprocessing #标准化数据模块
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_curve, auc
from scipy import interp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


def load_data(data_path, classes = 4, x_type = "X_w"):
    X_orig=np.array(pd.read_hdf(data_path, key=x_type))
    Y_orig=np.array(pd.read_hdf(data_path, key='Y'))  
    #X_orig = preprocessing.scale(X_orig)
    X_train, X_test, y_train, y_test = train_test_split(X_orig, Y_orig, test_size=0.3,random_state=8)

    train_set_x_orig = X_train
    train_set_y_orig = y_train
    test_set_x_orig = X_test
    test_set_y_orig = y_test

    X_s_train = train_set_x_orig
    Y_s_train = convert_to_one_hot(train_set_y_orig, classes).T
    X_s_test = test_set_x_orig
    Y_s_test = convert_to_one_hot(test_set_y_orig, classes).T

    return X_s_train, Y_s_train, X_s_test, Y_s_test, classes

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

class CNN(object):
    # 2019年3月16日 init
    # fs特徵提取 16-32-64

    def __init__(self, sess,
                 crop=True, batch_size=64, sample_num=64,
                 y_dim=4, ff_dim=32, df_dim=32, ffc_dim=1024, dfc_dim=1024,
                 dataset_name_s='default_s', dataset_name_t='default_t',
                 input_fname_pattern='.h5', checkpoint_dir=None,
                 data_dir='.\\datasets' or './datasets',x_type = None):

        """

        Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        y_dim: (optional) Dimension of dim for y. [None]
        z_dim: (optional) Dimension of dim for Z. [100]
        ff_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]

        """
        # the discriminator net affect on the classifier net

        self.sess = sess
        self.crop = crop
        self.model_name = "CNN"
        self.dataset_name_s = dataset_name_s
        self.dataset_name_t = dataset_name_t

        self.save_path = 'results/' + self.model_name + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.checkpoint_dir = checkpoint_dir + '/' + self.model_name + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.log_dir = 'log/' + self.model_name + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            del_file(self.log_dir)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.y_dim = y_dim
        self.y = None

        self.ff_dim = ff_dim
        self.df_dim = df_dim

        self.ffc_dim = ffc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.is_batch_norm = False
        self.select_net = 0

        # if not self.y_dim:
        # self.g_bn3 = batch_norm(name='g_bn3')

        self.bestTestAcc = 0.0
        self.bestTestAcc1pre = 0.0
        self.trainThre = 0.996
        self.testThre = 0.996

        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        data_path_s = os.path.join(self.data_dir, self.dataset_name_s + self.input_fname_pattern)
        data_path_t = os.path.join(self.data_dir, self.dataset_name_t + self.input_fname_pattern)

        kind = ["Wave", "Wave_fft", "Hilbert", "Hilbert_fft"][0]
        self.data_s = load_data(data_path_s, self.y_dim, x_type=x_type)
        self.data_t = load_data(data_path_t, self.y_dim, x_type=x_type)
        self.n_x = self.data_s[0].shape[1]
        self.input_X_s = self.data_s[0].shape[1]
        self.input_Y_s = self.data_s[1].shape[1]
        self.input_X_t = self.data_s[2].shape[1]
        self.input_Y_t = self.data_s[3].shape[1]

        if self.data_s[0].shape[0] < self.batch_size:
            raise Exception("[!] Entire dataset size is less than the configured batch_size")

        self.build_model()

    def build_model(self):

        tf.set_random_seed(1)  # to keep consistent results

        with tf.variable_scope("INPUTs"):
            self.X_s = tf.placeholder(tf.float32, shape=(None, self.input_X_s), name='X_s')
            self.X_t = tf.placeholder(tf.float32, shape=(None, self.input_X_t), name='X_t')
            self.Y_s = tf.placeholder(tf.float32, shape=(None, self.input_Y_s), name='Y_s')
            self.Y_t = tf.placeholder(tf.float32, shape=(None, self.input_Y_t), name='Y_t')

        with tf.variable_scope("NETs"):

            self.F_s, self.F_s_sum = self.feature_s(self.X_s, self.y, reuse=False)
            self.F_sc, self.F_sc_sum = self.feature_c(self.F_s, self.y, reuse=False)
            self.C_s, self.C_logits_s, self.C_s_sum = self.classifier(self.F_sc, self.y, reuse=False)

            self.P_s, self.P_logits_s, self.s_list = self.predictor_s(self.X_s)
            self.P_t, self.P_logits_t, self.t_list = self.predictor_s(self.X_t)

        self.f_s_sum = histogram_summary("f_s", self.F_s)
        self.f_sc_sum = histogram_summary("f_sc", self.F_sc)
        self.c_s_sum = histogram_summary("c_s", self.C_s)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        with tf.variable_scope("LOSSs"):
            self.reg = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(2.5e-5),
                weights_list=[var for var in tf.global_variables() if "w" in var.name])

            with tf.variable_scope("Classifier_loss"):
                self.c_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.C_logits_s, labels=self.Y_s))


        self.c_loss_sum = scalar_summary("c_loss", self.c_loss)


        t_vars = tf.trainable_variables()

        self.c_vars = [var for var in t_vars if 'c_' in var.name]
        self.fs_vars = [var for var in t_vars if 'fs_' in var.name]
        self.fc_vars = [var for var in t_vars if 'fc_' in var.name]

        print(self.c_vars)


        with tf.variable_scope("Accuracy"):
            # Calculate the correct predictions scource
            correct_prediction_s = tf.equal(tf.argmax(self.P_s, 1), tf.argmax(self.Y_s, 1))
            # Calculate accuracy on the test set
            self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction_s, "float"))
            # Calculate the correct predictions target
            correct_prediction_t = tf.equal(tf.argmax(self.P_t, 1), tf.argmax(self.Y_t, 1))
            # Calculate accuracy on the test set
            self.accuracy_t = tf.reduce_mean(tf.cast(correct_prediction_t, "float"))

        self.S_accuarcy_sum = scalar_summary("S_accuarcy", self.accuracy_s)
        self.T_accuarcy_sum = scalar_summary("T_accuarcy", self.accuracy_t)

        self.merge_acc_s = [self.S_accuarcy_sum]
        self.merge_acc_t = [self.T_accuarcy_sum]

        self.merge_c = [self.f_s_sum, self.f_sc_sum, self.c_s_sum, self.c_loss_sum] + self.F_s_sum + self.F_sc_sum + self.C_s_sum

        self.saver = tf.train.Saver()

    def train(self, config):
        self.train_01(config)

    def train_00(self, config):
        pass

    def train_01(self, config):

        tf.set_random_seed(1)  # to keep consistent results

        c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.c_loss, var_list=self.fs_vars + self.fc_vars + self.c_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.c_m_sum = merge_summary(self.merge_c)

        self.accuracy_s_sum = merge_summary(self.merge_acc_s)
        self.accuracy_t_sum = merge_summary(self.merge_acc_t)

        self.merge_all = tf.summary.merge_all()
        # self.writer = SummaryWriter("./log/" + self.model_name, self.sess.graph)
        self.writer = SummaryWriter(self.log_dir, self.sess.graph)

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes = self.data_s
        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t

        X_train = X_s_train
        Y_train = Y_s_train

        X_s_data = np.vstack((X_s_train, X_s_test))
        Y_s_data = np.vstack((Y_s_train, Y_s_test))
        X_t_data = np.vstack((X_t_train, X_t_test))
        Y_t_data = np.vstack((Y_t_train, Y_t_test))
        X_test = np.vstack((X_t_train, X_t_test))
        Y_test = np.vstack((Y_t_train, Y_t_test))
        print(X_test.shape)
        print(Y_test.shape)
        # lc = min(X_s_data.shape[0], X_t_data.shape[0])
        lc = 500
        X_s_data4sum, Y_s_data4sum = X_s_data[0:lc], Y_s_data[0:lc]
        X_t_data4sum, Y_t_data4sum = X_t_data[0:lc], Y_t_data[0:lc]

        self.X_s_data4sum = X_s_data4sum
        self.Y_s_data4sum = Y_s_data4sum
        self.X_t_data4sum = X_t_data4sum
        self.Y_t_data4sum = Y_t_data4sum

        del X_s_data4sum, Y_s_data4sum, X_t_data4sum, Y_t_data4sum

        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        Epoch_C = config.epoch_pre
        counter = 0
        seed = 3  # to keep consistent results
        minibatches_s = random_mini_batches_CNN1D(X_train, Y_train, self.batch_size, seed)
        minibatches_t = random_mini_batches_CNN1D(X_t_train, Y_t_train, self.batch_size, seed)
        i_s, i_t = 0, 0
        ns = len(minibatches_s) - 2
        nt = len(minibatches_t) - 2
        # Update C network
        epoch_C_cost = []
        for epoch in range(Epoch_C):

            (minibatch_X_s, minibatch_Y_s) = minibatches_s[i_s]
            if i_s == ns:
                i_s = 0
            else:
                i_s += 1
            (minibatch_X_t, minibatch_Y_t) = minibatches_t[i_t]
            if i_t == nt:
                i_t = 0
            else:
                i_t += 1
            # print(minibatch_X_s.shape)
            # print(minibatch_X_t.shape)
            _, cost = self.sess.run([c_optim, self.c_loss],
                                    feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                               self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
            summary_str, summary_str_acc_s, summary_str_acc_t = \
                self.sess.run([self.c_m_sum,self.accuracy_s_sum,self.accuracy_t_sum],
                              feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                         self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
            self.writer.add_summary(summary_str, counter)
            self.writer.add_summary(summary_str_acc_s, counter)
            self.writer.add_summary(summary_str_acc_t, counter)
            epoch_C_cost.append(cost)
            # if counter % 5== 0:
            #     self.save(config.checkpoint_dir, counter)
            if epoch % 1 == 0:
                print("Epoch_Fs_C_net: [%2d/%2d], c_loss: %.8f" \
                      % (epoch, Epoch_C, cost))
                # print calssifier result
                r_s = self.print_accuracy_s(self.data_s)
                r_t = self.print_accuracy_t(self.data_t)
                # r_s_t = self.print_accuracy_s_t(self.data_t)
                r_s_t = 0,0
                self.write_result(counter,r_s,r_t,r_s_t)
                # self.write_Bestresult(counter, r_s, r_t, r_s_t)
                self.write_Bestresult(counter, r_s, r_t, r_s_t)

            # if counter % 500 == 0:
            #     f_c_s = self.sess.run(self.s_list[0],
            #                           feed_dict={self.X_s: self.X_s_data4sum, self.Y_s: self.Y_s_data4sum})
            #     f_c_t = self.sess.run(self.t_list[0],
            #                           feed_dict={self.X_t: self.X_t_data4sum, self.Y_t: self.Y_t_data4sum})
            #     self.visualization(f_representation_s=f_c_s, y_s=self.Y_s_data4sum, f_representation_t=f_c_t,
            #                        y_t=self.Y_t_data4sum, step=counter)
            #     del f_c_s, f_c_t
            counter += 1
            # if counter % 50== 0:
            #     self.save(config.checkpoint_dir, counter)
        name = 'BestAccuracy'
        self.save_best(name=name)


    def feature_s(self, x, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        return self.feature_s_base(x, y, reuse=reuse, is_train=is_train, with_sum=with_sum, with_list=with_list)

    def feature_s_base02(self, x, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("feature_source") as scope:
            if reuse:
                scope.reuse_variables()
            x_in = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1])

            h0, w0, b0 = conv1d(x_in, self.ff_dim/4, k_l=5,d_l = 1, name='fs_h0_conv', with_w=True)
            # h0 = tf.nn.tanh(h0)
            h0 = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            output = max_pool1d(h0, name='fs_h0_pool')

            h1, w1, b1 = conv1d(output, self.ff_dim/2, k_l=3,d_l = 1, name='fs_h1_conv', with_w=True)
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.relu(h1)
            # h1 = lrelu(h1)
            output = max_pool1d(h1, name='fs_h1_pool')

            h2, w2, b2 = conv1d(output, self.ff_dim , k_l=3,d_l = 1, name='fs_h2_conv', with_w=True)
            h2 = tf.nn.relu(h2)
            output = max_pool1d(h2, name='fs_h2_pool')

            # h3, w3, b3 = conv1d(output, self.ff_dim * 2, name='fs_h3_conv', with_w=True)
            # h3 = tf.nn.relu(h3)
            # output = max_pool1d(h3, name='fs_h3_pool')

            h0_sum = histogram_summary("fs_h0", h0)
            # h1_sum = histogram_summary("fs_h1", h1)
            # h2_sum = histogram_summary("fs_h2", h2)

            feature_sum_s = [h0_sum]

            if with_sum:
                return output, feature_sum_s
            else:
                return output

    def feature_s_base01(self, x, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("feature_source") as scope:
            if reuse:
                scope.reuse_variables()
            x_in = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1])

            h0, w0, b0 = conv1d(x_in, output_dim=self.ff_dim/4, k_l=64,d_l = 2, name='fs_h0_conv', with_w=True)
            # h0 = tf.nn.tanh(h0)
            h0 = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            output = max_pool1d(h0, name='fs_h0_pool')

            h1, w1, b1 = conv1d(output, output_dim=self.ff_dim/2, k_l=3,d_l = 1, name='fs_h1_conv', with_w=True)
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.relu(h1)
            # h1 = lrelu(h1)
            output = max_pool1d(h1, name='fs_h1_pool')

            h2, w2, b2 = conv1d(output, output_dim=self.ff_dim, k_l=3, d_l = 1,  name='fs_h2_conv', with_w=True)
            h2 = tf.nn.relu(h2)
            output = max_pool1d(h2, name='fs_h2_pool')

            h3, w3, b3 = conv1d(output, output_dim=self.ff_dim*2, k_l=3, d_l = 1, name='fs_h3_conv', with_w=True)
            h3 = tf.nn.relu(h3)
            output = max_pool1d(h3, name='fs_h3_pool')

            h4, w4, b4 = conv1d(output, output_dim=self.ff_dim*2, k_l=3, d_l = 1, name='fs_h4_conv', with_w=True)
            h4 = tf.nn.relu(h4)
            output = max_pool1d(h4, name='fs_h4_pool')

            h0_sum = histogram_summary("fs_h0", h0)
            # h1_sum = histogram_summary("fs_h1", h1)
            # h2_sum = histogram_summary("fs_h2", h2)

            feature_sum_s = [h0_sum]

            if with_sum:
                return output, feature_sum_s
            else:
                return output

    def feature_s_base(self, x, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_source") as scope:
            if reuse:
                scope.reuse_variables()
            x_in = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1])

            h0, w0, b0 = conv1d(x_in, output_dim=self.ff_dim/4, k_l=64,d_l = 2, name='fs_h0_conv', with_w=True)
            # h0 = tf.nn.tanh(h0)
            h0 = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            output = max_pool1d(h0, name='fs_h0_pool')

            h1, w1, b1 = conv1d(output, output_dim=self.ff_dim/2, k_l=32,d_l = 2, name='fs_h1_conv', with_w=True)
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.relu(h1)
            # h1 = lrelu(h1)
            output = max_pool1d(h1, name='fs_h1_pool')

            h2, w2, b2 = conv1d(output, output_dim=self.ff_dim, k_l=16,d_l = 2,  name='fs_h2_conv', with_w=True)
            h2 = tf.nn.relu(h2)
            output = max_pool1d(h2, name='fs_h2_pool')

            h3, w3, b3 = conv1d(output, output_dim=self.ff_dim*2, k_l=8,d_l = 2, name='fs_h3_conv', with_w=True)
            h3 = tf.nn.relu(h3)
            output = max_pool1d(h3, name='fs_h3_pool')

            h4, w4, b4 = conv1d(output, output_dim=self.ff_dim*2, k_l=8,d_l = 2, name='fs_h4_conv', with_w=True)
            h4 = tf.nn.relu(h4)
            # h4 = tf.nn.tanh(h4)
            output = max_pool1d(h4, name='fs_h4_pool')

            h0_sum = histogram_summary("fs_h0", h0)
            # h1_sum = histogram_summary("fs_h1", h1)
            # h2_sum = histogram_summary("fs_h2", h2)

            feature_sum_s = [h0_sum]
            h_list = [output]

            # if with_sum:
            #     return output, feature_sum_s
            # else:
            #     return output

            if with_sum and not with_list:
                return output, feature_sum_s
            elif with_list and not with_sum:
                return output, h_list
            elif with_sum and with_list:
                return output, feature_sum_s, h_list
            else:
                return output

    def feature_c(self, x, y=None, reuse=False, is_train=True, with_sum=True ,with_list=False):

        return self.feature_c_base(x, y, reuse=reuse, is_train=is_train, with_sum=with_sum, with_list=with_list)

    def feature_c_base02(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_target") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = fully_conn(f, self.ffc_dim/8, name='fc_h0_conn')
            h0 = tf.nn.tanh(h0)
            # h0 = tf.nn.relu(h0)

            h0_sum = histogram_summary("fc_h0", h0)
            feature_sum_c = [h0_sum]
            h_list = [h0]
            output_ = h0

            if with_sum and not with_list:
                return output_, feature_sum_c
            elif with_list and not with_sum:
                return output_, h_list
            elif with_sum and with_list:
                return output_, feature_sum_c, h_list
            else:
                return output_

    def feature_c_base01(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_target") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = fully_conn(f, self.ffc_dim/4, name='fc_h0_conn')
            h0 = tf.nn.tanh(h0)
            output_ = h0
            # h0 = tf.nn.relu(h0)

            h1 = fully_conn(h0, self.ffc_dim/16, name='fc_h1_conn')
            h1 = tf.nn.tanh(h1)
            output_ = h1

            h0_sum = histogram_summary("fc_h0", h0)
            feature_sum_c = [h0_sum]
            h_list = [h0]
            output_ = h0

            if with_sum and not with_list:
                return output_, feature_sum_c
            elif with_list and not with_sum:
                return output_, h_list
            elif with_sum and with_list:
                return output_, feature_sum_c, h_list
            else:
                return output_

    def feature_c_base(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_target") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = fully_conn(f, self.ffc_dim/2, name='fc_h0_conn')
            h0 = tf.nn.tanh(h0)
            # h0 = tf.nn.relu(h0)

            h0_sum = histogram_summary("fc_h0", h0)
            feature_sum_c = [h0_sum]
            h_list = [h0]
            output_ = h0

            if with_sum and not with_list:
                return output_, feature_sum_c
            elif with_list and not with_sum:
                return output_, h_list
            elif with_sum and with_list:
                return output_, feature_sum_c, h_list
            else:
                return output_

    def classifier(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list = False):

        return self.classifier_output(f, y, reuse=reuse, is_train=is_train, with_sum=with_sum,  with_list=with_list)

    def classifier_output(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("classifier") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = linear(f, self.y_dim, scope='c_h0_lin')

            h0_sum = histogram_summary("c_h0", h0)
            classifier_sum = [h0_sum]
            h_list = [tf.nn.softmax(h0)]
            output = h0
            if with_sum and not with_list:
                return tf.nn.softmax(output), output, classifier_sum
            elif with_list and not with_sum:
                return tf.nn.softmax(output), output, h_list
            elif with_sum and with_list:
                return tf.nn.softmax(output), output, classifier_sum, h_list
            else:
                return tf.nn.softmax(output), output

    def predictor_s(self, x, y=None):

        F_s, F_s_list = self.feature_s(x, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        F_sc, F_sc_list = self.feature_c(F_s, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        P, P_logits, C_list = self.classifier(F_sc, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        h_list = F_s_list + F_sc_list + C_list
        return P, P_logits, h_list

    def predictor(self, x, y=None):

        F_s = self.feature_s(x, y, reuse=True, is_train=False, with_sum=False)
        F_c, F_c_list = self.feature_c(F_s, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        P, P_logits, C_list = self.classifier(F_c, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        h_list = []
        h_list.append(F_c_list[0])
        h_list.append(C_list[0])

        return P, P_logits, h_list

    def cal_distance(self, x1, x2):
        x1_in = tf.reshape(x1, [-1, x1.get_shape().as_list()[1], 1])
        x2_in = tf.reshape(x2, [-1, x2.get_shape().as_list()[1], 1])

        Dis = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x1_in - x2_in), 1)))

        return Dis

    def print_accuracy_s(self, data):
        data_s = data

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes_s = data_s

        # r =  self.accuracy_s.eval({self.X_s: X_s_train, self.Y_s: Y_s_train})
        r = self.sess.run(self.accuracy_s, feed_dict={self.X_s: X_s_train, self.Y_s: Y_s_train})
        r_c = self.get_multicalss_accuracy_s(X_s_train, Y_s_train, classes_s)
        # print(" Source Train Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Source Train Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_train = r
        r_c_train = r_c

        # r =  self.accuracy_s.eval({self.X_s: X_s_test, self.Y_s: Y_s_test})
        r = self.sess.run(self.accuracy_s, feed_dict={self.X_s: X_s_test, self.Y_s: Y_s_test})
        r_c = self.get_multicalss_accuracy_s(X_s_test, Y_s_test, classes_s)
        # print(" Source Test Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Source Test Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_test = r
        r_c_test = r_c

        X_data = np.vstack((X_s_train, X_s_test))
        Y_data = np.vstack((Y_s_train, Y_s_test))

        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_data, self.Y_t: Y_data})
        r_c = self.get_multicalss_accuracy_t(X_data, Y_data, classes_s)
        # print(" Target      Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Source      Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))

        return r_train, r_test, r, r_c_train, r_c_test, r_c

    def print_accuracy_t(self, data):

        data_t = data

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes_t = data_t

        # r =  self.accuracy_t.eval({self.X_t: X_t_train, self.Y_t: Y_t_train})
        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_t_train, self.Y_t: Y_t_train})
        r_c = self.get_multicalss_accuracy_t(X_t_train, Y_t_train, classes_t)
        print(" Target Train Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_train = r
        r_c_train = r_c

        # r =  self.accuracy_t.eval({self.X_t: X_t_test, self.Y_t: Y_t_test})
        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_t_test, self.Y_t: Y_t_test})
        r_c = self.get_multicalss_accuracy_t(X_t_test, Y_t_test, classes_t)
        # print(" Target Test Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Target Test Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_test = r
        r_c_test = r_c

        X_data = np.vstack((X_t_train, X_t_test))
        Y_data = np.vstack((Y_t_train, Y_t_test))

        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_data, self.Y_t: Y_data})
        r_c = self.get_multicalss_accuracy_t(X_data, Y_data, classes_t)
        # print(" Target      Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Target      Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))


        return r_train, r_test, r, r_c_train, r_c_test,r_c

    def get_multicalss_accuracy_s(self, X, Y, Classes):
        mc_accuracy = []
        index = np.argmax(Y, axis=1)
        # for i in yy:
        # print(i)
        # print(yy.shape)
        # print(Y_test.shape)
        test_data_X = pd.DataFrame(X, index=index)
        test_data_Y = pd.DataFrame(Y, index=index)

        for i in range(Classes):
            X_t = np.array(test_data_X.ix[i])
            Y_t = np.array(test_data_Y.ix[i])
            # print(i)
            # print(X_t.shape)
            # print(Y_t.shape)
            r = self.sess.run(self.accuracy_s, feed_dict={self.X_s: X_t, self.Y_s: Y_t})
            mc_accuracy.append(r)
            # mc_accuracy.append(self.accuracy_s.eval({self.X_s: X_t, self.Y_s: Y_t}))
            # print ("Class 0: Test Accuracy:", accuracy.eval({X_s: X_t, Y_s: Y_t, keep_prob:k_prob}))

        return mc_accuracy

    def get_multicalss_accuracy_t(self, X, Y, Classes):
        mc_accuracy = []
        index = np.argmax(Y, axis=1)
        # for i in yy:
        # print(i)
        # print(yy.shape)
        # print(Y_test.shape)
        test_data_X = pd.DataFrame(X, index=index)
        test_data_Y = pd.DataFrame(Y, index=index)

        for i in range(Classes):
            X_t = np.array(test_data_X.ix[i])
            Y_t = np.array(test_data_Y.ix[i])
            r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_t, self.Y_t: Y_t})
            mc_accuracy.append(r)
            # mc_accuracy.append(self.accuracy_t.eval({self.X_t: X_t, self.Y_t: Y_t}))
            # print ("Class 0: Test Accuracy:", accuracy.eval({X_s: X_t, Y_s: Y_t, keep_prob:k_prob}))

        return mc_accuracy


    def write_result(self, idx, r_s, r_t,r_s_t):
        path = self.save_path
        # name = self.model_name + '_Accuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t + '.txt'
        name = 'Accuracy'
        name_txt = name + '.txt'
        if idx == 0:
            with open(os.path.join(path, name_txt), 'w') as Record_f:
                Record_f.write('idx' + '\t')
                Record_f.write('Train_S' + '\t')
                Record_f.write('Test_S' + '\t')
                Record_f.write('All_S' + '\t')
                Record_f.write('Train_T' + '\t')
                Record_f.write('Test_T' + '\t')
                Record_f.write('All_T' + '\t')
                Record_f.write('Train_T_10' + '\t')
                Record_f.write('Test_T_10' + '\t')
                Record_f.write('All_T_10' + '\t')
                Record_f.write('Other1' + '\t')
                Record_f.write('Other2'+ '\n')

                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

        else:
            with open(os.path.join(path, name_txt), 'a') as Record_f:
                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

    # analysis 4 source

    def write_Bestresult_Source(self, idx, r_s, r_t,r_s_t):
        path = self.save_path
        # name = self.model_name + '_BestAccuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t
        name = 'BestAccuracy_Source'
        name_txt = name + '.txt'
        testAcc = r_s[1]
        trainAcc = r_s[0]
        if testAcc > self.bestTestAcc and trainAcc >= self.trainThre:
            self.bestTestAcc = testAcc
            self.writeROCandAUC_Source()
            self.writeReport_Source()
            self.save_best(name=name)

            with open(os.path.join(path, name_txt), 'w') as Record_f:
                Record_f.write('idx' + '\t')
                Record_f.write('Train_S' + '\t')
                Record_f.write('Test_S' + '\t')
                Record_f.write('All_S' + '\t')
                Record_f.write('Train_T' + '\t')
                Record_f.write('Test_T' + '\t')
                Record_f.write('All_T' + '\t')
                Record_f.write('Train_T_10' + '\t')
                Record_f.write('Test_T_10' + '\t')
                Record_f.write('All_T_10' + '\t')
                Record_f.write('Other1' + '\t')
                Record_f.write('Other2'+ '\n')

                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

    def writeROCandAUC_Source(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t
        name = 'ROCandAUC_Source'
        name_txt = name + '.txt'
        name_figure = name + '.jpeg'
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr['macro'], tpr['macro'], roc_auc['macro'] = self.cal_ROCandAUC_Source(sign_type='macro')
        fpr['micro'], tpr['micro'], roc_auc['micro'] = self.cal_ROCandAUC_Source(sign_type='micro')

        # print(fpr['macro'].shape)
        # print(tpr['macro'].shape)
        # print(fpr['micro'].shape)
        # print(tpr['micro'].shape)

        set_type = ['macro', 'micro']
        for s_t in set_type:
            with open(os.path.join(path, s_t + '_' + name_txt), 'w') as Record_f:
                Record_f.write(str(roc_auc[s_t]) + '\t')
                Record_f.write('fpr' + '\t')
                Record_f.write('tpr' + '\n')
                m = fpr[s_t].shape[0]
                for i in range(m):
                    Record_f.write(str(i) + '\t')
                    Record_f.write(str(fpr[s_t][i]) + '\t')
                    Record_f.write(str(tpr[s_t][i]) + '\n')
        self.plot4check(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                        path_name=os.path.join(path, name_figure))

    def cal_ROCandAUC_Source(self, sign_type='micro'):

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes = self.data_s
        # y_score = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        y_score = self.sess.run(self.P_logits_s, feed_dict={self.X_s: X_s_test})

        if sign_type is 'micro':
            fpr, tpr, thresholds = roc_curve(Y_s_test.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        elif sign_type is 'macro':
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classes):
                fpr[i], tpr[i], _ = roc_curve(Y_s_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= classes
            fpr = all_fpr
            tpr = mean_tpr
            roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def writeReport_Source(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t + '_Report.txt'
        name = 'ROCandAUC_Report_Source'
        name_txt = name + '.txt'

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes = self.data_s
        y_val_pred = self.sess.run(self.P_s, feed_dict={self.X_s: X_s_test})
        report = classification_report(y_true=np.argmax(Y_s_test, axis=1), y_pred=np.argmax(y_val_pred, axis=1),
                                       digits=4)

        with open(os.path.join(path, name_txt), 'w') as Record_f:
            for r in report:
                Record_f.write(r)

    def write_Bestresult(self, idx, r_s, r_t,r_s_t):
        path = self.save_path
        # name = self.model_name + '_BestAccuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t
        name = 'BestAccuracy'
        name_txt = name + '.txt'
        tempbestAcc = r_t[1]
        testAcc = r_s[1]
        trainAcc = r_s[1]
        if tempbestAcc > self.bestTestAcc and trainAcc >= self.trainThre and testAcc >= self.testThre:
            self.bestTestAcc = tempbestAcc
            self.writeROCandAUC()
            self.writeReport()
            self.save_best(name=name)

            with open(os.path.join(path, name_txt), 'w') as Record_f:
                Record_f.write('idx' + '\t')
                Record_f.write('Train_S' + '\t')
                Record_f.write('Test_S' + '\t')
                Record_f.write('All_S' + '\t')
                Record_f.write('Train_T' + '\t')
                Record_f.write('Test_T' + '\t')
                Record_f.write('All_T' + '\t')
                Record_f.write('Train_T_10' + '\t')
                Record_f.write('Test_T_10' + '\t')
                Record_f.write('All_T_10' + '\t')
                Record_f.write('Other1' + '\t')
                Record_f.write('Other2'+ '\n')

                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')


    def writeROCandAUC(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t
        name = 'ROCandAUC'
        name_txt = name + '.txt'
        name_figure = name + '.jpeg'
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr['macro'], tpr['macro'], roc_auc['macro'] = self.cal_ROCandAUC(sign_type='macro')
        fpr['micro'], tpr['micro'], roc_auc['micro'] = self.cal_ROCandAUC(sign_type='micro')

        # print(fpr['macro'].shape)
        # print(tpr['macro'].shape)
        # print(fpr['micro'].shape)
        # print(tpr['micro'].shape)

        set_type = ['macro', 'micro']
        for s_t in set_type:
            with open(os.path.join(path, s_t + '_' + name_txt), 'w') as Record_f:
                Record_f.write(str(roc_auc[s_t]) + '\t')
                Record_f.write('fpr' + '\t')
                Record_f.write('tpr' + '\n')
                m = fpr[s_t].shape[0]
                for i in range(m):
                    Record_f.write(str(i) + '\t')
                    Record_f.write(str(fpr[s_t][i]) + '\t')
                    Record_f.write(str(tpr[s_t][i]) + '\n')
        self.plot4check(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                   path_name=os.path.join(path, name_figure))

    def cal_ROCandAUC(self, sign_type='micro'):

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t
        # y_score = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        y_score = self.sess.run(self.P_logits_t, feed_dict={self.X_t: X_t_test})

        if sign_type is 'micro':
            fpr, tpr, thresholds = roc_curve(Y_t_test.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        elif sign_type is 'macro':
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classes):
                fpr[i], tpr[i], _ = roc_curve(Y_t_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= classes
            fpr = all_fpr
            tpr = mean_tpr
            roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def writeReport(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t + '_Report.txt'
        name = 'ROCandAUC_Report'
        name_txt = name + '.txt'

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t
        y_val_pred = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        report = classification_report(y_true=np.argmax(Y_t_test,axis=1),y_pred=np.argmax(y_val_pred,axis=1),digits=4)

        with open(os.path.join(path,name_txt), 'w') as Record_f:
            for r in report:
                Record_f.write(r)

    def plot4check(self, fpr, tpr, roc_auc, path_name):
        lw = 2
        plt.ion()
        plt.figure()
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area={0:0.2f})' ''.format(roc_auc['micro']),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr['macro'], tpr['macro'],
                 label='macro-average ROC curve (area={0:0.2f})' ''.format(roc_auc['macro']),
                 color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc='lower right')
        plt.savefig(path_name)
        # plt.show()
        plt.close()
        plt.ioff()
        # plt.pause(30)

    def updataAccuracy(self,r_s,r_t):
        S_train = self.Record_Accuracy['S_train']
        S_test = self.Record_Accuracy['S_test']
        S_all = self.Record_Accuracy['S_all']
        T_train = self.Record_Accuracy['T_train']
        T_test = self.Record_Accuracy['T_test']
        T_all = self.Record_Accuracy['T_all']

        n = len(S_train)
        thre = 99.5
        if n >= 100:
            if np.mean([np.mean(S_train), np.mean(S_test), np.mean(S_all)]) > thre:
                self.terminal_idx += 1

        S_train.append(r_s[0])
        S_test.append(r_s[1])
        S_all.append(r_s[2])
        T_train.append(r_t[0])
        T_test.append(r_t[1])
        T_all.append(r_t[2])

        self.Record_Accuracy['S_train'] = S_train
        self.Record_Accuracy['S_test'] = S_test
        self.Record_Accuracy['S_all'] = S_all
        self.Record_Accuracy['T_train'] = T_train
        self.Record_Accuracy['T_test'] = T_test
        self.Record_Accuracy['T_all'] = T_all

    def plotAccuracy4check(self):

        path = self.save_path
        name_figure = 'RecordAccuracy' + '.jpeg'
        path_name = os.path.join(path, name_figure)
        name_set = ['S_train','S_test','S_all',
                    'T_train','T_test','T_all']

        plt.ion()

        figure, ax = plt.subplots()
        for name in name_set:
            plt.plot(self.Record_Accuracy[name],
                     label= name,
                     linestyle='-', linewidth=3)


        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 12, }
        # 设置横纵坐标的名称以及对应字体格式
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        plt.ylim([0.0, 1.05])

        # 设置坐标刻度值的大小以及刻度值的字体
        plt.tick_params(labelsize=14)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        # print labels
        [label.set_fontname('Times New Roman') for label in labels]

        plt.xlabel('Accuracy')
        plt.ylabel('Epoch')
        plt.legend(loc='lower right',prop=font1)
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        # plt.legend(loc='lower right')
        plt.savefig(path_name)
        # plt.show()
        plt.close()
        plt.ioff()

    def save_best(self, name, step=100):

        # checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name)
        #
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, name),
                        global_step=step)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name_s,
            self.dataset_name_t[18:], self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = self.model_name
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualization(self, f_representation_s, y_s, f_representation_t, y_t, step):

        # Visualization of trained flatten layer (T-SNE)
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        tsne = TSNE(perplexity=30, n_components=2, n_iter=5000)
        plot_only = 100
        f_r = np.vstack([f_representation_s[:plot_only, :],f_representation_t[:plot_only, :]])
        # y = np.hstack([y_s[:plot_only],y_t[:plot_only]])
        low_dim_embs = tsne.fit_transform(f_r)

        low_dim_embs_s = low_dim_embs[0:plot_only,:]
        low_dim_embs_t = low_dim_embs[plot_only:2*plot_only,:]
        labels_s = np.argmax(y_s, axis=1)[:plot_only]
        labels_t = np.argmax(y_t, axis=1)[:plot_only]

        # low_dim_embs_s = tsne.fit_transform(f_representation_s[:plot_only, :])
        # labels_s = np.argmax(y_s, axis=1)[:plot_only]
        # low_dim_embs_t = tsne.fit_transform(f_representation_t[:plot_only, :])
        # labels_t = np.argmax(y_t, axis=1)[:plot_only]
        # plt.cla()
        fig_s = plt.figure(num='fig_s')
        plt.ion()
        fig_t = plt.figure(num='fig_t')
        plt.ion()
        self.plot_with_labels_st(fig_s=fig_s,lowDWeights_s=low_dim_embs_s,labels_s=labels_s,fig_t=fig_t,lowDWeights_t=low_dim_embs_t,labels_t=labels_t)
        # self.plot_with_labels(low_dim_embs_s, labels_s,ifsource=True)
        # self.plot_with_labels(low_dim_embs_t, labels_t,ifsource=False)
        path = 'visualizations'
        path = os.path.join(path, self.model_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        # plt.savefig(path +'/'+ self.model_name+'-' + str(step))
        fig_s.savefig(path + '/' + self.model_name + '-' + str(step) + '-Source')
        fig_t.savefig(path + '/' + self.model_name + '-' + str(step) + '-Target')
        plt.figure(num='fig_s')
        plt.ioff()
        plt.close()
        plt.figure(num='fig_t')
        plt.ioff()
        plt.close()

        # plt.show()

    def plot_with_labels_st(self,fig_s,lowDWeights_s, labels_s, fig_t, lowDWeights_t, labels_t):
        # plt.cla()
        X_s, Y_s = lowDWeights_s[:, 0], lowDWeights_s[:, 1]
        X_t, Y_t = lowDWeights_t[:, 0], lowDWeights_t[:, 1]
        x_min = min(X_s.min(),X_t.min())
        x_max = max(X_s.max(),X_t.max())
        y_min = min(Y_s.min(),Y_t.min())
        y_max = max(Y_s.max(),Y_t.max())
        for x_s, y_s, s_s in zip(X_s, Y_s, labels_s):
            c_s = cm.rainbow(int(255 * s_s / self.y_dim))
            plt.figure(num='fig_s')
            # plt.text(x_s, y_s, s_s, backgroundcolor=c_s, fontsize=9)
            plt.text(x_s, y_s, s_s, fontsize=9, bbox=dict(boxstyle='square', fc=c_s))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title('Visualize Source Full_conn layer')
            plt.show()
            plt.pause(0.000001)

        for x_s, y_s, s_s, x_t, y_t, s_t in zip(X_s, Y_s, labels_s, X_t, Y_t, labels_t):
            c_t = cm.rainbow(int(255 * s_t / self.y_dim))
            plt.figure(num='fig_t')
            plt.text(x_t, y_t, s_t, fontsize=9, bbox=dict(boxstyle='circle', fc=c_t))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title('Visualize Target Full_conn layer')
            plt.show()
            plt.pause(0.000001)

        # for x_s, y_s, s_s, x_t, y_t, s_t in zip(X_s, Y_s, labels_s, X_t, Y_t, labels_t):
        #     c_s = cm.rainbow(int(255 * s_s / self.y_dim))
        #     c_t = cm.rainbow(int(255 * s_t / self.y_dim))
        #     plt.figure(num='fig_s')
        #     # plt.text(x_s, y_s, s_s, backgroundcolor=c_s, fontsize=9)
        #     plt.text(x_s, y_s, s_s, fontsize=9, bbox=dict(boxstyle='square', fc=c_s))
        #     plt.xlim(x_min, x_max)
        #     plt.ylim(y_min, y_max)
        #     plt.title('Visualize Source Full_conn layer')
        #     plt.show()
        #     plt.pause(0.000001)
        #     plt.figure(num='fig_t')
        #     plt.text(x_t, y_t, s_t, fontsize=9, bbox=dict(boxstyle='circle', fc=c_t))
        #     plt.xlim(x_min, x_max)
        #     plt.ylim(y_min, y_max)
        #     plt.title('Visualize Target Full_conn layer')
        #     plt.show()
        #     plt.pause(0.000001)

    def plot_with_labels(self,lowDWeights, labels, ifsource = True):
        # plt.cla()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]

        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 * s / 4))
            if ifsource:
                plt.text(x, y, s, backgroundcolor=c, fontsize=9)
            else:
                plt.text(x, y, s, fontsize=9, bbox=dict(boxstyle='circle', fc=c))
            # plt.xlim(X.min(), X.max())
            # plt.ylim(Y.min(), Y.max())
            plt.title('Visualize last layer')
            plt.show()
            plt.pause(0.0001)


class MAAN(object):

    def __init__(self, sess,
                 crop=True, batch_size=64, sample_num=64,
                 y_dim=4, ff_dim=32, df_dim=32, ffc_dim=1024, dfc_dim=1024,
                 dataset_name_s='default_s', dataset_name_t='default_t',
                 input_fname_pattern='.h5', checkpoint_dir=None,
                 data_dir='.\\datasets' or './datasets', x_type = None):

        """

        Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        y_dim: (optional) Dimension of dim for y. [None]
        z_dim: (optional) Dimension of dim for Z. [100]
        ff_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]

        """
        # the discriminator net affect on the classifier net

        self.sess = sess
        self.crop = crop
        self.model_name = "HANs" + '_' + x_type
        self.dataset_name_s = dataset_name_s
        self.dataset_name_t = dataset_name_t

        self.save_path = 'results/' + self.model_name + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.checkpoint_dir = checkpoint_dir + '/' + self.model_name + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.log_dir = 'log/' + self.model_name + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            del_file(self.log_dir)

        self.base_model = "base_" + self.model_name
        self.base_dir = checkpoint_dir + '/' + self.base_model + '/' + self.dataset_name_s + '_' + self.dataset_name_t
        # self.load_name = 'BestAccuracy-100'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.batch_size = batch_size
        self.sample_num = sample_num
        self.epoch = 0
        self.epochWriteThre = 1

        self.y_dim = y_dim
        self.y = None

        self.ff_dim = ff_dim
        self.df_dim = df_dim

        self.ffc_dim = ffc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.is_batch_norm = False
        self.select_net = 0

        # if not self.y_dim:
        # self.g_bn3 = batch_norm(name='g_bn3')

        self.bestTestAcc = 0.0
        self.bestAccuracy_sign = False
        self.bestTestAcc1pre = 0.0
        self.bestTestAcc_source = 0.0
        self.trainThre = 0.94
        self.testThre = 0.94
        self.trainThre_pre = 0.999
        self.testThre_pre = 0.999
        # self.Record_Accuracy = {'S_train': [0], 'S_test': [0], 'S_all': [0],
        #                         'T_train': [0], 'T_test': [0], 'T_all': [0]}
        self.initAccuracy()
        # self.Record_WD = {'wd01': [0], 'wd02': [0]}
        self.initWD()
        self.terminal_idx = 0

        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        data_path_s = os.path.join(self.data_dir, self.dataset_name_s + self.input_fname_pattern)
        data_path_t = os.path.join(self.data_dir, self.dataset_name_t + self.input_fname_pattern)

        kind = ["Wave", "Wave_fft", "Hilbert", "Hilbert_fft"][0]
        self.data_s = load_data(data_path_s, self.y_dim, x_type=x_type)
        self.data_t = load_data(data_path_t, self.y_dim, x_type=x_type)
        self.n_x = self.data_s[0].shape[1]
        self.input_X_s = self.data_s[0].shape[1]
        self.input_Y_s = self.data_s[1].shape[1]
        self.input_X_t = self.data_s[2].shape[1]
        self.input_Y_t = self.data_s[3].shape[1]

        if self.data_s[0].shape[0] < self.batch_size:
            raise Exception("[!] Entire dataset size is less than the configured batch_size")

        self.build_model()

    def build_model(self):

        tf.set_random_seed(1)  # to keep consistent results

        with tf.variable_scope("INPUTs"):
            self.X_s = tf.placeholder(tf.float32, shape=(None, self.input_X_s), name='X_s')
            self.X_t = tf.placeholder(tf.float32, shape=(None, self.input_X_t), name='X_t')
            self.Y_s = tf.placeholder(tf.float32, shape=(None, self.input_Y_s), name='Y_s')
            self.Y_t = tf.placeholder(tf.float32, shape=(None, self.input_Y_t), name='Y_t')
            self.l_r = tf.placeholder(tf.float32)

        with tf.variable_scope("NETs"):

            self.F_s, self.F_s_sum = self.feature_s(self.X_s, self.y, reuse=False)
            self.F_sc, self.F_sc_sum = self.feature_c(self.F_s, self.y, reuse=False)
            self.C_s, self.C_logits_s, self.C_s_sum = self.classifier(self.F_sc, self.y, reuse=False)

            self.P_s, self.P_logits_s, self.s_list = self.predictor_s(self.X_s)
            self.P_t, self.P_logits_t, self.t_list = self.predictor_s(self.X_t)

            s_temp_f = self.s_list[0]
            t_temp_f = self.t_list[0]
            self.Df_s, self.Df_logits_s, self.Df_s_sum = self.discriminator_f(s_temp_f, self.y, reuse=False)
            self.Df_t, self.Df_logits_t, self.Df_t_sum = self.discriminator_f(t_temp_f, self.y, reuse=True)
            self.Dis_f = self.cal_distance(s_temp_f, t_temp_f)
            self.gp_f = self.gradient_penalty(self.discriminator_f, s_temp_f, t_temp_f)

            s_temp_c01 = self.s_list[1]
            t_temp_c01 = self.t_list[1]
            self.Dc01_s, self.Dc01_logits_s, self.Dc01_s_sum = self.discriminator_c01(s_temp_c01, self.y, reuse=False)
            self.Dc01_t, self.Dc01_logits_t, self.Dc01_t_sum = self.discriminator_c01(t_temp_c01, self.y, reuse=True)
            self.Dis_c01 = self.cal_distance(s_temp_c01, t_temp_c01)
            self.gp_c01 = self.gradient_penalty(self.discriminator_c01, s_temp_c01, t_temp_c01)

            s_temp_c02 = self.s_list[2]
            t_temp_c02 = self.t_list[2]
            self.Dc02_s, self.Dc02_logits_s, self.Dc02_s_sum = self.discriminator_c02(s_temp_c02, self.y, reuse=False)
            self.Dc02_t, self.Dc02_logits_t, self.Dc02_t_sum = self.discriminator_c02(t_temp_c02, self.y, reuse=True)
            self.Dis_c02 = self.cal_distance(s_temp_c02, t_temp_c02)
            self.gp_c02 = self.gradient_penalty(self.discriminator_c02, s_temp_c02, t_temp_c02)

        self.f_s_sum = histogram_summary("f_s", self.F_s)
        self.f_sc_sum = histogram_summary("f_sc", self.F_sc)
        self.c_s_sum = histogram_summary("c_s", self.C_s)

        self.s_distribution_f = histogram_summary("s_distribution_f", s_temp_f)
        self.t_distribution_f = histogram_summary("t_distribution_f", t_temp_f)
        self.df_s_sum = histogram_summary("df_s", self.Df_logits_s)
        self.df_t_sum = histogram_summary("df_t", self.Df_logits_t)

        self.s_distribution_c01 = histogram_summary("s_distribution_c01", s_temp_c01)
        self.t_distribution_c01 = histogram_summary("t_distribution_c01", t_temp_c01)
        self.dc01_s_sum = histogram_summary("dc01_s", self.Dc01_logits_s)
        self.dc01_t_sum = histogram_summary("dc01_t", self.Dc01_logits_t)

        self.s_distribution_c02 = histogram_summary("s_distribution_c02", s_temp_c02)
        self.t_distribution_c02 = histogram_summary("t_distribution_c02", t_temp_c02)
        self.dc02_s_sum = histogram_summary("dc02_s", self.Dc02_logits_s)
        self.dc02_t_sum = histogram_summary("dc02_t", self.Dc02_logits_t)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        with tf.variable_scope("LOSSs"):
            self.reg = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(2.5e-5),
                weights_list=[var for var in tf.global_variables() if "w" in var.name])

            with tf.variable_scope("Classifier_loss"):
                self.c_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.C_logits_s, labels=self.Y_s))

            with tf.variable_scope("Discriminator_f_loss"):
                self.df_loss_s = tf.reduce_mean(self.Df_logits_s)
                self.df_loss_t = tf.reduce_mean(self.Df_logits_t)
                self.wd_f = self.df_loss_s - self.df_loss_t
                self.df_loss = -(self.wd_f) + self.gp_f

            with tf.variable_scope("Feature_f_loss"):
                self.ff_loss = -tf.reduce_mean(self.Df_logits_t)
                self.ff_c_loss = 0.5*self.ff_loss + self.c_loss

            with tf.variable_scope("Discriminator_c01_loss"):
                self.dc01_loss_s = tf.reduce_mean(self.Dc01_logits_s)
                self.dc01_loss_t = tf.reduce_mean(self.Dc01_logits_t)
                # self.wd_c01 = self.dc01_loss_s - self.dc01_loss_t
                self.wd_c01 = tf.abs(self.dc01_loss_s - self.dc01_loss_t)
                self.dc01_loss = -(self.wd_c01) + self.gp_c01

            with tf.variable_scope("Feature_c01_loss"):
                self.fc01_loss = -tf.reduce_mean(self.Dc01_logits_t)
                # self.fc01_c_loss = 0.5*self.fc01_loss + self.c_loss
                self.fc01_c_loss = self.wd_c01 + self.c_loss
                # self.fc01_c_loss = -self.dc01_loss + self.c_loss

            with tf.variable_scope("Discriminator_c02_loss"):
                self.dc02_loss_s = tf.reduce_mean(self.Dc02_logits_s)
                self.dc02_loss_t = tf.reduce_mean(self.Dc02_logits_t)
                # self.wd_c02 = self.dc02_loss_s - self.dc02_loss_t
                self.wd_c02 = tf.abs(self.dc02_loss_s - self.dc02_loss_t)
                self.dc02_loss = -(self.wd_c02) + self.gp_c02

            with tf.variable_scope("Feature_c02_loss"):
                self.fc02_loss = -tf.reduce_mean(self.Dc02_logits_t)
                # self.fc02_c_loss = 0.5*self.fc02_loss + self.c_loss
                self.fc02_c_loss = self.wd_c02 + self.c_loss
                # self.fc02_c_loss = -self.dc02_loss + self.c_loss

            # self.wd = self.wd_c01 + self.wd_c02
            # self.f_loss = self.fc01_loss + self.fc02_loss
            # self.d_loss = self.dc01_loss + self.dc02_loss

        self.reg_sum = scalar_summary("reg", self.reg)

        self.c_loss_sum = scalar_summary("c_loss", self.c_loss)

        self.dis_f_sum = scalar_summary("dis_f_s_t", self.Dis_f)
        self.wd_f_sum = scalar_summary("wd_f", self.wd_f)
        self.gp_f_sum = scalar_summary("gp_f", self.gp_f)
        self.df_loss_sum = scalar_summary("df_loss", self.df_loss)
        self.ff_loss_sum = scalar_summary("ff_loss", self.ff_loss)
        self.ff_c_loss_sum = scalar_summary("ff_c_loss", self.ff_c_loss)

        self.dis_c01_sum = scalar_summary("dis_c01_s_t", self.Dis_c01)
        self.wd_c01_sum = scalar_summary("wd_c01", self.wd_c01)
        self.gp_c01_sum = scalar_summary("gp_c01", self.gp_c01)
        self.dc01_loss_sum = scalar_summary("dc01_loss", self.dc01_loss)
        self.fc01_loss_sum = scalar_summary("fc01_loss", self.fc01_loss)
        self.fc01_c_loss_sum = scalar_summary("fc01_c_loss", self.fc01_c_loss)

        self.dis_c02_sum = scalar_summary("dis_c02_s_t", self.Dis_c02)
        self.wd_c02_sum = scalar_summary("wd_c02", self.wd_c02)
        self.gp_c02_sum = scalar_summary("gp_c02", self.gp_c02)
        self.dc02_loss_sum = scalar_summary("dc02_loss", self.dc02_loss)
        self.fc02_loss_sum = scalar_summary("fc02_loss", self.fc02_loss)
        self.fc02_c_loss_sum = scalar_summary("fc02_c_loss", self.fc02_c_loss)

        # self.f_loss_sum = scalar_summary("f_loss", self.f_loss)
        # self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.c_vars = [var for var in t_vars if 'c_' in var.name]
        self.df_vars = [var for var in t_vars if 'df_' in var.name]
        self.dc01_vars = [var for var in t_vars if 'dc01_' in var.name]
        self.dc02_vars = [var for var in t_vars if 'dc02_' in var.name]
        self.fs_vars = [var for var in t_vars if 'fs_' in var.name]
        self.ft_vars = [var for var in t_vars if 'ft_' in var.name]
        self.fc_vars = [var for var in t_vars if 'fc_' in var.name]

        self.fs_sel_vars = self.fs_vars[-2:]

        print(self.c_vars)
        print(self.dc01_vars)
        print(self.dc02_vars)
        print(self.fs_vars)
        print(self.fc_vars)

        with tf.variable_scope("Accuracy"):
            # Calculate the correct predictions scource
            correct_prediction_s = tf.equal(tf.argmax(self.P_s, 1), tf.argmax(self.Y_s, 1))
            # Calculate accuracy on the test set
            self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction_s, "float"))
            # Calculate the correct predictions target
            correct_prediction_t = tf.equal(tf.argmax(self.P_t, 1), tf.argmax(self.Y_t, 1))
            # Calculate accuracy on the test set
            self.accuracy_t = tf.reduce_mean(tf.cast(correct_prediction_t, "float"))

        self.S_accuarcy_sum = scalar_summary("S_accuarcy", self.accuracy_s)
        self.T_accuarcy_sum = scalar_summary("T_accuarcy", self.accuracy_t)

        self.merge_acc_s = [self.S_accuarcy_sum]
        self.merge_acc_t = [self.T_accuarcy_sum]

        self.merge_c = [self.f_s_sum, self.f_sc_sum, self.c_s_sum, self.c_loss_sum] + self.F_s_sum + self.F_sc_sum + self.C_s_sum

        self.merge_df = [self.dis_f_sum, self.gp_f_sum, self.wd_f_sum, self.df_loss_sum] + self.Df_s_sum
        self.merge_ff = [self.df_s_sum, self.df_t_sum, self.s_distribution_f, self.t_distribution_f,
                           self.ff_loss_sum, self.ff_c_loss_sum] + self.Df_t_sum
        # self.merge_fc01_c = self.merge_fc01 + self.merge_c
        self.merge_ff_c = self.merge_ff

        self.merge_dc01 = [self.dis_c01_sum, self.gp_c01_sum, self.wd_c01_sum, self.dc01_loss_sum] + self.Dc01_s_sum
        self.merge_fc01 = [self.dc01_s_sum, self.dc01_t_sum, self.s_distribution_c01, self.t_distribution_c01,
                           self.fc01_loss_sum, self.fc01_c_loss_sum] + self.Dc01_t_sum
        # self.merge_fc01_c = self.merge_fc01 + self.merge_c
        self.merge_fc01_c = self.merge_fc01

        self.merge_dc02 = [self.dis_c02_sum, self.gp_c02_sum, self.wd_c02_sum, self.dc02_loss_sum] + self.Dc02_s_sum
        self.merge_fc02 = [self.dc02_s_sum, self.dc02_t_sum, self.s_distribution_c02, self.t_distribution_c02,
                           self.fc02_loss_sum, self.fc02_c_loss_sum] + self.Dc02_t_sum
        # self.merge_fc02_c = self.merge_fc02 + self.merge_c
        self.merge_fc02_c = self.merge_fc02

        self.saver = tf.train.Saver()

    def train(self, config):
        self.train_01(config)

    def train_01(self, config):

        tf.set_random_seed(1)  # to keep consistent results

        c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.c_loss, var_list=self.fs_vars + self.fc_vars + self.c_vars)

        df_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.df_loss, var_list=self.df_vars)
        ff_c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.ff_c_loss, var_list=self.fs_sel_vars)

        dc01_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.dc01_loss, var_list=self.dc01_vars)
        fc01_c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.fc01_c_loss, var_list=self.fs_sel_vars + self.fc_vars)

        dc02_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.dc02_loss, var_list=self.dc02_vars)
        fc02_c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.fc02_c_loss, var_list= self.fs_sel_vars + self.fc_vars + self.c_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        self.c_m_sum = merge_summary(self.merge_c)
        self.df_m_sum = merge_summary(self.merge_df)
        self.ff_c_m_sum = merge_summary(self.merge_ff_c)
        self.dc01_m_sum = merge_summary(self.merge_dc01)
        self.fc01_c_m_sum = merge_summary(self.merge_fc01_c)
        self.dc02_m_sum = merge_summary(self.merge_dc02)
        self.fc02_c_m_sum = merge_summary(self.merge_fc02_c)
        self.accuracy_s_sum = merge_summary(self.merge_acc_s)
        self.accuracy_t_sum = merge_summary(self.merge_acc_t)

        self.merge_all = tf.summary.merge_all()
        # self.writer = SummaryWriter("./log/" + self.model_name, self.sess.graph)
        self.writer = SummaryWriter(self.log_dir, self.sess.graph)

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes = self.data_s
        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t

        X_train = X_s_train
        Y_train = Y_s_train

        X_s_data = np.vstack((X_s_train, X_s_test))
        Y_s_data = np.vstack((Y_s_train, Y_s_test))
        X_t_data = np.vstack((X_t_train, X_t_test))
        Y_t_data = np.vstack((Y_t_train, Y_t_test))
        X_test = np.vstack((X_t_train, X_t_test))
        Y_test = np.vstack((Y_t_train, Y_t_test))
        print(X_test.shape)
        print(Y_test.shape)
        # lc = min(X_s_data.shape[0], X_t_data.shape[0])
        lc = 500
        X_s_data4sum, Y_s_data4sum = X_s_data[0:lc], Y_s_data[0:lc]
        X_t_data4sum, Y_t_data4sum = X_t_data[0:lc], Y_t_data[0:lc]

        self.X_s_data4sum = X_s_data4sum
        self.Y_s_data4sum = Y_s_data4sum
        self.X_t_data4sum = X_t_data4sum
        self.Y_t_data4sum = Y_t_data4sum

        del X_s_data4sum, Y_s_data4sum, X_t_data4sum, Y_t_data4sum

        start_time = time.time()
        # could_load, checkpoint_counter = self.load(self.load_dir)
        could_load, checkpoint_counter = self.load_base()
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            self.print_accuracy_s(self.data_s)
            self.print_accuracy_t(self.data_t)
        else:
            print(" [!] Load failed...")


        if not could_load:
            Epoch_C = config.epoch_pre
            counter = 0
            seed = 3  # to keep consistent results
            minibatches_s = random_mini_batches_CNN1D(X_train, Y_train, self.batch_size, seed)
            minibatches_t = random_mini_batches_CNN1D(X_t_train, Y_t_train, self.batch_size, seed)
            i_s, i_t = 0, 0
            ns = len(minibatches_s) - 2
            nt = len(minibatches_t) - 2
            # Update C network
            epoch_C_cost = []
            for epoch in range(Epoch_C):

                (minibatch_X_s, minibatch_Y_s) = minibatches_s[i_s]
                if i_s == ns:
                    i_s = 0
                else:
                    i_s += 1
                (minibatch_X_t, minibatch_Y_t) = minibatches_t[i_t]
                if i_t == nt:
                    i_t = 0
                else:
                    i_t += 1
                # print(minibatch_X_s.shape)
                # print(minibatch_X_t.shape)
                _, cost = self.sess.run([c_optim, self.c_loss],
                                        feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                                   self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
                summary_str, summary_str_acc_s, summary_str_acc_t = \
                    self.sess.run([self.c_m_sum, self.accuracy_s_sum, self.accuracy_t_sum],
                                  feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                             self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
                self.writer.add_summary(summary_str, counter)
                self.writer.add_summary(summary_str_acc_s, counter)
                self.writer.add_summary(summary_str_acc_t, counter)
                epoch_C_cost.append(cost)
                # if counter % 5== 0:
                #     self.save(config.checkpoint_dir, counter)
                if epoch % 1 == 0:
                    print("Epoch_Fs_C_net: [%2d/%2d], c_loss: %.8f" \
                          % (epoch, Epoch_C, cost))
                    # print calssifier result
                    r_s = self.print_accuracy_s(self.data_s)
                    r_t = self.print_accuracy_t(self.data_t)
                    # r_s_t = self.print_accuracy_s_t(self.data_t)
                    wd_s_t = self.cal_wasserstein_s_t(self.data_s, self.data_t)
                    r_s_t = wd_s_t
                    self.write_result(counter, r_s, r_t, r_s_t)
                    self.write_Bestresult(counter, r_s, r_t, r_s_t)
                    # self.write_Bestresult_Source(counter, r_s, r_t, r_s_t)
                    self.updataAccuracy(r_s, r_t)
                    self.updataWD(wd_s_t)

                if epoch % 100 == 0:
                    self.plotAccuracy4check()
                    self.plotWD4check()

                # if self.terminal_idx >= 200:
                #     break
                counter += 1
                # if counter % 50== 0:
                #     self.save(config.checkpoint_dir, counter)
            name = 'BestAccuracy'
            self.save_best(path=self.base_dir, name=name)

            self.initAccuracy()
            self.initWD()

        seed = 3  # to keep consistent results
        minibatches_s = random_mini_batches_CNN1D(X_train, Y_train, self.batch_size, seed)
        minibatches_t = random_mini_batches_CNN1D(X_t_train, Y_t_train, self.batch_size, seed)
        self.minibatches_s = minibatches_s
        self.minibatches_t = minibatches_t
        del minibatches_s, minibatches_t
        i_s, i_t = 0, 0
        ns = len(self.minibatches_s) - 2
        nt = len(self.minibatches_t) - 2

        global_step = tf.Variable(0,name='global_step',trainable=False)
        learning_rate = config.learning_rate
        learning_rate = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=global_step,
                                                   decay_steps=1000, decay_rate=0.9, staircase=True)

        for epoch in range(config.epoch):
            s_t = time.time()
            self.epoch = epoch
            # global_step = epoch

            (minibatch_X_s, minibatch_Y_s) = self.minibatches_s[i_s]
            if i_s == ns:
                i_s = 0
            else:
                i_s += 1
            (minibatch_X_t, minibatch_Y_t) = self.minibatches_t[i_t]
            if i_t == nt:
                i_t = 0
            else:
                i_t += 1

            sign_Classifier = False
            if sign_Classifier:
                _, cost = self.sess.run([c_optim, self.c_loss],
                                        feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                                   self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})

            # optimize discriminator_c01
            i_s, i_t = self.tuning_discriminator(config=config, n_critic=2, n_critic_init=5, epoch=epoch, d_optim=dc01_optim, d_loss=self.dc01_loss,
                                                 d_m_sum = self.dc01_m_sum, f_c_optim=fc01_c_optim, f_c_loss=self.fc01_c_loss,
                                                 f_c_m_sum=self.fc01_c_m_sum, counter=counter, ns=ns, nt=nt, i_s = i_s, i_t=i_t,
                                                 name_D="Epoch_Dc01", name_F="Epoch_Fc01")

            # optimize discriminator_c02
            i_s, i_t = self.tuning_discriminator(config=config, n_critic=2, n_critic_init=5, epoch=epoch, d_optim=dc02_optim, d_loss=self.dc02_loss,
                                                 d_m_sum = self.dc02_m_sum, f_c_optim=fc02_c_optim, f_c_loss=self.fc02_c_loss,
                                                 f_c_m_sum=self.fc02_c_m_sum, counter=counter, ns=ns, nt=nt, i_s = i_s, i_t=i_t,
                                                 name_D="Epoch_Dc02", name_F="Epoch_Fc02")


            summary_str, summary_str_acc_s, summary_str_acc_t = \
                self.sess.run([self.c_m_sum, self.accuracy_s_sum, self.accuracy_t_sum],
                                feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                            self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
            self.writer.add_summary(summary_str, counter)
            self.writer.add_summary(summary_str_acc_s, counter)
            self.writer.add_summary(summary_str_acc_t, counter)

            if self.epoch % self.epochWriteThre == 0:
                if sign_Classifier:
                    print("Epoch_Fs_C_net: [%2d/%2d], c_loss: %.8f" % (epoch, config.epoch, cost))
                # print calssifier result
                r_s = self.print_accuracy_s(self.data_s)
                r_t = self.print_accuracy_t(self.data_t)
                # r_s_t = self.print_accuracy_s_t(self.data_t)
                r_s_t = 0, 0
                wd_s_t = self.cal_wasserstein_s_t(self.data_s, self.data_t)
                r_s_t = wd_s_t
                self.write_result(counter, r_s, r_t, r_s_t)
                self.write_Bestresult_DA(counter, r_s, r_t, r_s_t)
                self.updataAccuracy(r_s, r_t)
                self.updataWD(wd_s_t)

            if epoch % 500 == 0:
                self.plotAccuracy4check(name='RecordAccuracy_DA')
                self.plotWD4check('RecordWD_DA')

            # if self.terminal_idx >= 10:
            #     break
            counter += 1
            e_t = time.time()
            print('The comsuing time is %s' %(e_t - s_t))

    def tuning_discriminator(self, config, n_critic=1, n_critic_init=25, epoch=0, d_optim=None, d_loss=None, d_m_sum=None,
                             f_c_optim=None, f_c_loss=None, f_c_m_sum=None, counter=1, ns=None, nt=None, i_s=None, i_t=None,
                             name_D="Epoch_D", name_F="Epoch_F"):

        # optimize discriminator_f
        n_critic = n_critic

        epoch = epoch
        if epoch % 500 == 0 or epoch < 25:
            n_critic = n_critic_init
        for i in range(n_critic):
            (minibatch_X_s, minibatch_Y_s) = self.minibatches_s[i_s]
            if i_s == ns:
                i_s = 0
            else:
                i_s += 1
            (minibatch_X_t, minibatch_Y_t) = self.minibatches_t[i_t]
            if i_t == nt:
                i_t = 0
            else:
                i_t += 1

            _, cost_D = self.sess.run([d_optim, d_loss],
                                      feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                                 self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
            if self.epoch % self.epochWriteThre == 0:
                print(name_D + ": [%2d/%2d]  D_net: [%2d/%2d], d_loss: %.8f" % (epoch, config.epoch, i, n_critic, cost_D))
        # summary_str = self.sess.run(d_m_sum, feed_dict={self.X_s: self.X_s_data4sum, self.Y_s: self.Y_s_data4sum,
        #                                                 self.X_t: self.X_t_data4sum, self.Y_t: self.Y_t_data4sum})
        summary_str = self.sess.run(d_m_sum, feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                                        self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
        self.writer.add_summary(summary_str, counter)

        (minibatch_X_s, minibatch_Y_s) = self.minibatches_s[i_s]
        if i_s == ns:
            i_s = 0
        else:
            i_s += 1
        (minibatch_X_t, minibatch_Y_t) = self.minibatches_t[i_t]
        if i_t == nt:
            i_t = 0
        else:
            i_t += 1
        _, cost_F = self.sess.run([f_c_optim, f_c_loss],
                                  feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                             self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
        # summary_str = self.sess.run(f_c_m_sum, feed_dict={self.X_s: self.X_s_data4sum, self.Y_s: self.Y_s_data4sum,
        #                                                   self.X_t: self.X_t_data4sum, self.Y_t: self.Y_t_data4sum})
        summary_str = self.sess.run(f_c_m_sum, feed_dict={self.X_s: minibatch_X_s, self.Y_s: minibatch_Y_s,
                                                          self.X_t: minibatch_X_t, self.Y_t: minibatch_Y_t})
        self.writer.add_summary(summary_str, counter)

        if self.epoch % self.epochWriteThre == 0:

            print(name_F + ": [%2d/%2d]  f_net: , f_loss: %.8f" % (epoch, config.epoch, cost_F))

        return  i_s, i_t

    def gradient_penalty(self, D_net, F_s, F_t):
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * F_s + (1 - epsilon) * F_t
        D_d_hat, D_logits_d_hat = D_net(x_hat, self.y, reuse=True, with_sum=False)

        gp_ddx = tf.gradients(D_logits_d_hat, x_hat)[0]
        # print(gp_ddx.get_shape().as_list())
        # gp_ddx = gp_ddx[0]
        print(gp_ddx.get_shape().as_list())
        gp_ddx = tf.sqrt(tf.reduce_sum(tf.square(gp_ddx), axis=1))
        gp_ddx = tf.reduce_mean(tf.square(gp_ddx - 1.0) * 1.0)

        return gp_ddx

    #classifier
    def feature_s(self, x, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        return self.feature_s_base(x, y, reuse=reuse, is_train=is_train, with_sum=with_sum, with_list=with_list)

    def feature_s_base02(self, x, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("feature_source") as scope:
            if reuse:
                scope.reuse_variables()
            x_in = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1])

            h0, w0, b0 = conv1d(x_in, self.ff_dim/4, k_l=5,d_l = 1, name='fs_h0_conv', with_w=True)
            # h0 = tf.nn.tanh(h0)
            h0 = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            output = max_pool1d(h0, name='fs_h0_pool')

            h1, w1, b1 = conv1d(output, self.ff_dim/2, k_l=3,d_l = 1, name='fs_h1_conv', with_w=True)
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.relu(h1)
            # h1 = lrelu(h1)
            output = max_pool1d(h1, name='fs_h1_pool')

            h2, w2, b2 = conv1d(output, self.ff_dim , k_l=3,d_l = 1, name='fs_h2_conv', with_w=True)
            h2 = tf.nn.relu(h2)
            output = max_pool1d(h2, name='fs_h2_pool')

            # h3, w3, b3 = conv1d(output, self.ff_dim * 2, name='fs_h3_conv', with_w=True)
            # h3 = tf.nn.relu(h3)
            # output = max_pool1d(h3, name='fs_h3_pool')

            h0_sum = histogram_summary("fs_h0", h0)
            # h1_sum = histogram_summary("fs_h1", h1)
            # h2_sum = histogram_summary("fs_h2", h2)

            feature_sum_s = [h0_sum]

            if with_sum:
                return output, feature_sum_s
            else:
                return output

    def feature_s_base01(self, x, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("feature_source") as scope:
            if reuse:
                scope.reuse_variables()
            x_in = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1])

            h0, w0, b0 = conv1d(x_in, output_dim=self.ff_dim/4, k_l=64,d_l = 2, name='fs_h0_conv', with_w=True)
            # h0 = tf.nn.tanh(h0)
            h0 = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            output = max_pool1d(h0, name='fs_h0_pool')

            h1, w1, b1 = conv1d(output, output_dim=self.ff_dim/2, k_l=3,d_l = 1, name='fs_h1_conv', with_w=True)
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.relu(h1)
            # h1 = lrelu(h1)
            output = max_pool1d(h1, name='fs_h1_pool')

            h2, w2, b2 = conv1d(output, output_dim=self.ff_dim, k_l=3, d_l = 1,  name='fs_h2_conv', with_w=True)
            h2 = tf.nn.relu(h2)
            output = max_pool1d(h2, name='fs_h2_pool')

            h3, w3, b3 = conv1d(output, output_dim=self.ff_dim*2, k_l=3, d_l = 1, name='fs_h3_conv', with_w=True)
            h3 = tf.nn.relu(h3)
            output = max_pool1d(h3, name='fs_h3_pool')

            h4, w4, b4 = conv1d(output, output_dim=self.ff_dim*2, k_l=3, d_l = 1, name='fs_h4_conv', with_w=True)
            h4 = tf.nn.relu(h4)
            output = max_pool1d(h4, name='fs_h4_pool')

            h0_sum = histogram_summary("fs_h0", h0)
            # h1_sum = histogram_summary("fs_h1", h1)
            # h2_sum = histogram_summary("fs_h2", h2)

            feature_sum_s = [h0_sum]

            if with_sum:
                return output, feature_sum_s
            else:
                return output

    def feature_s_base(self, x, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_source") as scope:
            if reuse:
                scope.reuse_variables()
            x_in = tf.reshape(x, [-1, x.get_shape().as_list()[1], 1])

            h0, w0, b0 = conv1d(x_in, output_dim=self.ff_dim/4, k_l=64,d_l = 2, name='fs_h0_conv', with_w=True)
            # h0 = tf.nn.tanh(h0)
            h0 = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            output = max_pool1d(h0, name='fs_h0_pool')

            h1, w1, b1 = conv1d(output, output_dim=self.ff_dim/2, k_l=32,d_l = 2, name='fs_h1_conv', with_w=True)
            # h1 = tf.nn.tanh(h1)
            h1 = tf.nn.relu(h1)
            # h1 = lrelu(h1)
            output = max_pool1d(h1, name='fs_h1_pool')

            h2, w2, b2 = conv1d(output, output_dim=self.ff_dim, k_l=16,d_l = 2,  name='fs_h2_conv', with_w=True)
            h2 = tf.nn.relu(h2)
            output = max_pool1d(h2, name='fs_h2_pool')

            h3, w3, b3 = conv1d(output, output_dim=self.ff_dim*2, k_l=8,d_l = 2, name='fs_h3_conv', with_w=True)
            h3 = tf.nn.relu(h3)
            output = max_pool1d(h3, name='fs_h3_pool')

            h4, w4, b4 = conv1d(output, output_dim=self.ff_dim*2, k_l=8,d_l = 2, name='fs_h4_conv', with_w=True)
            h4 = tf.nn.relu(h4)
            # h4 = tf.nn.tanh(h4)
            output = max_pool1d(h4, name='fs_h4_pool')

            h0_sum = histogram_summary("fs_h0", h0)
            # h1_sum = histogram_summary("fs_h1", h1)
            # h2_sum = histogram_summary("fs_h2", h2)

            feature_sum_s = [h0_sum]
            h_list = [output]

            # if with_sum:
            #     return output, feature_sum_s
            # else:
            #     return output

            if with_sum and not with_list:
                return output, feature_sum_s
            elif with_list and not with_sum:
                return output, h_list
            elif with_sum and with_list:
                return output, feature_sum_s, h_list
            else:
                return output

    def feature_c(self, x, y=None, reuse=False, is_train=True, with_sum=True ,with_list=False):

        return self.feature_c_base(x, y, reuse=reuse, is_train=is_train, with_sum=with_sum, with_list=with_list)

    def feature_c_base02(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_target") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = fully_conn(f, self.ffc_dim/8, name='fc_h0_conn')
            h0 = tf.nn.tanh(h0)
            # h0 = tf.nn.relu(h0)

            h0_sum = histogram_summary("fc_h0", h0)
            feature_sum_c = [h0_sum]
            h_list = [h0]
            output_ = h0

            if with_sum and not with_list:
                return output_, feature_sum_c
            elif with_list and not with_sum:
                return output_, h_list
            elif with_sum and with_list:
                return output_, feature_sum_c, h_list
            else:
                return output_

    def feature_c_base01(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_target") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = fully_conn(f, self.ffc_dim/4, name='fc_h0_conn')
            h0 = tf.nn.tanh(h0)
            output_ = h0
            # h0 = tf.nn.relu(h0)

            h1 = fully_conn(h0, self.ffc_dim/16, name='fc_h1_conn')
            h1 = tf.nn.tanh(h1)
            output_ = h1

            h0_sum = histogram_summary("fc_h0", h0)
            feature_sum_c = [h0_sum]
            h_list = [h0]
            output_ = h0

            if with_sum and not with_list:
                return output_, feature_sum_c
            elif with_list and not with_sum:
                return output_, h_list
            elif with_sum and with_list:
                return output_, feature_sum_c, h_list
            else:
                return output_

    def feature_c_base(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("feature_target") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = fully_conn(f, self.ffc_dim/2, name='fc_h0_conn')
            h0 = tf.nn.tanh(h0)
            # h0 = tf.nn.relu(h0)

            h0_sum = histogram_summary("fc_h0", h0)
            feature_sum_c = [h0_sum]
            h_list = [h0]
            output_ = h0

            if with_sum and not with_list:
                return output_, feature_sum_c
            elif with_list and not with_sum:
                return output_, h_list
            elif with_sum and with_list:
                return output_, feature_sum_c, h_list
            else:
                return output_

    def classifier(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list = False):

        return self.classifier_output(f, y, reuse=reuse, is_train=is_train, with_sum=with_sum,  with_list=with_list)

    def classifier_output(self, f, y=None, reuse=False, is_train=True, with_sum=True, with_list=False):

        with tf.variable_scope("classifier") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = linear(f, self.y_dim, scope='c_h0_lin')

            h0_sum = histogram_summary("c_h0", h0)
            classifier_sum = [h0_sum]
            h_list = [tf.nn.softmax(h0)]
            output = h0
            if with_sum and not with_list:
                return tf.nn.softmax(output), output, classifier_sum
            elif with_list and not with_sum:
                return tf.nn.softmax(output), output, h_list
            elif with_sum and with_list:
                return tf.nn.softmax(output), output, classifier_sum, h_list
            else:
                return tf.nn.softmax(output), output

    #discriminator

    def discriminator_f(self, f, y=None, reuse=False, with_sum=True):

        return self.discriminator_f_base(f, y, reuse, with_sum=with_sum)

    def discriminator_f_base(self, f, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("discriminator_f") as scope:
            if reuse:
                scope.reuse_variables()

            # h0, w0, b0 = conv1d(f, self.df_dim * 4, name='df_h0_conv', with_w=True)
            # # h0         = tf.nn.relu(h0)
            # h0 = lrelu(h0)
            # # h0 = max_pool1d(h0, name='d_h0_pool')
            # h1, w1, b1 = conv1d(f, self.df_dim * 2, name='df_h1_conv', with_w=True)
            # # h1         = tf.nn.relu(h1)
            # h1 = lrelu(h1)

            h2, w2, b2 = fully_conn(f, self.dfc_dim/2, name='df_h2_conn', with_w=True)
            # h2 = lrelu(h2)
            h2 = tf.nn.tanh(h2)
            h3, w3, b3 = linear(h2, 1, scope='df_h3_lin', with_w=True)

            # h0_sum = histogram_summary("d_h0", h0)
            # h1_sum = histogram_summary("df_h1", h1)
            h2_sum = histogram_summary("df_h2", h2)
            h3_sum = histogram_summary("df_h3", h3)
            discriminator_sum = [ h2_sum,  h3_sum]

            if with_sum:
                return tf.nn.sigmoid(h3), h3, discriminator_sum
            else:
                return tf.nn.sigmoid(h3), h3

    def discriminator_f_base01(self, f, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("discriminator_f") as scope:
            if reuse:
                scope.reuse_variables()

            h2, w2, b2 = fully_conn(f, self.dfc_dim*2, name='df_h2_conn', with_w=True)
            h2 = lrelu(h2)
            h3, w3, b3 = linear(h2, 1, scope='df_h3_lin', with_w=True)

            h2_sum = histogram_summary("df_h2", h2)
            h3_sum = histogram_summary("df_h3", h3)
            discriminator_sum = [h2_sum, h3_sum]

            if with_sum:
                return tf.nn.sigmoid(h3), h3, discriminator_sum
            else:
                return tf.nn.sigmoid(h3), h3

    def discriminator_c01(self, f, y=None, reuse=False, with_sum=True):

        return self.discriminator_c01_base(f, y, reuse, with_sum=with_sum)

    def discriminator_c01_base(self, f, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("discriminator_c01") as scope:
            if reuse:
                scope.reuse_variables()

            h1, w1, b1 = conv1d(f, self.df_dim * 4, name='dc01_h1_conv', with_w=True)

            h1 = lrelu(h1)

            h2, w2, b2 = fully_conn(h1, self.dfc_dim/1, name='dc01_h2_conn', with_w=True)
            h2 = lrelu(h2)
            h3, w3, b3 = linear(h2, 1, scope='dc01_h3_lin', with_w=True)


            h1_sum = histogram_summary("dc01_h1", h1)
            h2_sum = histogram_summary("dc01_h2", h2)
            h3_sum = histogram_summary("dc01_h3", h3)
            discriminator_sum = [h1_sum, h2_sum,  h3_sum]

            if with_sum:
                return tf.nn.sigmoid(h3), h3, discriminator_sum
            else:
                return tf.nn.sigmoid(h3), h3

    def discriminator_c01_base01(self, f, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("discriminator_c01") as scope:
            if reuse:
                scope.reuse_variables()

            h2, w2, b2 = fully_conn(f, self.dfc_dim*2, name='dc01_h2_conn', with_w=True)

            h2 = lrelu(h2)
            h3, w3, b3 = linear(h2, 1, scope='dc01_h3_lin', with_w=True)

            h2_sum = histogram_summary("dc01_h2", h2)
            h3_sum = histogram_summary("dc01_h3", h3)
            discriminator_sum = [h2_sum, h3_sum]

            if with_sum:
                return tf.nn.sigmoid(h3), h3, discriminator_sum
            else:
                return tf.nn.sigmoid(h3), h3

    def discriminator_c02(self, f, y=None, reuse=False, with_sum=True):

        return self.discriminator_c02_base(f, y, reuse, with_sum=with_sum)

    def discriminator_c02_base(self, f, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("discriminator_c02") as scope:
            if reuse:
                scope.reuse_variables()


            h1, w1, b1 = conv1d(f, self.df_dim * 2, name='dc02_h1_conv', with_w=True)

            h1 = lrelu(h1)

            h2, w2, b2 = fully_conn(h1, self.dfc_dim/1, name='dc02_h2_conn', with_w=True)
            h2 = lrelu(h2)
            h3, w3, b3 = linear(h2, 1, scope='dc02_h3_lin', with_w=True)

            h1_sum = histogram_summary("dc02_h1", h1)
            h2_sum = histogram_summary("dc02_h2", h2)
            h3_sum = histogram_summary("dc02_h3", h3)
            discriminator_sum = [h1_sum, h2_sum,  h3_sum]

            if with_sum:
                return tf.nn.sigmoid(h3), h3, discriminator_sum
            else:
                return tf.nn.sigmoid(h3), h3

    def discriminator_c02_base01(self, f, y=None, reuse=False, is_train=True, with_sum=True):

        with tf.variable_scope("discriminator_c02") as scope:
            if reuse:
                scope.reuse_variables()

            h2, w2, b2 = fully_conn(f, self.dfc_dim*2, name='dc02_h2_conn', with_w=True)
            # h2 = tf.nn.relu(h2)
            h2 = lrelu(h2)
            h3, w3, b3 = linear(h2, 1, scope='dc02_h3_lin', with_w=True)

            h2_sum = histogram_summary("dc02_h2", h2)
            h3_sum = histogram_summary("dc02_h3", h3)
            discriminator_sum = [h2_sum, h3_sum]

            if with_sum:
                return tf.nn.sigmoid(h3), h3, discriminator_sum
            else:
                return tf.nn.sigmoid(h3), h3

    #predictor
    def predictor_s(self, x, y=None):

        F_s, F_s_list = self.feature_s(x, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        F_sc, F_sc_list = self.feature_c(F_s, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        P, P_logits, C_list = self.classifier(F_sc, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        h_list = F_s_list + F_sc_list + C_list
        return P, P_logits, h_list

    def predictor(self, x, y=None):

        F_s = self.feature_s(x, y, reuse=True, is_train=False, with_sum=False)
        F_c, F_c_list = self.feature_c(F_s, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        P, P_logits, C_list = self.classifier(F_c, y, reuse=True, is_train=False, with_sum=False, with_list=True)
        h_list = []
        h_list.append(F_c_list[0])
        h_list.append(C_list[0])

        return P, P_logits, h_list

    def cal_distance(self, x1, x2):
        x1_in = tf.reshape(x1, [-1, x1.get_shape().as_list()[1], 1])
        x2_in = tf.reshape(x2, [-1, x2.get_shape().as_list()[1], 1])

        Dis = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x1_in - x2_in), 1)))

        return Dis

    def print_accuracy_s(self, data):
        data_s = data

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes_s = data_s

        # r =  self.accuracy_s.eval({self.X_s: X_s_train, self.Y_s: Y_s_train})
        r = self.sess.run(self.accuracy_s, feed_dict={self.X_s: X_s_train, self.Y_s: Y_s_train})
        r_c = self.get_multicalss_accuracy_s(X_s_train, Y_s_train, classes_s)
        # print(" Source Train Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Source Train Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_train = r
        r_c_train = r_c

        # r =  self.accuracy_s.eval({self.X_s: X_s_test, self.Y_s: Y_s_test})
        r = self.sess.run(self.accuracy_s, feed_dict={self.X_s: X_s_test, self.Y_s: Y_s_test})
        r_c = self.get_multicalss_accuracy_s(X_s_test, Y_s_test, classes_s)
        # print(" Source Test Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Source Test Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_test = r
        r_c_test = r_c

        X_data = np.vstack((X_s_train, X_s_test))
        Y_data = np.vstack((Y_s_train, Y_s_test))

        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_data, self.Y_t: Y_data})
        r_c = self.get_multicalss_accuracy_t(X_data, Y_data, classes_s)
        # print(" Target      Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Source      Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))

        return r_train, r_test, r, r_c_train, r_c_test, r_c

    def print_accuracy_t(self, data):

        data_t = data

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes_t = data_t

        # r =  self.accuracy_t.eval({self.X_t: X_t_train, self.Y_t: Y_t_train})
        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_t_train, self.Y_t: Y_t_train})
        r_c = self.get_multicalss_accuracy_t(X_t_train, Y_t_train, classes_t)
        print(" Target Train Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_train = r
        r_c_train = r_c

        # r =  self.accuracy_t.eval({self.X_t: X_t_test, self.Y_t: Y_t_test})
        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_t_test, self.Y_t: Y_t_test})
        r_c = self.get_multicalss_accuracy_t(X_t_test, Y_t_test, classes_t)
        # print(" Target Test Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Target Test Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))
        r_test = r
        r_c_test = r_c

        X_data = np.vstack((X_t_train, X_t_test))
        Y_data = np.vstack((Y_t_train, Y_t_test))

        r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_data, self.Y_t: Y_data})
        r_c = self.get_multicalss_accuracy_t(X_data, Y_data, classes_t)
        # print(" Target      Accuary = { %s : %s, %s, %s, %s}" % (r, r_c[0], r_c[1], r_c[2], r_c[3]))
        print(" Target      Accuary = { %s : %s, %s, %s}"
              % (r, r_c[0], r_c[1], r_c[2]))


        return r_train, r_test, r, r_c_train, r_c_test,r_c

    def get_multicalss_accuracy_s(self, X, Y, Classes):
        mc_accuracy = []
        index = np.argmax(Y, axis=1)
        # for i in yy:
        # print(i)
        # print(yy.shape)
        # print(Y_test.shape)
        test_data_X = pd.DataFrame(X, index=index)
        test_data_Y = pd.DataFrame(Y, index=index)

        for i in range(Classes):
            X_t = np.array(test_data_X.ix[i])
            Y_t = np.array(test_data_Y.ix[i])
            # print(i)
            # print(X_t.shape)
            # print(Y_t.shape)
            r = self.sess.run(self.accuracy_s, feed_dict={self.X_s: X_t, self.Y_s: Y_t})
            mc_accuracy.append(r)
            # mc_accuracy.append(self.accuracy_s.eval({self.X_s: X_t, self.Y_s: Y_t}))
            # print ("Class 0: Test Accuracy:", accuracy.eval({X_s: X_t, Y_s: Y_t, keep_prob:k_prob}))

        return mc_accuracy

    def get_multicalss_accuracy_t(self, X, Y, Classes):
        mc_accuracy = []
        index = np.argmax(Y, axis=1)
        # for i in yy:
        # print(i)
        # print(yy.shape)
        # print(Y_test.shape)
        test_data_X = pd.DataFrame(X, index=index)
        test_data_Y = pd.DataFrame(Y, index=index)

        for i in range(Classes):
            X_t = np.array(test_data_X.ix[i])
            Y_t = np.array(test_data_Y.ix[i])
            r = self.sess.run(self.accuracy_t, feed_dict={self.X_t: X_t, self.Y_t: Y_t})
            mc_accuracy.append(r)
            # mc_accuracy.append(self.accuracy_t.eval({self.X_t: X_t, self.Y_t: Y_t}))
            # print ("Class 0: Test Accuracy:", accuracy.eval({X_s: X_t, Y_s: Y_t, keep_prob:k_prob}))

        return mc_accuracy

################
    def cal_wasserstein_s_t(self, data_s, data_t):

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes_s = data_s
        X_t_train, Y_t_train, X_t_test, Y_t_test, classes_t = data_t


        wd01,wd02 = self.sess.run([self.wd_c01, self.wd_c02],
                                  feed_dict={self.X_s: X_s_train, self.Y_s: Y_s_train,
                                             self.X_t: X_t_train, self.Y_t: Y_t_train})

        return wd01, wd02

    def write_result(self, idx, r_s, r_t,r_s_t):
        path = self.save_path
        # name = self.model_name + '_Accuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t + '.txt'
        name = 'Accuracy'
        name_txt = name + '.txt'
        if idx == 0:
            with open(os.path.join(path, name_txt), 'w') as Record_f:
                Record_f.write('idx' + '\t')
                Record_f.write('Train_S' + '\t')
                Record_f.write('Test_S' + '\t')
                Record_f.write('All_S' + '\t')
                Record_f.write('Train_T' + '\t')
                Record_f.write('Test_T' + '\t')
                Record_f.write('All_T' + '\t')
                Record_f.write('Train_T_10' + '\t')
                Record_f.write('Test_T_10' + '\t')
                Record_f.write('All_T_10' + '\t')
                Record_f.write('Other1' + '\t')
                Record_f.write('Other2'+ '\n')

                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

        else:
            with open(os.path.join(path, name_txt), 'a') as Record_f:
                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

    def write_Bestresult_DA(self, idx, r_s, r_t, r_s_t):
        path = self.save_path
        # name = self.model_name + '_BestAccuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t
        name = 'BestAccuracy_DA'
        name_txt = name + '.txt'
        tempbestAcc = r_t[1]
        testAcc = r_s[1]
        trainAcc = r_s[0]
        if tempbestAcc > self.bestTestAcc and trainAcc >= self.trainThre and testAcc >= self.testThre:
            # if testAcc > self.bestTestAcc:
            self.bestTestAcc = tempbestAcc
            self.writeROCandAUC()
            self.writeReport()
            self.save_best(path=self.checkpoint_dir,name=name)

            with open(os.path.join(path, name_txt), 'w') as Record_f:
                Record_f.write('idx' + '\t')
                Record_f.write('Train_S' + '\t')
                Record_f.write('Test_S' + '\t')
                Record_f.write('All_S' + '\t')
                Record_f.write('Train_T' + '\t')
                Record_f.write('Test_T' + '\t')
                Record_f.write('All_T' + '\t')
                Record_f.write('Train_T_10' + '\t')
                Record_f.write('Test_T_10' + '\t')
                Record_f.write('All_T_10' + '\t')
                Record_f.write('Other1' + '\t')
                Record_f.write('Other2' + '\n')

                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

    def plot4check_DA(self, fpr, tpr, roc_auc, path_name):
        lw = 2
        plt.ion()
        plt.figure()
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area={0:0.2f})' ''.format(roc_auc['micro']),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr['macro'], tpr['macro'],
                 label='macro-average ROC curve (area={0:0.2f})' ''.format(roc_auc['macro']),
                 color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc='lower right')
        plt.savefig(path_name)
        # plt.show()
        plt.close()
        plt.ioff()
        # plt.pause(30)

    def writeROCandAUC_DA(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t
        name = 'ROCandAUC_DA'
        name_txt = name + '.txt'
        name_figure = name + '.jpeg'
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr['macro'], tpr['macro'], roc_auc['macro'] = self.cal_ROCandAUC_DA(sign_type='macro')
        fpr['micro'], tpr['micro'], roc_auc['micro'] = self.cal_ROCandAUC_DA(sign_type='micro')


        set_type = ['macro', 'micro']
        for s_t in set_type:
            with open(os.path.join(path, s_t + '_' + name_txt), 'w') as Record_f:
                Record_f.write(str(roc_auc[s_t]) + '\t')
                Record_f.write('fpr' + '\t')
                Record_f.write('tpr' + '\n')
                m = fpr[s_t].shape[0]
                for i in range(m):
                    Record_f.write(str(i) + '\t')
                    Record_f.write(str(fpr[s_t][i]) + '\t')
                    Record_f.write(str(tpr[s_t][i]) + '\n')
        self.plot4check_DA(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                           path_name=os.path.join(path, name_figure))

    def cal_ROCandAUC_DA(self, sign_type='micro'):

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t
        # y_score = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        y_score = self.sess.run(self.P_logits_t, feed_dict={self.X_t: X_t_test})

        if sign_type is 'micro':
            fpr, tpr, thresholds = roc_curve(Y_t_test.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        elif sign_type is 'macro':
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classes):
                fpr[i], tpr[i], _ = roc_curve(Y_t_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= classes
            fpr = all_fpr
            tpr = mean_tpr
            roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def writeReport(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t + '_Report.txt'
        name = 'ROCandAUC_Report_DA'
        name_txt = name + '.txt'

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t
        y_val_pred = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        report = classification_report(y_true=np.argmax(Y_t_test, axis=1), y_pred=np.argmax(y_val_pred, axis=1),
                                       digits=4)

        with open(os.path.join(path, name_txt), 'w') as Record_f:
            for r in report:
                Record_f.write(r)

    def write_Bestresult_Source(self, idx, r_s, r_t, r_s_t):
        path = self.save_path
        # name = self.model_name + '_BestAccuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t
        name = 'BestAccuracy_Source'
        name_txt = name + '.txt'
        testAcc = r_s[2]
        trainAcc = r_s[0]
        if testAcc > self.bestTestAcc_source and trainAcc >= self.trainThre:
            self.bestTestAcc_source = testAcc
            self.writeROCandAUC_Source()
            self.writeReport_Source()
            # self.save_best(name=name)

            with open(os.path.join(path, name_txt), 'w') as Record_f:
                Record_f.write('idx' + '\t')
                Record_f.write('Train_S' + '\t')
                Record_f.write('Test_S' + '\t')
                Record_f.write('All_S' + '\t')
                Record_f.write('Train_T' + '\t')
                Record_f.write('Test_T' + '\t')
                Record_f.write('All_T' + '\t')
                Record_f.write('Train_T_10' + '\t')
                Record_f.write('Test_T_10' + '\t')
                Record_f.write('All_T_10' + '\t')
                Record_f.write('Other1' + '\t')
                Record_f.write('Other2' + '\n')

                Record_f.write(str(idx) + '\t')
                Record_f.write(str(r_s[0]) + '\t')
                Record_f.write(str(r_s[1]) + '\t')
                Record_f.write(str(r_s[2]) + '\t')
                Record_f.write(str(r_t[0]) + '\t')
                Record_f.write(str(r_t[1]) + '\t')
                Record_f.write(str(r_t[2]) + '\t')
                Record_f.write(str(r_t[3]) + '\t')
                Record_f.write(str(r_t[4]) + '\t')
                Record_f.write(str(r_t[5]) + '\t')
                Record_f.write(str(r_s_t[0]) + '\t')
                Record_f.write(str(r_s_t[1]) + '\n')

    def writeROCandAUC_Source(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t
        name = 'ROCandAUC_Source'
        name_txt = name + '.txt'
        name_figure = name + '.jpeg'
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr['macro'], tpr['macro'], roc_auc['macro'] = self.cal_ROCandAUC_Source(sign_type='macro')
        fpr['micro'], tpr['micro'], roc_auc['micro'] = self.cal_ROCandAUC_Source(sign_type='micro')

        # print(fpr['macro'].shape)
        # print(tpr['macro'].shape)
        # print(fpr['micro'].shape)
        # print(tpr['micro'].shape)

        set_type = ['macro', 'micro']
        for s_t in set_type:
            with open(os.path.join(path, s_t + '_' + name_txt), 'w') as Record_f:
                Record_f.write(str(roc_auc[s_t]) + '\t')
                Record_f.write('fpr' + '\t')
                Record_f.write('tpr' + '\n')
                m = fpr[s_t].shape[0]
                for i in range(m):
                    Record_f.write(str(i) + '\t')
                    Record_f.write(str(fpr[s_t][i]) + '\t')
                    Record_f.write(str(tpr[s_t][i]) + '\n')
        self.plot4check(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                        path_name=os.path.join(path, name_figure))

    def cal_ROCandAUC_Source(self, sign_type='micro'):

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes = self.data_s
        # y_score = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        y_score = self.sess.run(self.P_logits_s, feed_dict={self.X_s: X_s_test})

        if sign_type is 'micro':
            fpr, tpr, thresholds = roc_curve(Y_s_test.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        elif sign_type is 'macro':
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classes):
                fpr[i], tpr[i], _ = roc_curve(Y_s_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= classes
            fpr = all_fpr
            tpr = mean_tpr
            roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def writeReport_Source(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t + '_Report.txt'
        name = 'ROCandAUC_Report_Source'
        name_txt = name + '.txt'

        X_s_train, Y_s_train, X_s_test, Y_s_test, classes = self.data_s
        y_val_pred = self.sess.run(self.P_s, feed_dict={self.X_s: X_s_test})
        report = classification_report(y_true=np.argmax(Y_s_test, axis=1), y_pred=np.argmax(y_val_pred, axis=1),
                                       digits=4)

        with open(os.path.join(path, name_txt), 'w') as Record_f:
            for r in report:
                Record_f.write(r)

        # analysis 4 target

    def write_Bestresult(self, idx, r_s, r_t, r_s_t):
        path = self.save_path
        # name = self.model_name + '_BestAccuracy_'+ self.dataset_name_s + '_' + self.dataset_name_t
        name = 'BestAccuracy'
        name_txt = name + '.txt'
        tempbestAcc = r_t[1]
        trainAcc = r_s[0]
        testAcc = r_s[1]
        if tempbestAcc > self.bestTestAcc and trainAcc >= self.trainThre_pre and testAcc >= self.testThre_pre:
            self.bestTestAcc = tempbestAcc
            self.writeROCandAUC()
            self.writeReport()
            self.save_best(path=self.base_dir, name=name)

            # if os.path.exists(os.path.join(path,name_txt)):
            if self.bestAccuracy_sign:
                with open(os.path.join(path, name_txt), 'a') as Record_f:
                    Record_f.write('idx' + '\t')
                    Record_f.write('Train_S' + '\t')
                    Record_f.write('Test_S' + '\t')
                    Record_f.write('All_S' + '\t')
                    Record_f.write('Train_T' + '\t')
                    Record_f.write('Test_T' + '\t')
                    Record_f.write('All_T' + '\t')
                    Record_f.write('Train_T_10' + '\t')
                    Record_f.write('Test_T_10' + '\t')
                    Record_f.write('All_T_10' + '\t')
                    Record_f.write('Other1' + '\t')
                    Record_f.write('Other2' + '\n')

                    Record_f.write(str(idx) + '\t')
                    Record_f.write(str(r_s[0]) + '\t')
                    Record_f.write(str(r_s[1]) + '\t')
                    Record_f.write(str(r_s[2]) + '\t')
                    Record_f.write(str(r_t[0]) + '\t')
                    Record_f.write(str(r_t[1]) + '\t')
                    Record_f.write(str(r_t[2]) + '\t')
                    Record_f.write(str(r_t[3]) + '\t')
                    Record_f.write(str(r_t[4]) + '\t')
                    Record_f.write(str(r_t[5]) + '\t')
                    Record_f.write(str(r_s_t[0]) + '\t')
                    Record_f.write(str(r_s_t[1]) + '\n')
            else:
                with open(os.path.join(path, name_txt), 'w') as Record_f:
                    self.bestAccuracy_sign = True
                    Record_f.write('idx' + '\t')
                    Record_f.write('Train_S' + '\t')
                    Record_f.write('Test_S' + '\t')
                    Record_f.write('All_S' + '\t')
                    Record_f.write('Train_T' + '\t')
                    Record_f.write('Test_T' + '\t')
                    Record_f.write('All_T' + '\t')
                    Record_f.write('Train_T_10' + '\t')
                    Record_f.write('Test_T_10' + '\t')
                    Record_f.write('All_T_10' + '\t')
                    Record_f.write('Other1' + '\t')
                    Record_f.write('Other2' + '\n')

                    Record_f.write(str(idx) + '\t')
                    Record_f.write(str(r_s[0]) + '\t')
                    Record_f.write(str(r_s[1]) + '\t')
                    Record_f.write(str(r_s[2]) + '\t')
                    Record_f.write(str(r_t[0]) + '\t')
                    Record_f.write(str(r_t[1]) + '\t')
                    Record_f.write(str(r_t[2]) + '\t')
                    Record_f.write(str(r_t[3]) + '\t')
                    Record_f.write(str(r_t[4]) + '\t')
                    Record_f.write(str(r_t[5]) + '\t')
                    Record_f.write(str(r_s_t[0]) + '\t')
                    Record_f.write(str(r_s_t[1]) + '\n')

            # with open(os.path.join(path, name_txt), 'w') as Record_f:
            #     Record_f.write('idx' + '\t')
            #     Record_f.write('Train_S' + '\t')
            #     Record_f.write('Test_S' + '\t')
            #     Record_f.write('All_S' + '\t')
            #     Record_f.write('Train_T' + '\t')
            #     Record_f.write('Test_T' + '\t')
            #     Record_f.write('All_T' + '\t')
            #     Record_f.write('Train_T_10' + '\t')
            #     Record_f.write('Test_T_10' + '\t')
            #     Record_f.write('All_T_10' + '\t')
            #     Record_f.write('Other1' + '\t')
            #     Record_f.write('Other2'+ '\n')
            #
            #     Record_f.write(str(idx) + '\t')
            #     Record_f.write(str(r_s[0]) + '\t')
            #     Record_f.write(str(r_s[1]) + '\t')
            #     Record_f.write(str(r_s[2]) + '\t')
            #     Record_f.write(str(r_t[0]) + '\t')
            #     Record_f.write(str(r_t[1]) + '\t')
            #     Record_f.write(str(r_t[2]) + '\t')
            #     Record_f.write(str(r_t[3]) + '\t')
            #     Record_f.write(str(r_t[4]) + '\t')
            #     Record_f.write(str(r_t[5]) + '\t')
            #     Record_f.write(str(r_s_t[0]) + '\t')
            #     Record_f.write(str(r_s_t[1]) + '\n')

    def writeROCandAUC(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t
        name = 'ROCandAUC'
        name_txt = name + '.txt'
        name_figure = name + '.jpeg'
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr['macro'], tpr['macro'], roc_auc['macro'] = self.cal_ROCandAUC(sign_type='macro')
        fpr['micro'], tpr['micro'], roc_auc['micro'] = self.cal_ROCandAUC(sign_type='micro')

        # print(fpr['macro'].shape)
        # print(tpr['macro'].shape)
        # print(fpr['micro'].shape)
        # print(tpr['micro'].shape)

        set_type = ['macro', 'micro']
        for s_t in set_type:
            with open(os.path.join(path, s_t + '_' + name_txt), 'w') as Record_f:
                Record_f.write(str(roc_auc[s_t]) + '\t')
                Record_f.write('fpr' + '\t')
                Record_f.write('tpr' + '\n')
                m = fpr[s_t].shape[0]
                for i in range(m):
                    Record_f.write(str(i) + '\t')
                    Record_f.write(str(fpr[s_t][i]) + '\t')
                    Record_f.write(str(tpr[s_t][i]) + '\n')
        self.plot4check(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                        path_name=os.path.join(path, name_figure))

    def cal_ROCandAUC(self, sign_type='micro'):

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t
        # y_score = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        y_score = self.sess.run(self.P_logits_t, feed_dict={self.X_t: X_t_test})

        if sign_type is 'micro':
            fpr, tpr, thresholds = roc_curve(Y_t_test.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        elif sign_type is 'macro':
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classes):
                fpr[i], tpr[i], _ = roc_curve(Y_t_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= classes
            fpr = all_fpr
            tpr = mean_tpr
            roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def writeReport(self):
        path = self.save_path
        # name = self.model_name + '_ROCandAUC_' + self.dataset_name_s + '_' + self.dataset_name_t + '_Report.txt'
        name = 'ROCandAUC_Report'
        name_txt = name + '.txt'

        X_t_train, Y_t_train, X_t_test, Y_t_test, classes = self.data_t
        y_val_pred = self.sess.run(self.P_t, feed_dict={self.X_t: X_t_test})
        report = classification_report(y_true=np.argmax(Y_t_test, axis=1), y_pred=np.argmax(y_val_pred, axis=1),
                                       digits=4)

        with open(os.path.join(path, name_txt), 'w') as Record_f:
            for r in report:
                Record_f.write(r)

    def plot4check(self, fpr, tpr, roc_auc, path_name):
        lw = 2
        plt.ion()
        plt.figure()
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area={0:0.2f})' ''.format(roc_auc['micro']),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr['macro'], tpr['macro'],
                 label='macro-average ROC curve (area={0:0.2f})' ''.format(roc_auc['macro']),
                 color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc='lower right')
        plt.savefig(path_name)
        # plt.show()
        plt.close()
        plt.ioff()
        # plt.pause(30)

    def initAccuracy(self):
        self.Record_Accuracy = {'S_train': [0], 'S_test': [0], 'S_all': [0],
                                'T_train': [0], 'T_test': [0], 'T_all': [0]}

    def updataAccuracy(self, r_s, r_t):
        S_train = self.Record_Accuracy['S_train']
        S_test = self.Record_Accuracy['S_test']
        S_all = self.Record_Accuracy['S_all']
        T_train = self.Record_Accuracy['T_train']
        T_test = self.Record_Accuracy['T_test']
        T_all = self.Record_Accuracy['T_all']

        n = len(S_train)
        mean_S_all = np.mean(S_all[-100:])
        thre = 0.995
        if n >= 100:
            if mean_S_all > thre:
                self.terminal_idx += 1

        S_train.append(r_s[0])
        S_test.append(r_s[1])
        S_all.append(r_s[2])
        T_train.append(r_t[0])
        T_test.append(r_t[1])
        T_all.append(r_t[2])

        self.Record_Accuracy['S_train'] = S_train
        self.Record_Accuracy['S_test'] = S_test
        self.Record_Accuracy['S_all'] = S_all
        self.Record_Accuracy['T_train'] = T_train
        self.Record_Accuracy['T_test'] = T_test
        self.Record_Accuracy['T_all'] = T_all

    def plotAccuracy4check(self, name = 'RecordAccuracy'):

        path = self.save_path
        # name_figure = 'RecordAccuracy' + '.jpeg'
        name_figure = name + '.jpeg'
        path_name = os.path.join(path, name_figure)
        keys = list(self.Record_Accuracy.keys())


        plt.ion()

        figure, ax = plt.subplots()
        for key in keys:
            plt.plot(self.Record_Accuracy[key],
                     label=key,
                     linestyle='-', linewidth=1.5)

        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 12, }
        # 设置横纵坐标的名称以及对应字体格式
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        plt.ylim([0.0, 1.05])

        # 设置坐标刻度值的大小以及刻度值的字体
        plt.tick_params(labelsize=14)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        # print labels
        [label.set_fontname('Times New Roman') for label in labels]

        plt.xlabel('Accuracy')
        plt.ylabel('Epoch')
        # plt.legend(loc='lower right', prop=font1)
        plt.legend(loc='best', prop=font1)
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        # plt.legend(loc='lower right')
        plt.savefig(path_name)
        # plt.show()
        plt.close()
        plt.ioff()

    def initWD(self):
        self.Record_WD = {'wd01': [0], 'wd02': [0]}

    def updataWD(self,wd_s_t):

        wd01 = self.Record_WD['wd01']
        wd02 = self.Record_WD['wd02']

        wd01.append(wd_s_t[0])
        wd02.append(wd_s_t[1])

        self.Record_WD['wd01'] = wd01
        self.Record_WD['wd02'] = wd02

    def plotWD4check(self, name = 'RecordWD'):

        path = self.save_path
        # name_figure = 'RecordWD' + '.jpeg'
        name_figure = name + '.jpeg'
        path_name = os.path.join(path, name_figure)
        keys = list(self.Record_WD.keys())

        plt.ion()
        figure, axes = plt.subplots(2,1)
        i = 0
        colors = ['b','g']
        for key in keys:
            ax = axes[i]
            c = colors[i]
            ax.plot(self.Record_WD[key],
                     label=key,color=c,
                     linestyle='-', linewidth=1.5)
            # plt.plot(self.Record_WD[key],
            #          label=key,
            #          linestyle='-', linewidth=1.5)
            # 设置坐标刻度值的大小以及刻度值的字体
            # plt.tick_params(labelsize=10)
            ax.tick_params(labelsize=10)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            # print labels
            [label.set_fontname('Times New Roman') for label in labels]
            font1 = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 12, }
            # 设置横纵坐标的名称以及对应字体格式
            font2 = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 14,
                     }
            if i == 0:
                ax.set_title('Epoch')
            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('WD')
            ax.legend(loc='best', prop=font1)

            i += 1

        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        # plt.legend(loc='lower right')
        plt.savefig(path_name)
        # plt.show()
        plt.close()
        plt.ioff()

    def save_best(self, path, name, step=100):

        # checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name)
        #
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(path, name),
                        global_step=step)

    def load_base(self):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = self.base_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            try:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            except:
                print(" [*] Failed to find a checkpoint")
                return False, 0
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name_s,
            self.dataset_name_t[18:], self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = self.model_name
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualization(self, f_representation_s, y_s, f_representation_t, y_t, step):

        # Visualization of trained flatten layer (T-SNE)
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        tsne = TSNE(perplexity=30, n_components=2, n_iter=5000)
        plot_only = 100
        f_r = np.vstack([f_representation_s[:plot_only, :],f_representation_t[:plot_only, :]])
        # y = np.hstack([y_s[:plot_only],y_t[:plot_only]])
        low_dim_embs = tsne.fit_transform(f_r)

        low_dim_embs_s = low_dim_embs[0:plot_only,:]
        low_dim_embs_t = low_dim_embs[plot_only:2*plot_only,:]
        labels_s = np.argmax(y_s, axis=1)[:plot_only]
        labels_t = np.argmax(y_t, axis=1)[:plot_only]

        # low_dim_embs_s = tsne.fit_transform(f_representation_s[:plot_only, :])
        # labels_s = np.argmax(y_s, axis=1)[:plot_only]
        # low_dim_embs_t = tsne.fit_transform(f_representation_t[:plot_only, :])
        # labels_t = np.argmax(y_t, axis=1)[:plot_only]
        # plt.cla()
        fig_s = plt.figure(num='fig_s')
        plt.ion()
        fig_t = plt.figure(num='fig_t')
        plt.ion()
        self.plot_with_labels_st(fig_s=fig_s,lowDWeights_s=low_dim_embs_s,labels_s=labels_s,fig_t=fig_t,lowDWeights_t=low_dim_embs_t,labels_t=labels_t)
        # self.plot_with_labels(low_dim_embs_s, labels_s,ifsource=True)
        # self.plot_with_labels(low_dim_embs_t, labels_t,ifsource=False)
        path = 'visualizations'
        path = os.path.join(path, self.model_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        # plt.savefig(path +'/'+ self.model_name+'-' + str(step))
        fig_s.savefig(path + '/' + self.model_name + '-' + str(step) + '-Source')
        fig_t.savefig(path + '/' + self.model_name + '-' + str(step) + '-Target')
        plt.figure(num='fig_s')
        plt.ioff()
        plt.close()
        plt.figure(num='fig_t')
        plt.ioff()
        plt.close()

        # plt.show()

    def plot_with_labels_st(self,fig_s,lowDWeights_s, labels_s, fig_t, lowDWeights_t, labels_t):
        # plt.cla()
        X_s, Y_s = lowDWeights_s[:, 0], lowDWeights_s[:, 1]
        X_t, Y_t = lowDWeights_t[:, 0], lowDWeights_t[:, 1]
        x_min = min(X_s.min(),X_t.min())
        x_max = max(X_s.max(),X_t.max())
        y_min = min(Y_s.min(),Y_t.min())
        y_max = max(Y_s.max(),Y_t.max())
        for x_s, y_s, s_s in zip(X_s, Y_s, labels_s):
            c_s = cm.rainbow(int(255 * s_s / self.y_dim))
            plt.figure(num='fig_s')
            # plt.text(x_s, y_s, s_s, backgroundcolor=c_s, fontsize=9)
            plt.text(x_s, y_s, s_s, fontsize=9, bbox=dict(boxstyle='square', fc=c_s))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title('Visualize Source Full_conn layer')
            plt.show()
            plt.pause(0.000001)

        for x_s, y_s, s_s, x_t, y_t, s_t in zip(X_s, Y_s, labels_s, X_t, Y_t, labels_t):
            c_t = cm.rainbow(int(255 * s_t / self.y_dim))
            plt.figure(num='fig_t')
            plt.text(x_t, y_t, s_t, fontsize=9, bbox=dict(boxstyle='circle', fc=c_t))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title('Visualize Target Full_conn layer')
            plt.show()
            plt.pause(0.000001)

        # for x_s, y_s, s_s, x_t, y_t, s_t in zip(X_s, Y_s, labels_s, X_t, Y_t, labels_t):
        #     c_s = cm.rainbow(int(255 * s_s / self.y_dim))
        #     c_t = cm.rainbow(int(255 * s_t / self.y_dim))
        #     plt.figure(num='fig_s')
        #     # plt.text(x_s, y_s, s_s, backgroundcolor=c_s, fontsize=9)
        #     plt.text(x_s, y_s, s_s, fontsize=9, bbox=dict(boxstyle='square', fc=c_s))
        #     plt.xlim(x_min, x_max)
        #     plt.ylim(y_min, y_max)
        #     plt.title('Visualize Source Full_conn layer')
        #     plt.show()
        #     plt.pause(0.000001)
        #     plt.figure(num='fig_t')
        #     plt.text(x_t, y_t, s_t, fontsize=9, bbox=dict(boxstyle='circle', fc=c_t))
        #     plt.xlim(x_min, x_max)
        #     plt.ylim(y_min, y_max)
        #     plt.title('Visualize Target Full_conn layer')
        #     plt.show()
        #     plt.pause(0.000001)

    def plot_with_labels(self,lowDWeights, labels, ifsource = True):
        # plt.cla()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]

        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 * s / 4))
            if ifsource:
                plt.text(x, y, s, backgroundcolor=c, fontsize=9)
            else:
                plt.text(x, y, s, fontsize=9, bbox=dict(boxstyle='circle', fc=c))
            # plt.xlim(X.min(), X.max())
            # plt.ylim(Y.min(), Y.max())
            plt.title('Visualize last layer')
            plt.show()
            plt.pause(0.0001)



