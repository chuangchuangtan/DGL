#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dgl_inceptionv3.py

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing


import tensorflow as tf
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import viz

from utils import *
from utils_args import *
from inception_v3_c2_followspg import *

class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.uint8, [None,
                    args.final_size, args.final_size, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label'),
                tf.placeholder(tf.float32, [None, 2, 2], 'bbox')]

    def build_graph(self, image, label, bbox):
        ctx = get_current_tower_context()
        is_training = ctx.is_training


        image = image_preprocess(image, args, bgr=True)
        label_onehot = tf.one_hot(label,args.classnum)
        image_summaries('input-images', image)

        with slim.arg_scope(inception_v3_arg_scope(weight_decay=0.0005)):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm,normalizer_params={'is_training': is_training}):
                logits, end_points = inception_v3(image, 1000,is_training=is_training,global_pool=True)#
        _, indices = tf.nn.top_k(logits, 5)
        indices = tf.identity(indices, name='top5')


        loss = compute_loss_and_error(logits, label)

        add_moving_summary(loss)
        return tf.identity(loss, name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate',
                        initializer=args.base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        add_moving_summary(lr)
        if args.load:
            gradprocs = [gradproc.ScaleGradient(
                [('InceptionV3/Conv2d_1a_3x3.*/weights', 1.),
                ('InceptionV3/Conv2d_1a_3x3.*/beta', 2.),
                ('InceptionV3/Conv2d_2a_3x3.*/weights', 1.),
                ('InceptionV3/Conv2d_2a_3x3.*/beta',2.),
                ('InceptionV3/Conv2d_2b_3x3.*/weights', 1.),
                ('InceptionV3/Conv2d_2b_3x3.*/beta', 2.),
                ('InceptionV3/Conv2d_3b_3x3.*/weights', 1.),
                ('InceptionV3/Conv2d_3b_3x3.*/beta', 2.),
                ('InceptionV3/Conv2d_4a_3x3.*/weights', 1.),
                ('InceptionV3/Conv2d_4a_3x3.*/beta', 2.),
                ('InceptionV3/Mixed_5b.*/weights', 1.),
                ('InceptionV3/Mixed_5b.*/beta', 2.),
                ('InceptionV3/Mixed_5c.*/weights', 1.),
                ('InceptionV3/Mixed_5c.*/beta', 2.),
                ('InceptionV3/Mixed_5d.*/weights', 1.),
                ('InceptionV3/Mixed_5d.*/beta', 2.),
                ('InceptionV3/Mixed_6a.*/weights', 1.),
                ('InceptionV3/Mixed_6a.*/beta', 2.),
                ('InceptionV3/Mixed_6b.*/weights', 1.),
                ('InceptionV3/Mixed_6b.*/beta', 2.),
                ('InceptionV3/Mixed_6c.*/weights', 1.),
                ('InceptionV3/Mixed_6c.*/beta', 2.),
                ('InceptionV3/Mixed_6d.*/weights', 1.),
                ('InceptionV3/Mixed_6d.*/beta', 2.),
                ('InceptionV3/Mixed_6e.*/weights', 1.),
                ('InceptionV3/Mixed_6e.*/beta', 2.),
                ('InceptionV3/Logits/add2.*/weights', 10.),
                ('InceptionV3/Logits/add2.*/biases', 20.),
                ('InceptionV3/Logits/add1.*/weights', 10.),
                ('InceptionV3/Logits/add1.*/biases', 20.),
                ('InceptionV3/Logits/Conv2d_1c_1x1_1000.*/weights', 10.),
                ('InceptionV3/Logits/Conv2d_1c_1x1_1000.*/biases', 20.)])]
            return optimizer.apply_grad_processors(opt, gradprocs)
        else:
            return opt


if __name__ == '__main__':
    args = get_args()

    nr_gpu = get_nr_gpu()
    TOTAL_BATCH_SIZE = int(args.batch)
    BATCH_SIZE = TOTAL_BATCH_SIZE // nr_gpu
    args.batch = BATCH_SIZE # batch per gpu

    model = Model()
    DEPTH = args.depth

    log_dir = '/min-val-error-top1.index'


    logdir = './train_log/' + args.logdir

    logger.set_logger_dir(logdir)
    config = get_config(model, args)

    if args.load:
        args.load = './pretrain_model/inception_v3.ckpt'
        config.session_init = get_model_loader(args.load)

    launch_train_with_config(config,
        SyncMultiGPUTrainerParameterServer(nr_gpu))
