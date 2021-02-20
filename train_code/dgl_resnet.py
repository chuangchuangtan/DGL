#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dgl_resnet.py

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
from models_resnet import *


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
        image = tf.transpose(image, [0, 3, 1, 2])
        label_onehot = tf.one_hot(label,args.classnum)
        image_summaries('input-images', image)
        logits = resnet(image, DEPTH, args)

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
            [('conv0.*/W', 1.), 
             ('conv0.*/gamma', 1.), 
             ('conv0.*/beta', 2.), 
             ('group[0-2].*/W', 1.), 
             ('group[0-2].*/gamma', 1.), 
             ('group[0-2].*/beta', 2.),
             ('linear/W', 1.),('linear/b', 2.),])]
            return optimizer.apply_grad_processors(opt, gradprocs)
        else:
            return opt
        # return opt


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
        if args.mode == 'resnet': args.load = './pretrain_model/ImageNet-ResNet50.npz'
        elif args.mode == 'se': args.load = './pretrain_model/ImageNet-ResNet50-SE.npz'
        config.session_init = get_model_loader(args.load)

    launch_train_with_config(config,
        SyncMultiGPUTrainerParameterServer(nr_gpu))
