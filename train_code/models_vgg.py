# -*- coding: utf-8 -*-
# File: model_vgg.py

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import *
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected)

from tensorpack.tfutils.tower import get_current_tower_context
from utils import image_summaries


def convnormrelu(x, name, chan, option, kernel_size=3,strides=(1, 1), padding='SAME'):
    x = Conv2D(name, x, chan, kernel_size=kernel_size,strides=strides, padding=padding)
    x = tf.nn.relu(x, name=name + '_relu')
    return x

@auto_reuse_variable_scope
def vgg_gap(image, option, importance=False):
    ctx = get_current_tower_context()
    is_training = ctx.is_training
    with argscope(Conv2D,
        kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
            argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling],
                data_format='channels_first'):

        l = convnormrelu(image, 'conv1_1', 64, option)
        l = convnormrelu(l, 'conv1_2', 64, option)
        l = MaxPooling('pool1', l, 2)

        l = convnormrelu(l, 'conv2_1', 128, option)
        l = convnormrelu(l, 'conv2_2', 128, option)
        l = MaxPooling('pool2', l, 2)

        l = convnormrelu(l, 'conv3_1', 256, option)
        l = convnormrelu(l, 'conv3_2', 256, option)
        l = convnormrelu(l, 'conv3_3', 256, option)
        l = MaxPooling('pool3', l, 2)

        l = convnormrelu(l, 'conv4_1', 512, option)
        l = convnormrelu(l, 'conv4_2', 512, option)
        l = convnormrelu(l, 'conv4_3', 512, option)
        l = MaxPooling('pool4', l, 2)

        l = convnormrelu(l, 'conv5_1', 512, option)
        l = convnormrelu(l, 'conv5_2', 512, option)
        l = convnormrelu(l, 'conv5_3', 512, option)

        convmaps = convnormrelu(l, 'add', 1024,option, kernel_size=3)

        pre_logits = GlobalAvgPooling('gap', convmaps)

        logits = FullyConnected('linear',
            pre_logits, option.classnum,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        return logits, convmaps
