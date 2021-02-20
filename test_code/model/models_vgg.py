# -*- coding: utf-8 -*-
# File: vgg_model.py

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import *
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected)

from tensorpack.tfutils.tower import get_current_tower_context
# from utils import image_summaries, att_summaries
def convnormrelu(x, name, chan, kernel_size=3, padding='SAME'):
    x = Conv2D(name, x, chan, kernel_size=kernel_size, padding=padding,data_format='channels_last')
    x = tf.nn.relu(x, name=name + '_relu')

    return x
@auto_reuse_variable_scope
def vgg_gap(image,classnum):
    ctx = get_current_tower_context()
    is_training = ctx.is_training
    end_points ={}

    with argscope(Conv2D,
        kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
            argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling],
                data_format='channels_last'):
#  'conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'add'
        l = convnormrelu(image, 'conv1_1', 64)
        end_points['conv1_1'] = l
        l = convnormrelu(l, 'conv1_2', 64)
        end_points['conv1_2'] = l
        l = MaxPooling('pool1', l, 2,data_format='channels_last')
        end_points['pool1'] = l
        
        l = convnormrelu(l, 'conv2_1', 128)
        end_points['conv2_1'] = l
        l = convnormrelu(l, 'conv2_2', 128)
        end_points['conv2_2'] = l
        l = MaxPooling('pool2', l, 2,data_format='channels_last')
        end_points['pool2'] = l

        l = convnormrelu(l, 'conv3_1', 256)
        end_points['conv3_1'] = l
        l = convnormrelu(l, 'conv3_2', 256)
        end_points['conv3_2'] = l
        l = convnormrelu(l, 'conv3_3', 256)
        end_points['conv3_3'] = l
        l = MaxPooling('pool3', l, 2,data_format='channels_last')
        end_points['pool3'] = l

        l = convnormrelu(l, 'conv4_1', 512)
        end_points['conv4_1'] = l
        l = convnormrelu(l, 'conv4_2', 512)
        end_points['conv4_2'] = l
        l = convnormrelu(l, 'conv4_3', 512)
        end_points['conv4_3'] = l
        l = MaxPooling('pool4', l, 2,data_format='channels_last')
        end_points['pool4'] = l
        
        l = convnormrelu(l, 'conv5_1', 512)
        end_points['conv5_1'] = l
        l = convnormrelu(l, 'conv5_2', 512)
        end_points['conv5_2'] = l
        l = convnormrelu(l, 'conv5_3', 512)
        end_points['conv5_3'] = l

        convmaps = convnormrelu(l, 'add', 1024, kernel_size=3)
        end_points['add'] = convmaps
        
        pre_logits = GlobalAvgPooling('gap', convmaps, data_format='channels_last')
        
        FC_name = {200:'linear_cub', 1000: 'linear'}
        logits = FullyConnected(FC_name[classnum],
            pre_logits, 1000,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        return logits, end_points
