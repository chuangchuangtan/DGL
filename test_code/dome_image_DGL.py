#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dome_image_GRCAM

import argparse
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorpack.models import *
from tensorpack.tfutils import argscope
from tensorpack.tfutils.tower import PredictTowerContext

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
from imagenet_utils import ImageNetModel
from model import *

tf.logging.set_verbosity(tf.logging.ERROR) # or any {DEBUG, INFO, WARN, ERROR, FATAL}


def get_layers_name(bachbone='inceptionv3'):
    return {
    'inceptionv3':['add1', 'add2', 'Mixed_6e', 'Mixed_6d', 'Mixed_6c', 'Mixed_6b','Mixed_6e_branch_0' ,'Mixed_6e_branch_1' ,'Mixed_6e_branch_2','Mixed_6e_branch_3', 'Mixed_6a', 'Mixed_5d', 'Mixed_5c', 'Mixed_5b', 'MaxPool_5a_3x3', 'Conv2d_4a_3x3', 'Conv2d_3b_1x1', 'MaxPool_3a_3x3', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3'],
    
    'resnet50':['group3/block2_shortcut', 'group3/block1_shortcut', 'group3/block0_shortcut', 'group2/block5_shortcut', 'group2/block4_shortcut', 'group2/block3_shortcut', 'group2/block2_shortcut', 'group2/block1_shortcut', 'group2/block0_shortcut', 'group1/block3_shortcut', 'group1/block2_shortcut', 'group1/block1_shortcut'],
    
    'resnet50se':['group3/block2_shortcut', 'group3/block1_shortcut', 'group3/block0_shortcut', 'group2/block5_shortcut', 'group2/block4_shortcut', 'group2/block3_shortcut', 'group2/block2_shortcut', 'group2/block1_shortcut', 'group2/block0_shortcut', 'group1/block3_shortcut', 'group1/block2_shortcut', 'group1/block1_shortcut'],
    
    'vgg':['add','conv5_3','conv5_2','conv5_1','pool4','conv4_3','conv4_2','conv4_1','pool3','conv3_3','conv3_2','conv3_1','pool2','conv2_2','conv2_1','pool1','conv1_2','conv1_1']
    }[bachbone]


def normalize_maps(maps):
    maps = np.array(maps)
    maps = maps - maps.min()
    if maps.max() != 0:
        maps = maps / maps.max()
    return maps
    
FCweights_name = {"cub_inceptionv3": 'InceptionV3/Logits/Conv2d_1c_1x1_200/weights:0',
           "cub_resnet50se": 'linearcub/W:0', 
           "cub_vgg": 'linear_cub/W:0', 
           "imagenet_inceptionv3": 'InceptionV3/Logits/Conv2d_1c_1x1_1000/weights:0', 
           "imagenet_resnet50": 'linear/W:0', 
           "imagenet_resnet50se": 'linear/W:0', 
           "imagenet_vgg": 'linear/W:0'}

class Model(ImageNetModel):
    def __init__(self, mode, dataset):
        assert mode in ['resnet50', 'resnet50se', 'inceptionv3', 'vgg']
        assert dataset in ['imagenet', 'cub']
        classnum_dict = {'imagenet': 1000, 'cub': 200}
        self.classnum = classnum_dict[dataset]
        self.mode = mode
    
    def get_logits(self, image):
        if self.mode == 'vgg':
            with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
                with PredictTowerContext(''):
                    return models_vgg.vgg_gap(image, self.classnum) if self.mode == 'vgg' else resnet_model.resnet_backbone(image, [3, 4, 6, 3],resnet_model.resnet_group, resnet_model.resnet_bottleneck, self.classnum)
        elif self.mode in ['resnet50','resnet50se']:
            with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
                with PredictTowerContext(''):
                    return resnet_model.resnet_backbone(image, [3, 4, 6, 3],resnet_model.resnet_group, resnet_model.se_resnet_bottleneck, self.classnum) if self.mode == 'resnet50se' else resnet_model.resnet_backbone(image, [3, 4, 6, 3],resnet_model.resnet_group, resnet_model.resnet_bottleneck, self.classnum)
        else: 
            is_training = False
            with slim.arg_scope(inceptionv3.inception_v3_arg_scope(weight_decay=0.0005)):
                with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm,normalizer_params={'is_training': is_training}):
                    return inceptionv3.inception_v3(image, self.classnum,is_training=is_training,global_pool=True)#
def get_1maxarg( matrix):
    matrix_max = tf.zeros_like(matrix) + tf.reduce_max(matrix,axis=1,keep_dims=True)
    matrix1 = tf.where(matrix < matrix_max, tf.zeros_like(matrix), matrix)
    return tf.maximum(tf.sign(matrix1) ,0.0)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # generic:
    parser.add_argument('--gpu',  default=0 ,help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', default='/opt/data/private/tcc/DGL2journal/DGL_model/Imagenet_inceptionV3_model-240000',type=str,help='load a model for evaluation')
    parser.add_argument('--data_path',  default='./ILSVRC2012_val_00000062.JPEG')
    # model:
    parser.add_argument('--mode', choices=['resnet50', 'resnet50se', 'inceptionv3', 'vgg'],
                        help='variants of mode to use', default='inceptionv3')
    parser.add_argument('--dataset', choices=['imagenet', 'cub'],
                        help='variants of dataset to use', default='imagenet')
    parser.add_argument('--pa', default=10.0, type=float)
    parser.add_argument('--clayer', default=2, type=int)
    parser.add_argument('--image_size', default=224, type=int)

    args = parser.parse_args()

    eval_graph = tf.Graph()
    with eval_graph.as_default():

        timages = tf.placeholder(tf.string, [1, ])

        model = Model(args.mode, args.dataset)
        # read and pre-process image
        image_size = args.image_size
        image = tf.image.resize_bilinear(tf.expand_dims(tf.image.decode_jpeg(tf.read_file(tf.squeeze(timages)), channels=3),0), [image_size, image_size],align_corners=False)
        preprocessed_images = model.image_preprocess(tf.to_float(tf.reverse(image,axis=[-1])))
        logits, end_points = model.get_logits(preprocessed_images)
        
        prob = tf.nn.softmax(logits)
        prob = tf.expand_dims(tf.squeeze(prob),0)
        logits = tf.expand_dims(tf.squeeze(logits),0)
        argmax_prob = tf.argmax(prob,axis =1)

        cost_H = tf.reduce_sum(tf.multiply(logits,tf.stop_gradient(get_1maxarg(prob))))
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=args.pa*tf.stop_gradient(get_1maxarg(prob)))
        # 2： gradients
        target_layer = end_points[get_layers_name(args.mode)[args.clayer]]
        target_layer_grad = tf.stop_gradient(tf.gradients(cost, target_layer)[0])
        target_layer_grad_H = tf.stop_gradient(tf.gradients(cost_H, target_layer)[0])

        enhanced_map = tf.nn.l2_normalize(target_layer,[1,2,3]) - tf.nn.l2_normalize(target_layer_grad,[1,2,3])
        pixel_level_class_selection = target_layer_grad_H * enhanced_map
        pcs_map = tf.reduce_sum(tf.squeeze(pixel_level_class_selection),2)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    config.gpu_options.allow_growth = True
    with tf.Session(graph=eval_graph,config=config) as sess:

        checkpoint = args.load
        # checkpoint = args.load
        saver.restore(sess, checkpoint)
        print('Loaded checkpoint: {}'.format(checkpoint))
        print(" eval {} on {} layer".format(args.mode, get_layers_name(args.mode)[args.clayer]))
        weights_cam = sess.run(tf.squeeze(eval_graph.get_tensor_by_name(FCweights_name[args.dataset + '_' + args.mode]))).T
        pcs_map_run, images, argmax_prob_run,last_conv_map =  sess.run([pcs_map,preprocessed_images,argmax_prob,end_points[get_layers_name(args.mode)[0]]], feed_dict={timages: [args.data_path]})

        shape = pcs_map_run.shape
        
        # obtain heatmaps for top1 class
        channel = 2048 if 'resnet50' in args.mode else 1024 
        cam = np.dot(weights_cam[argmax_prob_run,:].reshape((1,channel)),last_conv_map[0,:,:,:].transpose(2,0,1).reshape(channel,-1)).reshape(shape)
        
        # normalize attention map
        cam_map_run = normalize_maps(cam)
        pcs_map_run = normalize_maps(pcs_map_run)
        images = normalize_maps(images)
        
        print("pcs_map_run.shape: ",pcs_map_run.shape)
        # save results
        image_bet = 0.3
        dgl_heatmap = cv2.applyColorMap(np.uint8(cv2.resize(pcs_map_run,(args.image_size,args.image_size))*255), cv2.COLORMAP_JET)
        dgl_out = cv2.addWeighted(np.uint8((images[0,:,:,:])*255.0), image_bet, np.uint8(dgl_heatmap), 1-image_bet, 0)
        cv2.imwrite('./'+args.mode + '_' + get_layers_name(args.mode)[args.clayer].replace('/','_')+'_dgl.png', dgl_out)

        cam_heatmap = cv2.applyColorMap(np.uint8(cv2.resize(cam_map_run,(args.image_size,args.image_size))*255), cv2.COLORMAP_JET)
        cam_out = cv2.addWeighted(np.uint8((images[0,:,:,:])*255.0), image_bet, np.uint8(cam_heatmap), 1-image_bet, 0)
        cv2.imwrite('./'+args.mode + '_cam.png', cam_out)

                            