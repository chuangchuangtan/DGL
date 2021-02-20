#!/bin/bash


# python dgl_resnet.py \
       # --gpu 0,1 \
       # --data /opt/data/private/tcc/data/data/imagenet/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/ \
       # --dataname ILSVRC12 \
       # --imagenet \
       # --base-lr 0.002 \
       # --logdir imagenet_resnet50se \
       # --load resnet50 \
       # --batch 32 \
       # --mode se \
       # --epoch 2 \
       # --final-size 224

# python dgl_resnet.py \
       # --gpu 0,1 \
       # --data /opt/data/private/tcc/data/data/imagenet/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/ \
       # --dataname ILSVRC12 \
       # --imagenet \
       # --base-lr 0.002 \
       # --logdir imagenet_resnet50 \
       # --load resnet50 \
       # --batch 32 \
       # --mode resnet \
       # --epoch 30 \
       # --final-size 224

# python dgl_inceptionv3.py \
       # --gpu 0,1 \
       # --data /opt/data/private/tcc/data/data/imagenet/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/ \
       # --dataname ILSVRC12 \
       # --imagenet \
       # --base-lr 0.002 \
       # --logdir imagenet_inceptionv3 \
       # --load inceptionv3 \
       # --batch 32 \
       # --mode resnet \
       # --epoch 30 \
       # --final-size 224


python dgl_vgg.py \
       --gpu 0,1 \
       --data /opt/data/private/tcc/data/data/imagenet/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/ \
       --dataname ILSVRC12 \
       --imagenet \
       --base-lr 0.002 \
       --logdir imagenet_vgg \
       --load vgg \
       --batch 32 \
       --mode resnet \
       --epoch 30 \
       --final-size 224


