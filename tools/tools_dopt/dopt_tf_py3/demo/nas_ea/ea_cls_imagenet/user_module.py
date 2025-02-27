#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:user_module Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
""" NASEASearch """

import multiprocessing
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

if tf.__version__ == '1.12.0':
    from official.resnet.imagenet_main import input_fn, NUM_CLASSES, NUM_IMAGES
elif tf.__version__ == '2.1.0' or tf.__version__.startswith('2.8'):
    from official.vision.image_classification.imagenet_preprocessing import input_fn, \
        NUM_IMAGES, NUM_CLASSES


class BasePreNet(tf.keras.Model):
    """
    BasePreNet
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        super(BasePreNet, self).__init__()

    @abstractmethod
    def call(self, inputs, training=True):
        """
        Build model's input macro-architecture.

        :param inputs:
        :param training:
        :return: A tensor - input of the TBS block.
        """
        raise NotImplementedError


class BasePostNet(tf.keras.Model):
    """
    BasePostNet
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        super(BasePostNet, self).__init__()

    @abstractmethod
    def call(self, inputs, training=True):
        """
        Build model's output macro-architecture.

        :param inputs:
        :param training:
        :return: A tensor - model's output.
        """
        raise NotImplementedError


class UserModuleInterface:
    """
    UserModuleInterface
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

    @abstractmethod
    def build_dataset_search(self, dataset_dir, is_training):
        """
        Build dataset for nas search

        :param dataset_dir:
        :param is_training:
        :return:  dataset iterator and data num
        """
        raise NotImplementedError

    @abstractmethod
    def loss_op(self, labels, logits):
        """
        Loss Function

        :param labels: GT labels.
        :param logits: logits of network's forward pass.
        :return: A Tensor - loss function's loss
        """
        raise NotImplementedError

    @abstractmethod
    def lr_scheduler(self, lr_init, global_step):
        """
        Define learning rate update scheduler.

        :param lr_init: a Python number. The initial learning rate.
        :param global_step: a Python number. Global step to use for the decay computation.
        :return: A scalar Tensor of the same type as learning_rate. The decayed learning rate.
        """
        raise NotImplementedError

    @abstractmethod
    def metrics_op(self, inputs, outputs):
        """
        define accuracy function

        :param inputs: GT labels.
        :param outputs: outputs of network's forward pass.
        :return:
        """
        raise NotImplementedError


class UserModule(UserModuleInterface):
    """
    UserModule
    """

    def __init__(self, epoch, batch_size):
        super(UserModule, self).__init__(epoch, batch_size)
        self.epoch = epoch
        self.batch_size = batch_size

    def loss_op(self, labels, logits):
        """

        :param labels:
        :param logits:
        :return:
        """
        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(prediction_loss)
        return loss

    def lr_scheduler(self, lr_init, global_step):
        """

        :param lr_init:
        :param global_step:
        :return:
        """
        learn_rate = tf.constant(lr_init, name="learning_rate")
        return learn_rate

    def metrics_op(self, inputs, outputs):
        """

        :param inputs:
        :param outputs:
        :return:
        """
        inputs = tf.cast(inputs, tf.int64)
        inputs = tf.squeeze(inputs)
        prediction_accuracy = tf.cast(tf.keras.backend.in_top_k(outputs, inputs, 1), tf.float32)
        metrics = tf.reduce_mean(prediction_accuracy)
        return metrics

    def build_dataset_search(self, dataset_dir, is_training):
        """

        :param dataset_dir:
        :param is_training:
        :return:
        """
        if is_training:
            data_num = NUM_IMAGES['train']
        else:
            data_num = NUM_IMAGES['validation']

        cpu_num = multiprocessing.cpu_count()
        dataset = input_fn(
            is_training=is_training,
            data_dir=dataset_dir,
            batch_size=self.batch_size,
            num_epochs=self.epoch,
            datasets_num_private_threads=cpu_num // 2)
        return dataset, data_num


class PreNet(BasePreNet):
    """
    PreNet
    """

    def __init__(self):
        super(PreNet, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(64, [7, 7], strides=[2, 2],
                                            padding='same', name='in_conv1_ds0')
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=3, momentum=0.9, name='pre/bn')
        self.act = tf.keras.layers.ReLU()
        self.max_pool = tf.keras.layers.MaxPooling2D([3, 3], strides=2, padding='SAME')

    def call(self, inputs, training=True):
        """

        :param inputs:
        :param training:
        :return:
        """
        output = self.conv0(inputs)
        output = self.batch_norm(output, training=training)
        output = self.act(output)
        output = self.max_pool(output)
        return output


class PostNet(BasePostNet):
    """
    PostNet
    """

    def __init__(self):
        super(PostNet, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected = tf.keras.layers.Dense(NUM_CLASSES,
                                                     activation='softmax', name='fullconnection')

    def call(self, inputs, training=True):
        """

        :param inputs:
        :param training:
        :return:
        """
        output = self.pool(inputs)
        output = self.fully_connected(output)
        return output

