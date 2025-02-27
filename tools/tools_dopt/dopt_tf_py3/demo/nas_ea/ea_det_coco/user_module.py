#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:user_module Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
""" NASEASearch """

from abc import ABCMeta
from abc import abstractmethod
import json
import os
import logging
from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ssd_keras.data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_keras.data_generator.object_detection_2d_geometric_ops import Resize
from ssd_keras.data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_keras.data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from ssd_keras.eval_utils.coco_utils import get_coco_category_maps
from ssd_keras.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_keras.keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_keras.ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_keras.ssd_encoder_decoder.ssd_output_decoder import decode_detections_fast
from ssd_keras.data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


class BasePreNet(tf.keras.Model):
    """
    class BasePreNet
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
    class BasePostNet
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
    class UserModuleInterface
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

    @abstractmethod
    def build_dataset_search(self, dataset_dir, is_training, is_shuffle):
        """
        Build dataset for nas search

        :param dataset_dir:
        :param is_training:
        :param is_shuffle:
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
        :param global_step: a Python number. Global step to use for
        the decay computation.
        :return: A scalar Tensor of the same type as learning_rate.
        The decayed learning rate.
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
    class UserModule
    """

    def __init__(self, epoch, batch_size):
        super(UserModule, self).__init__(epoch, batch_size)
        self.epoch = epoch
        self.batch_size = batch_size
        self.img_height = 300
        self.img_width = 300
        self.predictor_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.n_classes = 80
        self.scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.aspect_ratios = [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5],
        ]
        self.variances = [0.1, 0.1, 0.2, 0.2]
        self.val_batch_size = 1

    def loss_op(self, labels, logits):
        """

        :param labels:
        :param logits:
        :return:
        """
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        loss = ssd_loss.compute_loss(labels, logits)
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
        valid_dir = inputs[0]
        model = inputs[1]
        data_generator = outputs[0]
        proxy_val_image_ids = outputs[1]
        data_size = outputs[2]

        annotations_filename = os.path.join(valid_dir, "annotations/instances_val2017.json")
        _, classes_to_cats, _, _ = get_coco_category_maps(annotations_filename)
        if hvd is not None:
            results_file = 'detections_val2017_ssd300_results_{}.json'.format(hvd.rank())
        else:
            results_file = 'detections_val2017_ssd300_results.json'


        save_model_predict_result(out_file=results_file,
                            model=model,
                            img_height=self.img_height,
                            img_width=self.img_width,
                            classes_to_cats=classes_to_cats,
                            data_generator=data_generator,
                            data_size=data_size,
                            batch_size=self.val_batch_size,
                            proxy_val_image_ids=proxy_val_image_ids)

        coco_gt = COCO(annotations_filename)
        try:
            coco_dt = coco_gt.loadRes(results_file)
        except IndexError:
            logging.error("not sufficient warmup")
            exit(-1)
        image_ids = sorted(proxy_val_image_ids)
        coco_eval = COCOeval(cocoGt=coco_gt,
                            cocoDt=coco_dt,
                            iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        res_map = coco_eval.stats[0]
        return res_map

    def build_dataset_search(self, dataset_dir, is_training=True, is_shuffle=True):
        """

        :param dataset_dir:
        :param is_training:
        :param is_shuffle:
        :return:
        """
        if is_training:
            train_images_dir = os.path.join(dataset_dir, "train2017")
            labels_filename = os.path.join(dataset_dir, "annotations/instances_train2017.json")
            train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
            train_dataset.parse_json(images_dirs=[train_images_dir],
                                     annotations_filenames=[labels_filename],
                                     ground_truth_available=True,
                                     include_classes='all',
                                     ret=False,
                                     verbose=False)
            ssd_data_augmentation = SSDDataAugmentation(img_height=self.img_height,
                                                        img_width=self.img_width,
                                                        background=[123, 117, 104])
            ssd_input_encoder = SSDInputEncoder(img_height=self.img_height,
                                                img_width=self.img_width,
                                                n_classes=self.n_classes,
                                                predictor_sizes=self.predictor_sizes,
                                                scales=self.scales,
                                                aspect_ratios_per_layer=self.aspect_ratios,
                                                two_boxes_for_ar1=True,
                                                steps=self.steps,
                                                offsets=self.offsets,
                                                clip_boxes=False,
                                                variances=self.variances,
                                                matching_type='multi',
                                                pos_iou_threshold=0.5,
                                                neg_iou_limit=0.5,
                                                normalize_coords=True)
            train_generator = train_dataset.generate(batch_size=self.batch_size,
                                                     shuffle=is_shuffle,
                                                     transformations=[ssd_data_augmentation],
                                                     label_encoder=ssd_input_encoder,
                                                     returns={'processed_images',
                                                              'encoded_labels'},
                                                     keep_images_without_gt=False)
            train_dataset_size = train_dataset.get_dataset_size()
            return train_generator, train_dataset_size

        images_dir = os.path.join(dataset_dir, "val2017")
        filename = os.path.join(dataset_dir, "annotations/instances_val2017.json")
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        val_dataset.parse_json(images_dirs=[images_dir],
                               annotations_filenames=[filename],
                               ground_truth_available=True,
                               include_classes='all',
                               ret=False,
                               verbose=False)
        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=self.img_height, width=self.img_width)
        transformations = [
            convert_to_3_channels,
            resize,
        ]
        val_generator = val_dataset.generate(batch_size=self.val_batch_size,
                                             shuffle=is_shuffle,
                                             transformations=transformations,
                                             label_encoder=None,
                                             returns={'processed_images',
                                                      'image_ids',
                                                      'inverse_transform'},
                                             keep_images_without_gt=True,
                                             is_get_proxy=True)
        val_dataset_size = val_dataset.get_dataset_size()
        return [val_generator, val_dataset], val_dataset_size


class PreNet(BasePreNet):
    """
    class PreNet
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
    class PostNet
    """

    def __init__(self):
        super(PostNet, self).__init__()
        self.act = tf.nn.relu
        self.img_height = 300
        self.img_width = 300
        self.l2_regularization = 0.0005
        self.aspect_ratios_global = None
        self.aspect_ratios_per_layer = [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5],
        ]
        self.n_classes = 80 + 1
        self.scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        l2_reg = self.l2_regularization  # Make the internal name shorter.
        self.steps = [8, 16, 32, 64, 100, 300]
        self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.two_boxes_for_ar1 = True
        self.variances = [0.1, 0.1, 0.2, 0.2]
        self.clip_boxes = False
        self.coords = 'centroids'
        self.normalize_coords = True

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        if self.aspect_ratios_per_layer:
            n_boxes = []
            for aspect_ratio in self.aspect_ratios_per_layer:
                if (1 in aspect_ratio) & self.two_boxes_for_ar1:
                    n_boxes.append(len(aspect_ratio) + 1)
                else:
                    n_boxes.append(len(aspect_ratio))

        self.fc6 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3, 3], dilation_rate=(6, 6),
                                          padding='same',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=l2(l2_reg),
                                          activation=self.act, name='fc6')

        self.fc7 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], padding='same',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=l2(l2_reg),
                                          activation=self.act, name='fc7')

        self.conv6_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], padding='same',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv6_1')
        self.conv6_1_pad = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)),
                                                         name='conv6_padding')
        self.conv6_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=(2, 2),
                                              padding='valid',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv6_2')

        self.conv7_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv7_1')
        self.conv7_1_pad = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)),
                                                         name='conv7_padding')
        self.conv7_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=(2, 2),
                                              padding='valid',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv7_2')

        self.conv8_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv8_1')
        self.conv8_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                                              padding='valid',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv8_2')

        self.conv9_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv9_1')
        self.conv9_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                                              padding='valid',
                                              kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg),
                                              activation=self.act, name='conv9_2')

        # Build the convolutional predictor layers on top of the base network
        # We precidt `n_classes` confidence values for each box, hence the confidence predictors
        # have depth `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`

        self.conv4_3_norm_mbox_conf = tf.keras.layers.Conv2D(filters=n_boxes[0] * self.n_classes,
                                                             kernel_size=[3, 3],
                                                             padding='same',
                                                             kernel_initializer='he_normal',
                                                             kernel_regularizer=l2(l2_reg),
                                                             activation=None,
                                                             name='conv4_3_norm_mbox_conf')

        self.fc7_mbox_conf = tf.keras.layers.Conv2D(filters=n_boxes[1] * self.n_classes,
                                                    kernel_size=[3, 3],
                                                    padding='same',
                                                    kernel_initializer='he_normal',
                                                    kernel_regularizer=l2(l2_reg),
                                                    activation=None, name='fc7_mbox_conf')

        self.conv6_2_mbox_conf = tf.keras.layers.Conv2D(filters=n_boxes[2] * self.n_classes,
                                                        kernel_size=[3, 3],
                                                        padding='same',
                                                        kernel_initializer='he_normal',
                                                        kernel_regularizer=l2(l2_reg),
                                                        activation=None, name='conv6_2_mbox_conf')

        self.conv7_2_mbox_conf = tf.keras.layers.Conv2D(filters=n_boxes[3] * self.n_classes,
                                                        kernel_size=[3, 3],
                                                        padding='same',
                                                        kernel_initializer='he_normal',
                                                        kernel_regularizer=l2(l2_reg),
                                                        activation=None, name='conv7_2_mbox_conf')

        self.conv8_2_mbox_conf = tf.keras.layers.Conv2D(filters=n_boxes[4] * self.n_classes,
                                                        kernel_size=[3, 3],
                                                        padding='same',
                                                        kernel_initializer='he_normal',
                                                        kernel_regularizer=l2(l2_reg),
                                                        activation=None, name='conv8_2_mbox_conf')

        self.conv9_2_mbox_conf = tf.keras.layers.Conv2D(filters=n_boxes[5] * self.n_classes,
                                                        kernel_size=[3, 3],
                                                        padding='same',
                                                        kernel_initializer='he_normal',
                                                        kernel_regularizer=l2(l2_reg),
                                                        activation=None, name='conv9_2_mbox_conf')

        # We predict 4 box coordinates for each box, hence the localization predictors have
        # depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        self.conv4_3_norm_mbox_loc = tf.keras.layers.Conv2D(filters=n_boxes[0] * 4,
                                                            kernel_size=[3, 3], padding='same',
                                                            kernel_initializer='he_normal',
                                                            kernel_regularizer=l2(l2_reg),
                                                            activation=None,
                                                            name='conv4_3_norm_mbox_loc')

        self.fc7_mbox_loc = tf.keras.layers.Conv2D(filters=n_boxes[1] * 4, kernel_size=[3, 3],
                                                   padding='same',
                                                   kernel_initializer='he_normal',
                                                   kernel_regularizer=l2(l2_reg),
                                                   activation=None, name='fc7_mbox_loc')

        self.conv6_2_mbox_loc = tf.keras.layers.Conv2D(filters=n_boxes[2] * 4, kernel_size=[3, 3],
                                                       padding='same',
                                                       kernel_initializer='he_normal',
                                                       kernel_regularizer=l2(l2_reg),
                                                       activation=None, name='conv6_2_mbox_loc')

        self.conv7_2_mbox_loc = tf.keras.layers.Conv2D(filters=n_boxes[3] * 4, kernel_size=[3, 3],
                                                       padding='same',
                                                       kernel_initializer='he_normal',
                                                       kernel_regularizer=l2(l2_reg),
                                                       activation=None, name='conv7_2_mbox_loc')

        self.conv8_2_mbox_loc = tf.keras.layers.Conv2D(filters=n_boxes[4] * 4, kernel_size=[3, 3],
                                                       padding='same',
                                                       kernel_initializer='he_normal',
                                                       kernel_regularizer=l2(l2_reg),
                                                       activation=None, name='conv8_2_mbox_loc')

        self.conv9_2_mbox_loc = tf.keras.layers.Conv2D(filters=n_boxes[5] * 4, kernel_size=[3, 3],
                                                       padding='same',
                                                       kernel_initializer='he_normal',
                                                       kernel_regularizer=l2(l2_reg),
                                                       activation=None, name='conv9_2_mbox_loc')

    def call(self, inputs, training=True):
        """

        :param inputs:
        :param training:
        :return:
        """

        ############################################################################
        # Compute the anchor box parameters.
        ############################################################################
        tbs_feature_choose1, tbs_feature_choose2, inputs = inputs[0], inputs[1], inputs[2]
        # Set the aspect ratios for each predictor layer. These are only needed for
        # the anchor box layers.
        fc6 = self.fc6(inputs)
        fc7 = self.fc7(fc6)
        conv6_1 = self.conv6_1(fc7)
        conv6_1_pad = self.conv6_1_pad(conv6_1)
        conv6_2 = self.conv6_2(conv6_1_pad)

        conv7_1 = self.conv7_1(conv6_2)
        conv7_1_pad = self.conv7_1_pad(conv7_1)
        conv7_2 = self.conv7_2(conv7_1_pad)

        conv8_1 = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(conv8_1)

        # Feed tbs_feature_choose1 into the L2 normalization layer
        resnet_stage1_norm = L2Normalization(gamma_init=20,
                                             name='conv4_3_norm')(tbs_feature_choose1)

        ### Build the convolutional predictor layers on top of the base network
        # We precidt `n_classes` confidence values for each box, hence the confidence predictors
        # have depth `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
        conv4_3_norm_mbox_conf = self.conv4_3_norm_mbox_conf(resnet_stage1_norm)
        fc7_mbox_conf = self.fc7_mbox_conf(tbs_feature_choose2)
        conv6_2_mbox_conf = self.conv6_2_mbox_conf(fc7)
        conv7_2_mbox_conf = self.conv7_2_mbox_conf(conv6_2)
        conv8_2_mbox_conf = self.conv8_2_mbox_conf(conv7_2)
        conv9_2_mbox_conf = self.conv9_2_mbox_conf(conv8_2)

        # We predict 4 box coordinates for each box, hence the localization predictors
        # have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        conv4_3_norm_mbox_loc = self.conv4_3_norm_mbox_loc(resnet_stage1_norm)
        fc7_mbox_loc = self.fc7_mbox_loc(tbs_feature_choose2)
        conv6_2_mbox_loc = self.conv6_2_mbox_loc(fc7)
        conv7_2_mbox_loc = self.conv7_2_mbox_loc(conv6_2)
        conv8_2_mbox_loc = self.conv8_2_mbox_loc(conv7_2)
        conv9_2_mbox_loc = self.conv9_2_mbox_loc(conv8_2)
        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        conv4_3_norm_mbox_priorbox = AnchorBoxes(self.img_height, self.img_width,
                                                 this_scale=self.scales[0],
                                                 next_scale=self.scales[1],
                                                 aspect_ratios=self.aspect_ratios_per_layer[0],
                                                 two_boxes_for_ar1=self.two_boxes_for_ar1,
                                                 this_steps=self.steps[0],
                                                 this_offsets=self.offsets[0],
                                                 clip_boxes=self.clip_boxes,
                                                 variances=self.variances, coords=self.coords,
                                                 normalize_coords=self.normalize_coords,
                                                 name='conv4_3_norm_mbox_priorbox')(
            conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxes(self.img_height, self.img_width, this_scale=self.scales[1],
                                        next_scale=self.scales[2],
                                        aspect_ratios=self.aspect_ratios_per_layer[1],
                                        two_boxes_for_ar1=self.two_boxes_for_ar1,
                                        this_steps=self.steps[1],
                                        this_offsets=self.offsets[1], clip_boxes=self.clip_boxes,
                                        variances=self.variances, coords=self.coords,
                                        normalize_coords=self.normalize_coords,
                                        name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxes(self.img_height, self.img_width,
                                            this_scale=self.scales[2],
                                            next_scale=self.scales[3],
                                            aspect_ratios=self.aspect_ratios_per_layer[2],
                                            two_boxes_for_ar1=self.two_boxes_for_ar1,
                                            this_steps=self.steps[2], this_offsets=self.offsets[2],
                                            clip_boxes=self.clip_boxes,
                                            variances=self.variances, coords=self.coords,
                                            normalize_coords=self.normalize_coords,
                                            name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxes(self.img_height, self.img_width,
                                            this_scale=self.scales[3],
                                            next_scale=self.scales[4],
                                            aspect_ratios=self.aspect_ratios_per_layer[3],
                                            two_boxes_for_ar1=self.two_boxes_for_ar1,
                                            this_steps=self.steps[3], this_offsets=self.offsets[3],
                                            clip_boxes=self.clip_boxes,
                                            variances=self.variances, coords=self.coords,
                                            normalize_coords=self.normalize_coords,
                                            name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxes(self.img_height, self.img_width,
                                            this_scale=self.scales[4],
                                            next_scale=self.scales[5],
                                            aspect_ratios=self.aspect_ratios_per_layer[4],
                                            two_boxes_for_ar1=self.two_boxes_for_ar1,
                                            this_steps=self.steps[4], this_offsets=self.offsets[4],
                                            clip_boxes=self.clip_boxes,
                                            variances=self.variances, coords=self.coords,
                                            normalize_coords=self.normalize_coords,
                                            name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxes(self.img_height, self.img_width,
                                            this_scale=self.scales[5],
                                            next_scale=self.scales[6],
                                            aspect_ratios=self.aspect_ratios_per_layer[5],
                                            two_boxes_for_ar1=self.two_boxes_for_ar1,
                                            this_steps=self.steps[5], this_offsets=self.offsets[5],
                                            clip_boxes=self.clip_boxes,
                                            variances=self.variances, coords=self.coords,
                                            normalize_coords=self.normalize_coords,
                                            name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
        ### Reshape
        # Reshape the class predictions, yielding 3D tensors of shape
        # `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        conv4_3_norm_mbox_conf_reshape = tf.keras.layers.Reshape((-1, self.n_classes),
                                                      name='conv4_3_norm_mbox_conf_reshape')(
            conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape = tf.keras.layers.Reshape((-1, self.n_classes),
                                                        name='fc7_mbox_conf_reshape')(
            fc7_mbox_conf)
        conv6_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, self.n_classes),
                                                            name='conv6_2_mbox_conf_reshape')(
            conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, self.n_classes),
                                                            name='conv7_2_mbox_conf_reshape')(
            conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, self.n_classes),
                                                            name='conv8_2_mbox_conf_reshape')(
            conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, self.n_classes),
                                                            name='conv9_2_mbox_conf_reshape')(
            conv9_2_mbox_conf)

        # Reshape the box predictions, yielding 3D tensors of shape
        # `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        conv4_3_norm_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4),
                                                      name='conv4_3_norm_mbox_loc_reshape')(
            conv4_3_norm_mbox_loc)
        fc7_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4),
                                                       name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
        conv6_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4),
                                                           name='conv6_2_mbox_loc_reshape')(
            conv6_2_mbox_loc)
        conv7_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4),
                                                           name='conv7_2_mbox_loc_reshape')(
            conv7_2_mbox_loc)
        conv8_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4),
                                                           name='conv8_2_mbox_loc_reshape')(
            conv8_2_mbox_loc)
        conv9_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4),
                                                           name='conv9_2_mbox_loc_reshape')(
            conv9_2_mbox_loc)

        # Reshape the anchor box tensors, yielding 3D tensors of shape
        # `(batch, height * width * n_boxes, 8)`
        conv4_3_mbox_priorbox_reshape = \
            tf.keras.layers.Reshape((-1, 8), name='conv4_3_mbox_priorbox_reshape')(
                conv4_3_norm_mbox_priorbox)
        fc7_mbox_priorbox_reshape = tf.keras.layers.Reshape((-1, 8),
                                                    name='fc7_mbox_priorbox_reshape')(
            fc7_mbox_priorbox)
        conv6_2_mbox_priorbox_reshape = tf.keras.layers.Reshape((-1, 8),
                                                     name='conv6_2_mbox_priorbox_reshape')(
            conv6_2_mbox_priorbox)
        conv7_2_mbox_priorbox_reshape = tf.keras.layers.Reshape((-1, 8),
                                                     name='conv7_2_mbox_priorbox_reshape')(
            conv7_2_mbox_priorbox)
        conv8_2_mbox_priorbox_reshape = tf.keras.layers.Reshape((-1, 8),
                                                     name='conv8_2_mbox_priorbox_reshape')(
            conv8_2_mbox_priorbox)
        conv9_2_mbox_priorbox_reshape = tf.keras.layers.Reshape((-1, 8),
                                                     name='conv9_2_mbox_priorbox_reshape')(
            conv9_2_mbox_priorbox)

        ### Concatenate the predictions from the different layers

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively)
        # are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        mbox_conf = tf.keras.layers.concatenate([conv4_3_norm_mbox_conf_reshape,
                                                 fc7_mbox_conf_reshape,
                                                 conv6_2_mbox_conf_reshape,
                                                 conv7_2_mbox_conf_reshape,
                                                 conv8_2_mbox_conf_reshape,
                                                 conv9_2_mbox_conf_reshape],
                                                axis=1, name='mbox_conf')

        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        mbox_loc = tf.keras.layers.concatenate([conv4_3_norm_mbox_loc_reshape,
                                                fc7_mbox_loc_reshape,
                                                conv6_2_mbox_loc_reshape,
                                                conv7_2_mbox_loc_reshape,
                                                conv8_2_mbox_loc_reshape,
                                                conv9_2_mbox_loc_reshape],
                                               axis=1, name='mbox_loc')

        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        mbox_priorbox = tf.keras.layers.concatenate([conv4_3_mbox_priorbox_reshape,
                                                     fc7_mbox_priorbox_reshape,
                                                     conv6_2_mbox_priorbox_reshape,
                                                     conv7_2_mbox_priorbox_reshape,
                                                     conv8_2_mbox_priorbox_reshape,
                                                     conv9_2_mbox_priorbox_reshape],
                                                    axis=1, name='mbox_priorbox')

        # The box coordinate predictions will go into the loss function just the way they are,
        # but for the class predictions, we'll apply a softmax activation layer first
        mbox_conf_softmax = tf.keras.activations.softmax(mbox_conf)

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        predictions = tf.keras.layers.concatenate([mbox_conf_softmax, mbox_loc, mbox_priorbox],
                                                  axis=2,
                                                  name='predictions')

        return predictions


def save_model_predict_result(out_file,
                        model,
                        img_height,
                        img_width,
                        classes_to_cats,
                        data_generator,
                        data_size,
                        batch_size,
                        proxy_val_image_ids=None
                        ):
    """

    :param out_file:
    :param model:
    :param img_height:
    :param img_width:
    :param classes_to_cats:
    :param data_generator:
    :param data_size:
    :param batch_size:
    :param proxy_val_image_ids:
    :return:
    """
    results = []
    if batch_size <= 0:
        raise Exception('Batch_size must be greater than 0!')
    n_batches = int(ceil(data_size / batch_size))
    for _ in range(n_batches):
        batch_x, batch_image_ids, batch_inverse_transforms = next(data_generator)
        batch_x = tf.cast(batch_x, tf.float32)
        if batch_image_ids[0] in proxy_val_image_ids:
            y_pred = model(batch_x, training=False)
        else:
            continue
        y_pred = np.array(y_pred)
        y_pred = decode_detections_fast(y_pred,
                                        confidence_thresh=0.01,
                                        top_k=200,
                                        img_height=img_height,
                                        img_width=img_width)

        y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)

        # Convert each predicted box into the results format.
        for k, batch_item in enumerate(y_pred):
            for box in batch_item:
                boxes_xmin = float(round(box[2], 1))
                boxes_ymin = float(round(box[3], 1))
                boxes_width = float(round(box[4] - box[2], 1))
                boxes_height = float(round(box[5] - box[3], 1))
                new_boxes = {
                    'image_id': batch_image_ids[k],
                    'category_id': classes_to_cats[box[0]],
                    'score': float(round(box[1], 3)),
                    'bbox': [boxes_xmin, boxes_ymin, boxes_width, boxes_height],
                }
                results.append(new_boxes)

    with os.fdopen(
        os.open(
            out_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o700
        ), 'w'
    ) as fwriter:
        json.dump(results, fwriter)

