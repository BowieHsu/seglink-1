import sys, os
import tensorflow as tf
import numpy as np

import ops
import pdb

FLAGS = tf.app.flags.FLAGS


class Vgg16Model():
  def __init__(self):
    self.outputs = {}

  def _vgg_conv_relu(self, x, n_in, n_out, scope, fc7=False, trainable=True):
    with tf.variable_scope(scope):
      if fc7 == False:
        conv = ops.conv2d(x, n_in, n_out, 3, trainable=trainable, relu=True)
      else:
        conv = ops.conv2d(x, n_in, n_out, 1, trainable=trainable, relu=True)
    return conv

  def _vgg_atrous_conv_relu(self, x, n_in, n_out, scope):
    with tf.variable_scope(scope):
      conv = ops.atrous_conv2d(x, n_in, n_out, 3, 6,
                               weight_init='xavier',relu=True)
    return conv

  def _vgg_max_pool(self, x, scope, pool5=False):
    with tf.variable_scope(scope):
      if not pool5:
        pool = ops.max_pool(x, 2, 2, 'SAME')
      else:
        pool = ops.max_pool(x, 3, 1, 'SAME')
    return pool

  def _vgg_conv_vhp(self, x, n_in, n_mid, n_out, scope, trainable=True):
    with tf.variable_scope(scope + '_V'):
      conv_v = ops.conv2d_h_w(x, n_in, n_mid, 3, 1, trainable=trainable, relu=False)
    with tf.variable_scope(scope + '_H'):
      conv_h = ops.conv2d_h_w(conv_v, n_mid, n_mid, 1, 3, trainable=trainable, relu=False)
    with tf.variable_scope(scope + '_P'):
      conv_p = ops.conv2d(conv_h, n_mid, n_out, 1, trainable=trainable, relu=True)
    return conv_p

  def _vgg_conv_vh(self, x, n_in, n_mid, n_out, scope, trainable=True):
    with tf.variable_scope(scope + '_V'):
      conv_v = ops.conv2d_h_w(x, n_in, n_mid, 3, 1, trainable=trainable, relu=False)
    with tf.variable_scope(scope + '_H'):
      conv_h = ops.conv2d_h_w(conv_v, n_mid, n_out, 1, 3, trainable=trainable, relu=False)
    return conv_h

  def build_model(self, images, scope=None):
    with tf.variable_scope(scope or 'vgg16'):
      # conv stage 1
      relu1_1 = self._vgg_conv_relu(images, 3, 64, 'conv1_1', trainable=False)
      pdb.set_trace()
      relu1_2 = self._vgg_conv_vhp(relu1_1, 64, 22, 59, 'conv1_2', trainable=False)
      pool1 = self._vgg_max_pool(relu1_2, 'pool1')

      # conv stage 2
      relu2_1 = self._vgg_conv_vhp(pool1, 59, 37, 118, 'conv2_1', trainable=False)
      relu2_2 = self._vgg_conv_vhp(relu2_1, 118, 47, 119, 'conv2_2', trainable=False)
      pool2 = self._vgg_max_pool(relu2_2, 'pool2')

      # layers below pool2 are freezed
      pool2 = tf.stop_gradient(pool2)
      # conv stage 3
      relu3_1 = self._vgg_conv_vhp(pool2, 119, 83, 226, 'conv3_1')
      relu3_2 = self._vgg_conv_vhp(relu3_1, 226, 89, 243, 'conv3_2')
      relu3_3 = self._vgg_conv_vhp(relu3_2, 243, 106, 256, 'conv3_3')
      pool3 = self._vgg_max_pool(relu3_3, 'pool3')
      # conv stage 4
      relu4_1 = self._vgg_conv_vhp(pool3, 256, 175, 482, 'conv4_1')
      relu4_2 = self._vgg_conv_vhp(relu4_1, 482, 192, 457, 'conv4_2')
      relu4_3 = self._vgg_conv_vhp(relu4_2, 457, 227, 512, 'conv4_3') # => 38 x 38
      pool4 = self._vgg_max_pool(relu4_3, 'pool4')

      # conv stage 5
      relu5_1 = self._vgg_conv_vh(pool4, 512, 398, 512, 'conv5_1')
      relu5_2 = self._vgg_conv_vh(relu5_1, 512, 390, 512, 'conv5_2')
      relu5_3 = self._vgg_conv_vh(relu5_2, 512, 379, 512, 'conv5_3')

      # pool5 has ksize 3 and stride 1
      pool5 = self._vgg_max_pool(relu5_3, 'pool5', pool5=True) # => 19 x 19
      # atrous_conv6 (fc6)
      # relu_fc6 = self._vgg_atrous_conv_relu(pool5, 512, 1024, 'fc6') # => 19 x 19
      # relu_fc7 = self._vgg_conv_relu(relu_fc6, 1024, 1024, 'fc7', fc7=True) # => 19 x 19

      outputs = {
        'conv4_3': relu4_3
        # 'fc7': relu_fc7
      }
      return outputs

