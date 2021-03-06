import sys, os
import tensorflow as tf
import joblib
import numpy as np
import argparse

parentdir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(parentdir, '../../src'))
import model_vgg16

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_scope', default='vgg16',
                    help='Scope for the tensorflow model.')
parser.add_argument('--ckpt_path', default='../model/VGG_ILSVRC_16_layers_ssd.ckpt',
                    help='Checkpoint save path.')
parser.add_argument('--caffe_weights_path', default='../model/VGG_ILSVRC_16_layers_weights.pkl',
                    help='weights dump path.')
args = parser.parse_args()


def convert_caffemodel_to_ckpt():
  caffe_weights = joblib.load(args.caffe_weights_path)

  # create network
  vgg16 = model_vgg16.Vgg16Model()
  model_scope = args.model_scope
  vgg16.build_model(tf.placeholder(tf.float32, shape=[1,300,300,3]), scope=model_scope)

  # auxillary functions for conversion
  def load_conv_weight(target_name, src_name):
    target_name = model_scope + '/' + target_name
    # [n_out, n_in, h, w] => [h, w, n_in, n_out]
    src = np.transpose(caffe_weights[src_name][0], (2,3,1,0))
    return tf.assign(tf.get_variable(target_name), src)

  def load_conv_bias(target_name, src_name):
    target_name = model_scope + '/' + target_name
    src = caffe_weights[src_name][1]
    return tf.assign(tf.get_variable(target_name), src)

  # loding caffemodel weights
  with tf.Session() as session:
    tf.get_variable_scope().reuse_variables()
    assigns = [
      load_conv_weight('conv1_1/conv2d/weight', 'conv1_1'),
      load_conv_bias('conv1_1/conv2d/bias', 'conv1_1'),
      load_conv_weight('conv1_2_V/conv2d/weight', 'conv1_2_V'),
      load_conv_bias('conv1_2_V/conv2d/bias', 'conv1_2_V'),
      load_conv_weight('conv1_2_H/conv2d/weight', 'conv1_2_H'),
      load_conv_bias('conv1_2_H/conv2d/bias', 'conv1_2_H'),
      load_conv_weight('conv1_2_P/conv2d/weight', 'conv1_2_P'),
      load_conv_bias('conv1_2_P/conv2d/bias', 'conv1_2_P'),

      load_conv_weight('conv2_1_V/conv2d/weight', 'conv2_1_V'),
      load_conv_bias('conv2_1_V/conv2d/bias', 'conv2_1_V'),
      load_conv_weight('conv2_1_H/conv2d/weight', 'conv2_1_H'),
      load_conv_bias('conv2_1_H/conv2d/bias', 'conv2_1_H'),
      load_conv_weight('conv2_1_P/conv2d/weight', 'conv2_1_P'),
      load_conv_bias('conv2_1_P/conv2d/bias', 'conv2_1_P'),

      load_conv_weight('conv2_2_V/conv2d/weight', 'conv2_2_V'),
      load_conv_bias('conv2_2_V/conv2d/bias', 'conv2_2_V'),
      load_conv_weight('conv2_2_H/conv2d/weight', 'conv2_2_H'),
      load_conv_bias('conv2_2_H/conv2d/bias', 'conv2_2_H'),
      load_conv_weight('conv2_2_P/conv2d/weight', 'conv2_2_P'),
      load_conv_bias('conv2_2_P/conv2d/bias', 'conv2_2_P'),

      load_conv_weight('conv3_1_V/conv2d/weight', 'conv3_1_V'),
      load_conv_bias('conv3_1_V/conv2d/bias', 'conv3_1_V'),
      load_conv_weight('conv3_1_H/conv2d/weight', 'conv3_1_H'),
      load_conv_bias('conv3_1_H/conv2d/bias', 'conv3_1_H'),
      load_conv_weight('conv3_1_P/conv2d/weight', 'conv3_1_P'),
      load_conv_bias('conv3_1_P/conv2d/bias', 'conv3_1_P'),

      load_conv_weight('conv3_2_V/conv2d/weight', 'conv3_2_V'),
      load_conv_bias('conv3_2_V/conv2d/bias', 'conv3_2_V'),
      load_conv_weight('conv3_2_H/conv2d/weight', 'conv3_2_H'),
      load_conv_bias('conv3_2_H/conv2d/bias', 'conv3_2_H'),
      load_conv_weight('conv3_2_P/conv2d/weight', 'conv3_2_P'),
      load_conv_bias('conv3_2_P/conv2d/bias', 'conv3_2_P'),

      load_conv_weight('conv3_3_V/conv2d/weight', 'conv3_3_V'),
      load_conv_bias('conv3_3_V/conv2d/bias', 'conv3_3_V'),
      load_conv_weight('conv3_3_H/conv2d/weight', 'conv3_3_H'),
      load_conv_bias('conv3_3_H/conv2d/bias', 'conv3_3_H'),
      load_conv_weight('conv3_3_P/conv2d/weight', 'conv3_3_P'),
      load_conv_bias('conv3_3_P/conv2d/bias', 'conv3_3_P'),

      load_conv_weight('conv4_1_V/conv2d/weight', 'conv4_1_V'),
      load_conv_bias('conv4_1_V/conv2d/bias', 'conv4_1_V'),
      load_conv_weight('conv4_1_H/conv2d/weight', 'conv4_1_H'),
      load_conv_bias('conv4_1_H/conv2d/bias', 'conv4_1_H'),
      load_conv_weight('conv4_1_P/conv2d/weight', 'conv4_1_P'),
      load_conv_bias('conv4_1_P/conv2d/bias', 'conv4_1_P'),

      load_conv_weight('conv4_2_V/conv2d/weight', 'conv4_2_V'),
      load_conv_bias('conv4_2_V/conv2d/bias', 'conv4_2_V'),
      load_conv_weight('conv4_2_H/conv2d/weight', 'conv4_2_H'),
      load_conv_bias('conv4_2_H/conv2d/bias', 'conv4_2_H'),
      load_conv_weight('conv4_2_P/conv2d/weight', 'conv4_2_P'),
      load_conv_bias('conv4_2_P/conv2d/bias', 'conv4_2_P'),

      load_conv_weight('conv4_3_V/conv2d/weight', 'conv4_3_V'),
      load_conv_bias('conv4_3_V/conv2d/bias', 'conv4_3_V'),
      load_conv_weight('conv4_3_H/conv2d/weight', 'conv4_3_H'),
      load_conv_bias('conv4_3_H/conv2d/bias', 'conv4_3_H'),
      load_conv_weight('conv4_3_P/conv2d/weight', 'conv4_3_P'),
      load_conv_bias('conv4_3_P/conv2d/bias', 'conv4_3_P'),

      load_conv_weight('conv5_1_V/conv2d/weight', 'conv5_1_V'),
      load_conv_bias('conv5_1_V/conv2d/bias', 'conv5_1_V'),
      load_conv_weight('conv5_1_H/conv2d/weight', 'conv5_1_H'),
      load_conv_bias('conv5_1_H/conv2d/bias', 'conv5_1_H'),
      load_conv_weight('conv5_2_V/conv2d/weight', 'conv5_2_V'),
      load_conv_bias('conv5_2_V/conv2d/bias', 'conv5_2_V'),
      load_conv_weight('conv5_2_H/conv2d/weight', 'conv5_2_H'),
      load_conv_bias('conv5_2_H/conv2d/bias', 'conv5_2_H'),
      load_conv_weight('conv5_3_V/conv2d/weight', 'conv5_3_V'),
      load_conv_bias('conv5_3_V/conv2d/bias', 'conv5_3_V'),
      load_conv_weight('conv5_3_H/conv2d/weight', 'conv5_3_H'),
      load_conv_bias('conv5_3_H/conv2d/bias', 'conv5_3_H'),
    ]
    with tf.control_dependencies(assigns):
      load_op = tf.no_op(name='load_op')
    session.run(load_op)

    # save checkpoint
    saver = tf.train.Saver()
    saver.save(session, args.ckpt_path)


if __name__ == '__main__':
  convert_caffemodel_to_ckpt()
