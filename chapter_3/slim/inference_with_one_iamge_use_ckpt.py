# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/slim:export_inference_graph
bazel-bin/tensorflow_models/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255 \
--logtostderr

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'default_image_size', 224,
    'The image size to use if the model does not define it.')

tf.app.flags.DEFINE_string('dataset_name', 'satellite',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'satellite/data', 'Directory to save intermediate dataset files to')

FLAGS = tf.app.flags.FLAGS

import numpy as np

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def main(_):
    # if not FLAGS.output_file:
    #   raise ValueError('You must supply the path to save to with --output_file')

    image='../test_image2.jpg'
    # 设置默认图，在单张输入时可省略
    with tf.Graph().as_default() as graph:

        # # 定义GPU设置
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.visible_device_list = '0'

        # 读取图片，图片格式为tensor
        tf.logging.set_verbosity(tf.logging.INFO)
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        image_data = tf.image.decode_jpeg(image_data)
        image_data = preprocess_for_eval(image_data, 299, 299)
        image_data = tf.expand_dims(image_data, 0)

        # # 使用Session运行tensor，将其转换成需要的格式。如果不使用，可以在输入时直接调用 image_data.eval().
        # with tf.Session(config=config) as sess:
        #     image_data = sess.run(image_data)

        # 根据原格式设置输入数据和模型，
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'validation',
                                              FLAGS.dataset_dir)
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=FLAGS.is_training)

        if hasattr(network_fn, 'default_image_size'):
          image_size = network_fn.default_image_size
        else:
          image_size = FLAGS.default_image_size

        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                     shape=[1, image_size, image_size, 3])
        network_fn(placeholder)

        #　执行计算
        with tf.Session(config=config) as sess:
            # 读取模型参数
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint("./satellite/train_dir")
            saver.restore(sess, ckpt)

            # 提取结果
            softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            predictions = sess.run(softmax_tensor,
                                   {'input:0': image_data.eval()})
            predictions = np.squeeze(predictions)
            print(predictions)



if __name__ == '__main__':
  tf.app.run()
