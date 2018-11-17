from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim

# _CONV_DEFS specifies the resnet body
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['kernel', 'stride', 'depth', 'num', 't']) # t is the expension factor
_CONV_DEFS = [
    Conv(kernel=[7, 7], stride=2, depth=64),
    # InvertedResidual(kernel=[3, 3], stride=1, depth=64, num=3, t=0.25),
    InvertedResidual(kernel=[3, 3], stride=1, depth=256, num=3, t=0.25),
    InvertedResidual(kernel=[3, 3], stride=2, depth=512, num=4, t=0.25),
    InvertedResidual(kernel=[3, 3], stride=2, depth=1024, num=6, t=0.25),
    InvertedResidual(kernel=[3, 3], stride=2, depth=2048, num=3, t=0.25),    
    # InvertedResidual(kernel=[3, 3], stride=2, depth=160, num=3, t=0.25),
    # InvertedResidual(kernel=[3, 3], stride=1, depth=320, num=1, t=0.25),
    # Conv(kernel=[1, 1], stride=1, depth=1280)
]
def subsample(inputs, factor, scope=None):
      
  if factor == 1:
    return inputs
  else:
    return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

@slim.add_arg_scope
def _inverted_residual_bottleneck(inputs, depth, stride, expand_ratio, lambda_decay, scope=None):
  with tf.variable_scope(scope, 'InvertedResidual', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(
        inputs, activation_fn=tf.nn.relu, scope='preact')
    output = slim.conv2d(preact, expand_ratio*inputs.get_shape().as_list()[-1], 1, stride=1,
                              activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope='conv1')
    """
    slim.conv2d(inputs,num_outputs,kernel_size,stride=1, padding='SAME',data_format=None,rate=1,activation_fn=nn.relu,normalizer_fn=None,
          normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),biases_regularizer=None,
          reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
    """
    output = slim.conv2d(inputs, expand_ratio*inputs.get_shape().as_list()[-1], 3, stride=stride,
                              activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope='conv2')
    output = slim.conv2d(output, depth, 1, stride=1,
                              activation_fn=None, normalizer_fn=None, scope='conv3')

    if depth==depth_in:
      shortcut = subsample(inputs, stride)
    else:
      shortcut = slim.conv2d(inputs, depth, 1, stride=stride,
                              activation_fn=None, normalizer_fn=None, 
                              scope='shortcut')
    # print('shortcht', shortcut.get_shape())
    lambda_c = slim.model_variable('lambda_c', shape=[1, 1, 1, output.get_shape().as_list()[-1]],
                       initializer=tf.ones_initializer(),
                       regularizer=slim.l1_regularizer(lambda_decay)
                       )
    lambda_c = soft_thresholding(lambda_c, lambda_decay)
    output = tf.multiply(output, tf.tile(lambda_c, 
                                 [output.get_shape().as_list()[0], output.get_shape().as_list()[1], output.get_shape().as_list()[2], 1]))
    output = shortcut + output

    return output



def resnet_v2_base(inputs,
                      lambda_decay,
                      min_depth=8,
                      depth_multiplier=1.0,
                      final_endpoint='InvertedResidual_{}_{}'.format(2048, 2),
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
  
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  with tf.variable_scope(scope, 'ResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
    
      current_stride = 1

      rate = 1

      net = inputs
      for i, conv_def in enumerate(conv_defs):
        if output_stride is not None and current_stride == output_stride:
          layer_stride = 1
          layer_rate = rate
          rate *= conv_def.stride
        else:
          layer_stride = conv_def.stride
          layer_rate = 1
          current_stride *= conv_def.stride

        if isinstance(conv_def, Conv):
          end_point = 'Conv2d_%d' % i
          net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point)
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          end_points[end_point] = net
          

        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            end_point = 'InvertedResidual_{}_{}'.format(conv_def.depth, n)
            stride = conv_def.stride if n == 0 else 1
            net = _inverted_residual_bottleneck(net, depth(conv_def.depth), stride, conv_def.t, lambda_decay, scope=end_point)
            lambda_b = slim.model_variable('InvertedResidual_{}_{}/lambda_b'.format(conv_def.depth, n), shape=[1, 1, 1, 1],
                       initializer=tf.ones_initializer(),
                       regularizer=slim.l1_regularizer(lambda_decay)
                       )
            lambda_b = soft_thresholding(lambda_b, lambda_decay)
            net = tf.multiply(net, tf.tile(lambda_b, 
                                 [net.get_shape().as_list()[0], net.get_shape().as_list()[1], net.get_shape().as_list()[2], 1]))
            end_points[end_point] = net
            if end_point == final_endpoint:
              return net, end_points
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.type, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def resnet_v2(inputs,
                 lambda_decay,
                 num_classes=1000,                 
                 dropout_keep_prob=0.997,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,               
                 reuse=None,
                 scope='ResnetV2'
                 ): 
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'ResnetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = resnet_v2_base(inputs, lambda_decay,
                                          min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,                                         
                                          conv_defs=conv_defs, scope=scope)
      # with tf.variable_scope('Logits'):
      #   if global_pool:
      #     # Global average pooling.
      #     net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
      #     end_points['global_pool'] = net
      #   else:
      #     # Pooling with a fixed kernel size.
      #     kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
      #     net = slim.avg_pool2d(net, kernel_size, padding='VALID',
      #                           scope='AvgPool_1a')
      #     end_points['AvgPool_1a'] = net
      #   if not num_classes:
      #     return net, end_points
      #   # 1 x 1 x 1024
      #   net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
      #   logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
      #                        normalizer_fn=None, scope='Conv2d_1c_1x1')
      #   if spatial_squeeze:
      #     logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      # end_points['Logits'] = logits
      # if prediction_fn:
      #   end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return net, end_points

resnet_v2.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


resnet_v2_075 = wrapped_partial(resnet_v2, depth_multiplier=0.75)
resnet_v2_050 = wrapped_partial(resnet_v2, depth_multiplier=0.50)
resnet_v2_025 = wrapped_partial(resnet_v2, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def resnet_v2_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           stddev=0.09,
                           regularize_depthwise=False):
  """Defines the default  arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.

  Returns:
    An `arg_scope` to use for the  model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': 0.997,
      'epsilon': 0.001,
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc

def soft_thresholding(input, thr):
    zero_tensor = tf.zeros(input.get_shape().as_list(), input.dtype.base_dtype)
    one_tensor = tf.ones(input.get_shape().as_list(), input.dtype.base_dtype)
    return tf.sign(input) * tf.maximum(zero_tensor, tf.abs(input) - thr * one_tensor)
