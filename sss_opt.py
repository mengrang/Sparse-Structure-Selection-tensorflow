# coding:uft-8
import tensorflow as tf

accum_list = tf.zeros_like()
def train(loss_val, net_list, fc_list, lambda_list, lr, gamma, clip_value): 
    opt1 = tf.train.GradientDescentOptimizer
    opt2 = tf.train.MomentumOptimizer
 
    fc_optimizer = opt2(learning_rate=lr, momentum=0.9, use_nesterov=True)
    net_optimizer = opt2(learning_rate=lr * 0.01, momentum=0.9, use_nesterov=True)
    lambda_optimizer = opt1(learning_rate=1.)
    
    lambda_grads = tf.gradients(loss_val, lambda_list)
    net_grads = tf.gradients(loss_val, net_list)
    fc_grads = tf.gradients(loss_val, fc_list)
    # lambda optimator
    clipped_lambda_grads = tf.clip_by_value(lambda_grads, -clip_value, clip_value)
    accum_list = [(0.9 * accum + lr * grad) for accum, grad \
                 in zip(accum_list, clipped_lambda_grads)]

    z_list = [(_lambda - 0.9 * accum) for _lambda, accum \
             in zip(lambda_list, accum_list)]       # equ 10

    z_list = [soft_thresholding(z, lr * gamma) for z in z_list]

    accum_list = [(-z + _lambda - accum) for z, _lambda, accum \
                in zip(z_list, lambda_list, accum_list)]    # equ 11

    lambda_accums_vars = [(0.9 * accum, var) for accum, var in zip(accum_list, lambda_list) \
                            if accum is not None]

    clipped_net_grads_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var \
                            in zip(net_grads, net_list) if grad is not None]
    clipped_fc_grads_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var \
                            in zip(fc_grads, fc_list) if grad is not None]
    if FLAGS.debug:
        for grad, var in clipped_fc_grads_vars:
            tf.summary.histogram(var.op.name + "/gradient", grad)
        for grad, var in clipped_net_grads_vars:
            tf.summary.histogram(var.op.name + "/gradient", grad)
        for accum, var in lambda_accums_vars:
            tf.summary.histogram(var.op.name + "/accum", accum)
    train_fc = fc_optimizer.apply_gradients(clipped_fc_grads)
    train_net = net_optimizer.apply_gradients(clipped_net_grads)
    train_lambda = lambda_optimizer.apply_gradients(lambda_accums_vars)
    train_op = tf.group(train_fc, train_net, train_lambda)
    
    return train_op

def soft_thresholding(input, thr):
    zero_tensor = tf.zeros(input.get_shape().as_list(), input.dtype.base_dtype)
    one_tensor = tf.ones(input.get_shape().as_list(), input.dtype.base_dtype)
    return tf.sign(input) * tf.maximum(zero_tensor, tf.abs(input) - thr * one_tensor)

    