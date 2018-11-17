# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
import resnet

import os
import h5py
import time
import sys
from tflearn.data_utils import shuffle
import numpy as np
from time import time
from config import FLAGS


M = FLAGS.num_class
k = 10
"""
""TEACHERNET
"""
def teacher(input_images, 
            keep_prob,
            lambda_decay=FLAGS.lambda_decay,
            is_training=True,
            weight_decay=0.00004,
            batch_norm_decay=0.99,
            batch_norm_epsilon=0.001):
    with tf.variable_scope("Teacher_model"):  
        net, endpoints = resnet.resnet_v2(inputs=input_images,
                                lambda_decay=lambda_decay,
                                num_classes=FLAGS.num_class,
                                is_training=True,                               
                                scope='resnet_v2_50')   
        # co_trained layers
        var_scope = 'Teacher_model/resnet_v2_50/'
        co_list_0 = slim.get_model_variables(var_scope + 'Conv2d_0')
        # co_list_1 = slim.get_model_variables(var_scope +'InvertedResidual_16_')
        # co_list_2 = slim.get_model_variables(var_scope +'InvertedResidual_24_')
        t_co_list = co_list_0 
        
        base_var_list = slim.get_variables()
        # for _ in range(2):
        #      base_var_list.pop()
        lambda_c_list = slim.get_variables_by_name('lambda_c')       
        lambda_b_list = slim.get_variables_by_name('lambda_b')
        t_lambda_list = lambda_c_list + lambda_b_list
        # print(lambda_b_list)
        # exit()
        t_net_var_list =[]
        for v in base_var_list:
            if v not in t_co_list and v not in t_lambda_list:
                t_net_var_list.append(v)
        # feature & attention
        t_g0 = endpoints["InvertedResidual_{}_{}".format(256, 2)]
        t_at0 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g0), -1), axis=0, name='t_at0')
        t_g1 = endpoints["InvertedResidual_{}_{}".format(512, 3)]
        t_at1 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g1), -1), axis=0, name='t_at1')
        part_feature = endpoints["InvertedResidual_{}_{}".format(1024, 5)]
        t_at2 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(part_feature), -1), axis=0, name='t_at2')
        object_feature = endpoints["InvertedResidual_{}_{}".format(2048, 2)]
        t_at3 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(object_feature), -1), axis=0, name='t_at3')
        # print(t_at1.get_shape().as_list())
        # exit()
        t_g = (t_g0, t_g1, part_feature, object_feature)
        t_at = (t_at0, t_at1, t_at2, t_at3)
        
        fc_obj = slim.max_pool2d(object_feature, (6, 8), scope="GMP1")
        batch_norm_params = {
            'center': True,
            'scale': True,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }
        
        fc_obj = slim.conv2d(fc_obj,
                            M,
                            [1, 1],
                            activation_fn=None,    
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope='fc_obj')
        fc_obj = tf.nn.dropout(fc_obj, keep_prob=keep_prob)
        fc_obj = slim.flatten(fc_obj)
        # 
        fc_part = slim.conv2d(part_feature,
                            M * k,          #卷积核个数
                            [1, 1],         #卷积核高宽
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,                               # 标准化器设置为BN
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                            )
        # print('part',fc_part.get_shape())
        fc_part = slim.max_pool2d(fc_part, (12, 16), scope="GMP2")
        ft_list = tf.split(fc_part,
                        num_or_size_splits=FLAGS.num_class,
                        axis=-1)            #最后一维度（C）
        
        cls_list = []
        for i in range(M):
            ft = tf.transpose(ft_list[i], [0, 1, 3, 2])
            cls = layers_lib.pool(ft,
                                [1, 10],
                                "AVG")
            cls = layers.flatten(cls)
            cls_list.append(cls)
        fc_ccp = tf.concat(cls_list, axis=-1) #cross_channel_pooling (N, M)

        fc_part = slim.conv2d(fc_part,
                            FLAGS.num_class,
                            [1, 1],
                            activation_fn=None,                         
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope="fc_part")
        fc_part = tf.nn.dropout(fc_part, keep_prob=keep_prob)
        fc_part = slim.flatten(fc_part)
        
        t_var_list = slim.get_model_variables()
        t_fc_var_list = []
        for var in t_var_list:
            if var not in base_var_list:
                t_fc_var_list.append(var)
                
    return t_g, t_at, fc_obj, fc_part, fc_ccp, t_co_list, t_net_var_list, t_fc_var_list, t_lambda_list, t_var_list
    
def train(loss_val, net_list, fc_list, lambda_list, lr, clip_value): 
    opt = tf.train.MomentumOptimizer
    # fc_optimizer = opt(learning_rate=lr, momentum=0.9, use_nesterov=True)
    # net_optimizer = opt(learning_rate=lr * 0.01, momentum=0.9, use_nesterov=True)
    # lambda_optimizer = opt(learning_rate=lr, momentum=0.9, use_nesterov=True)

    # tower_lambda_grads = []
    # tower_net_grads = []
    # tower_fc_grads = []
    
    # for i in [1, 7]：
    #     with tf.debice('/gpu:%d'%i)
    #         with tf.name_scope('GPU_%d'%i) as scope:
    #             #tf.get_variable的命名空间
    #             tf.get_variable_scope().reuse_variables()
    #             #使用当前gpu计算所有变量的梯度
    #             lambda_grads = tf.gradients(loss_val, lambda_list)  
    #             net_grads = tf.gradients(loss_val, net_list)
    #             fc_grads = tf.gradients(loss_val, fc_list)

    #             clipped_net_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(net_grads, net_list) \
    #                             if grad is not None]
    #             clipped_fc_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(fc_grads, fc_list) \
    #                                     if grad is not None]
    #             clipped_lambda_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(lambda_grads, lambda_list) \
    #                                     if grad is not None]
    #             tower_lambda_grads.append(lambda_grads)
    #             tower_net_grads.append(net_grads)
    #             tower_fc_grads.append(fc_grads)
    # #计算变量的平均梯度
    # lambda_grads = average_gradients(tower_lambda_grads)
    # net_grads = average_gradients(tower_net_grads)
    # fc_grads = average_gradients(tower_fc_grads)
    # #使用平均梯度更新参数
    # apply_gradient_op = opt.apply_gradients(grads,global_step = global)
 

    fc_optimizer = opt(learning_rate=lr, momentum=0.9, use_nesterov=True)
    net_optimizer = opt(learning_rate=lr * 0.01, momentum=0.9, use_nesterov=True)
    lambda_optimizer = opt(learning_rate=lr, momentum=0.9, use_nesterov=True)
    
    lambda_grads = tf.gradients(loss_val, lambda_list)
    net_grads = tf.gradients(loss_val, net_list)
    fc_grads = tf.gradients(loss_val, fc_list)

    clipped_net_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(net_grads, net_list) \
                                if grad is not None]
    clipped_fc_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(fc_grads, fc_list) \
                            if grad is not None]
    clipped_lambda_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(lambda_grads, lambda_list) \
                            if grad is not None]

    if FLAGS.debug:
        for grad, var in clipped_fc_grads:
            tf.summary.histogram(var.op.name + "/gradient", grad)
        for grad, var in clipped_net_grads:
            tf.summary.histogram(var.op.name + "/gradient", grad)
        for grad, var in clipped_lambda_grads:
            tf.summary.histogram(var.op.name + "/gradient", grad)
    train_fc = fc_optimizer.apply_gradients(clipped_fc_grads)
    train_net = net_optimizer.apply_gradients(clipped_net_grads)
    train_lambda = lambda_optimizer.apply_gradients(clipped_lambda_grads)
    train_op = tf.group(train_fc, train_net, train_lambda)
    
    return train_op

def accuracy_top1(y_true, predictions):
    acc_top1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(predictions, axis=-1)), tf.float32), axis=-1)
    return acc_top1

def accuracy_top5(y_true, predictions):
    acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(y_true, axis=-1), k=5), tf.float32), axis=-1)
    return acc_top5         


def main(argv=None):
    is_training = True
    input_images = tf.placeholder(dtype=tf.float32,
                                      shape=[FLAGS.batch_size, 192, 256, 3],
                                      name="input_images")
    y_true = tf.placeholder(dtype=tf.float32,
                            shape=[FLAGS.batch_size, FLAGS.num_class],
                            name="y_true")
    keep_prob = tf.placeholder(dtype=tf.float32,
                                name="dropout")
    learning_rate = tf.placeholder(dtype=tf.float64,
                                    name="learning_rate")
    clip_value = tf.placeholder(dtype=tf.float32,
                                name="clip_value")
        
    """
    ""inference
    """
    t_g, t_at, fc_obj, fc_part, fc_ccp, t_co_list, t_net_var_list, t_fc_var_list, t_lambda_list, t_var_list= teacher(input_images, keep_prob, is_training=is_training)
    
    fc_part = tf.nn.softmax(fc_part)
    fc_ccp = tf.nn.softmax(fc_ccp)
    fc_obj = tf.nn.softmax(fc_obj)
    t_predictions = (fc_part + 0.1 * fc_ccp + fc_obj) / 3.0
    
    """
    ""teachernet loss
    """
    if not FLAGS.mimic:
        obj_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_obj))
        part_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_part))
        ccp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_ccp))
        loss = 0.1 * ccp_loss + part_loss + obj_loss
        acc_top1 = accuracy_top1(y_true, t_predictions)
        acc_top5 = accuracy_top5(y_true, t_predictions)
        """
        ""Summary
        """
        tf.summary.scalar("t_obj_loss", obj_loss)
        tf.summary.scalar("t_part_loss", part_loss)
        tf.summary.scalar("t_ccp_loss", ccp_loss)
        tf.summary.scalar("t_loss", loss)
        tf.summary.scalar("acc_top1", acc_top1)
        tf.summary.scalar("acc_top5", acc_top5)

        
        train_op = train(loss, t_net_var_list, t_fc_var_list, t_lambda_list, learning_rate, clip_value)                
        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()

    
    """
    " Loading Data
    """
    print("Loading Data......")
    if FLAGS.mode == 'train':
        with h5py.File(os.path.join(FLAGS.data_dir, "trainset.h5"), "r") as f:
            X_train = f["X"][:]
            Y_train = f["Y"][:]
        
            print(X_train.shape)
            print(Y_train.shape)
            f.close()
            print("\tLoaded Train Data......")
    with h5py.File(os.path.join(FLAGS.data_dir, "testset.h5"), "r") as f:
        X_test = f["X"][:]
        Y_test = f["Y"][:]
        
        print(X_test.shape)
        print(Y_test.shape)
        f.close()
        print("\tLoaded Test Data......")  
    print("Verifying the data......")
    # for i in range(100):
    #     if i*200 > 6000:
    #         break
    #     train_img = X_train[i*100]
    #     test_img = X_test[i*200]
    #     cv2.imshow("Train", np.uint8(train_img))
    #     cv2.imshow("Test", np.uint8(test_img))
    #
    #     print(i, "Train class id", np.argmax(Y_train[i*100], axis=0))
    #     print(i, "Test class id", np.argmax(Y_test[i*200], axis=0))
    #     cv2.waitKey()
    # exit()
    
    """
    " Setting up Saver
    """
    
    print("Setting up Saver...")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    
    saver_t = tf.train.Saver(t_var_list, max_to_keep=3)
    saver_t_co = tf.train.Saver(t_co_list, max_to_keep=3)
    saver_t_lambda = tf.train.Saver(t_lambda_list, max_to_keep=3)

    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.t_s_logs_dir, 'train'),
                                        sess.graph)
    #改路径 FLAGS.train_dir
    valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.t_s_logs_dir, 'valid'),
                                        sess.graph)
   

    print("Initialize global variables")                                   
    sess.run(tf.global_variables_initializer())

    """
    " Resume
    """
    ckpt_t = tf.train.get_checkpoint_state(FLAGS.t_s_logs_dir)
    if ckpt_t and ckpt_t.model_checkpoint_path:
        print('teacher:', ckpt_t.model_checkpoint_path)
        saver_t.restore(sess, ckpt_t.model_checkpoint_path)
        saver_t_co.restore(sess, ckpt_t.model_checkpoint_path)
        saver_t_lambda.restore(sess, ckpt_t.model_checkpoint_path)
        print("Model restored...")  

    """
    " Training...
    """
    if FLAGS.mode == 'train':      
        train_batch = int(X_train.shape[0] / FLAGS.batch_size)
        valid_batch = int(X_test.shape[0] / FLAGS.batch_size)
        last_loss = 10000.
        patience = 0
        best_acc = 0.0
        clipvalue = 1e-3
        global_step = tf.train.get_or_create_global_step()
        epoch_st = global_step // train_batch + 1

        current = 1e-3
        for epoch in range(40, FLAGS.epoches if FLAGS.debug else 1):
            print("Epoch %i ----> Starting......" % epoch)
            X_train, Y_train = shuffle(X_train, Y_train)
            start_time = time()
            
            """
            " Build learning rate
            """
            if epoch <= 20:
                lr = 1e-3 / 20.0 * epoch
                current = lr
            elif epoch > 20 and epoch < 30:
                lr = 1e-3
                current = lr
            else:
                lr = current
                       
            for step in range(train_batch):
                batch_x = X_train[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                batch_y = Y_train[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                summary, _ = sess.run([summary_op, train_op],
                                    feed_dict={input_images: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 0.3,
                                                learning_rate: lr,
                                                clip_value: clipvalue})
                train_writer.add_summary(summary, step + train_batch * (epoch-1))
                """
                ' print the train loss
                """
                if (epoch * train_batch + step) % FLAGS.verbose == 0:
                    if not FLAGS.mimic:
                        loss_t, loss_ct, loss_ot, loss_pt, acc_1t, acc_5t = \
                            sess.run([loss, ccp_loss, obj_loss, part_loss, acc_top1, acc_top5],
                                    feed_dict={input_images: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 0.3,
                                                learning_rate: lr,
                                                clip_value: clipvalue})
                        print("Step %i, Train_loss \33[91m%.4f\033[0m, ccp_loss %0.4f, obj_loss %0.4f, part_loss %0.4f, acc_1 \33[91m%.4f\033[0m, acc_5 %0.4f, lr: %.7f, clipvalue: %.7f" %
                                    ((epoch-1) * train_batch + step, loss_t, loss_ct, loss_ot, loss_pt, acc_1t, acc_5t, lr, clipvalue))
                    
            acc1_reg = []
            acc5_reg = []
            loss_reg = []
            for step in range(valid_batch):
                batch_x = X_test[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                batch_y = Y_test[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                loss_v, acc_1v, acc_5v, summary = sess.run([loss, acc_top1, acc_top5, summary_op],
                                                        feed_dict={input_images: batch_x,
                                                                    y_true: batch_y,
                                                                    keep_prob: 1.,
                                                                    learning_rate: lr,
                                                                    clip_value: clipvalue})
                valid_writer.add_summary(summary, step + valid_batch * (epoch-1))
                acc1_reg.append(acc_1v)
                acc5_reg.append(acc_5v)
                loss_reg.append(loss_v)
            avg_acc1 = np.mean(np.array(acc1_reg))
            avg_acc5 = np.mean(np.array(acc5_reg))
            avg_loss = np.mean(np.array(loss_reg))
            print("Valid_loss ----> %0.4f Valid_acc ----> %0.4f, %0.4f" % (avg_loss, avg_acc1, avg_acc5))
            """
            " Save the best model
            """
            if avg_acc1 > best_acc:
                best_acc = avg_acc1
                saver_t.save(sess=sess,
                        save_path=FLAGS.t_s_logs_dir,
                        global_step=epoch)
                print("Save the best model with val_acc %0.4f" % best_acc)
            else:
                print("Val_acc stay with val_acc %0.4f" % best_acc)

            if last_loss - avg_loss > 1e-5 and avg_loss - last_loss < 1e-5:
                last_loss = avg_loss
                patience = 0
                print("Patience %i with updated val_loss %0.4f" % (patience, last_loss))
            else:
                patience = patience + 1
                print("Patience %i with stayed val_loss %0.4f" % (patience, last_loss))

            if patience >= FLAGS.patience:
                patience = 0
                last_loss = 10000
                current = current * 0.5
                clipvalue = clipvalue * 0.1
                print("Lr decay, update the learning rate when lr = %0.4f" % lr)
            end_time = time()
            print("Epoch %i ----> Ended in %0.4f" % (epoch, end_time - start_time))
            train_writer.close()
            valid_writer.close()
        print("......Ended")

        print("Ending......")
    if FLAGS.mode =='test':
        acc1_reg = []
        acc5_reg = []
        loss_reg = []
        for step in range(valid_batch):
            batch_x = X_test[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
            batch_y = Y_test[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
            loss_v, acc_1v, acc_5v, summary = sess.run([loss, acc_top1, acc_top5, summary_op],
                                                    feed_dict={input_images: batch_x,
                                                                y_true: batch_y,
                                                                keep_prob: 1.,
                                                                learning_rate: lr,
                                                                clip_value: clipvalue})
            valid_writer.add_summary(summary, step + valid_batch * (epoch-1))
            acc1_reg.append(acc_1v)
            acc5_reg.append(acc_5v)
            loss_reg.append(loss_v)
        avg_acc1 = np.mean(np.array(acc1_reg))
        avg_acc5 = np.mean(np.array(acc5_reg))
        avg_loss = np.mean(np.array(loss_reg))
        print("Valid_loss ----> %0.4f Valid_acc ----> %0.4f, %0.4f" % (avg_loss, avg_acc1, avg_acc5))
if __name__ == "__main__":
    tf.app.run()