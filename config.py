import tensorflow as tf
"""
" Log Configuration
"""
tf.app.flags.DEFINE_string(name="data_dir", default="./datasets", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="t_s_logs_dir", default="./logs_sss/", help="The directory to the model checkpoint, tensorboard and log.")

tf.app.flags.DEFINE_string(name="teacher_dir", default="./logs_sss/teacher", help="The directory to the pre-trained  teachernet weights.")

tf.app.flags.DEFINE_integer(name="batch_size", default=96, help="The number of samples in each batch.")

tf.app.flags.DEFINE_integer(name="num_class", default=12, help="The number of classes.")

tf.app.flags.DEFINE_integer(name="epoches", default=1000, help="The number of training epoch.") 

tf.app.flags.DEFINE_integer(name="verbose", default=4, help="The number of training step to show the loss and accuracy.")

tf.app.flags.DEFINE_integer(name="patience", default=3, help="The patience of the early stop.")

tf.app.flags.DEFINE_boolean(name="debug", default=True, help="Debug mode or not")

tf.app.flags.DEFINE_float(name="alpha",default=0.5, help="The weight of knowledge distillation ")

tf.app.flags.DEFINE_float(name="lambda_decay",default=5e-5, help="lambda_decay ")

tf.app.flags.DEFINE_float(name="beta", default=400.0, help="The weight of knowledge distillation ")

tf.app.flags.DEFINE_integer(name="T", default=2, help="The temprature of knowledge distillation ")

tf.app.flags.DEFINE_boolean(name="mimic",default=False, help="MImic mode or not")

tf.flags.DEFINE_string(name="mode", default="train", help="Mode train/ test/ visualize")

FLAGS = tf.app.flags.FLAGS