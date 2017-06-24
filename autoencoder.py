import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import saver_pb2
from sklearn.metrics import r2_score
from functools import partial
from helper_funcs import reset_graph, group_list
from data_prep import DataPrep


#Prepping data
prep = Dataprep(dummy_pipe=True)
X_train,_,X_test,_,test,_ = prep.load_data('data/train.csv', 'data/test.csv')
X_train, X_test ,_ = prep.transform(X_train, X_test, test)
#concatenating all training data for training:
X_train = np.concatenate((X_train, X_test), axis=0)

n_epochs = 100
batch_size = 300

#initializing variables
reset_graph()
n_inputs = X_train.shape[1]
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
he_init = tf.contrib.layers.variance_scaling_initializer() #He initialization
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer = he_init, kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        X_gen = group_list(X_train, batch_size)
        n_batches = X_train.shape[0] // batch_size
        try:
            for iteration in range(n_batches):
                X_batch = next(X_gen)
                sess.run(training_op, feed_dict={X:X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X:X_batch})
        except StopIteration:
            pass
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "models/my_model_all_layers.ckpt")

