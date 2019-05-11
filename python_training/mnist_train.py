import numpy as np
import sys
import os
# Suppress TF messages and Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)


EPOCH_NUMBER = 2

NETWORK_NAME = str(sys.argv[1])

layer_num = 4
n_input = 0
n_hidden1 = 0
n_hidden2 = 0
n_hidden3 = 0
n_output = 0


def read_network_conf(network_name):
    conf_file_name = network_name
    global n_input, n_hidden1, n_hidden2, n_hidden3, n_output
    conf_file = open("../"+conf_file_name)
    for _ in range(5): conf_file.readline()
    n_input = int(conf_file.readline().split()[0])
    n_hidden1 = int(conf_file.readline().split()[0])
    n_hidden2 = int(conf_file.readline().split()[0])
    n_hidden3 = int(conf_file.readline().split()[0])
    n_output = int(conf_file.readline().split()[0])


def print_conf():
    print("\n\n**********************************************************\n")
    print("\t	" + str(NETWORK_NAME) + "  is going to be trained.")
    print()
    print("\t	Input : " + str(n_input))
    print("\t	Hidden 1 : " + str(n_hidden1))
    print("\t	Hidden 2 : " + str(n_hidden2))
    print("\t	Hidden 3 : " + str(n_hidden3))
    print("\t	Output : " + str(n_output) + "\n")


def save_2d_array_c(w_array, name, w_type="ab"):
    file = open("../parameters/"+NETWORK_NAME[0:-5]+"_weights.bin", w_type)
    for i in range(w_array.shape[1]):
        tmp = w_array[:, i]
        for j in range(w_array.shape[0]):
            #file.writelines(bytearray(tmp[j]))
            tmp.tofile(file)
    file.close()


def save_c_weights():
    print("\nSaved weights and biases to: parameters/"+
            NETWORK_NAME[0:-5] +"_weights.bin\n")
    save_2d_array_c(weights["w1"].eval(), "w1", w_type="wb")
    save_2d_array_c(weights["w2"].eval(), "w2")
    save_2d_array_c(weights["w3"].eval(), "w3")
    save_2d_array_c(weights["wout"].eval(), "wout")

    # save_1d_array_c(biases["b1"].eval(), "b1")
    # save_1d_array_c(biases["b2"].eval(), "b2")
    # save_1d_array_c(biases["b3"].eval(), "b3")
    # save_1d_array_c(biases["bout"].eval(), "bout")


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

read_network_conf(NETWORK_NAME)

with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    n_train = mnist.train.num_examples
    n_validation = mnist.validation.num_examples
    n_test = mnist.test.num_examples

    learning_rate = 1e-4
    n_iterations = 1000
    batch_size = 128
    dropout = 0.5

    # building the TensorFlow Graph

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    keep_prob = tf.placeholder(tf.float32)

    weights = {
        'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
        'wout': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
    }
    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
        'bout': tf.Variable(tf.constant(0.1, shape=[n_output]))
    }

    layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))
    layer_drop = tf.nn.dropout(layer_3, keep_prob)
    output_layer = tf.nn.softmax(tf.matmul(layer_3, weights['wout']) + biases['bout'])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=output_layer
        ))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess.run(init)

    print_conf()
    print("For " + str(EPOCH_NUMBER) + " epoch.\n")
    # Training
    # train on mini batches
    for epoch in range(EPOCH_NUMBER):
        for i in range(n_iterations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={
                X: batch_x, Y: batch_y, keep_prob: dropout
            })

            # print loss and accuracy (per minibatch)
            if i % 100 == 0:
                minibatch_loss, minibatch_accuracy = sess.run(
                    [cross_entropy, accuracy],
                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
                )

        test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
        print("Accuracy on test set:", test_accuracy)

    save_c_weights()
