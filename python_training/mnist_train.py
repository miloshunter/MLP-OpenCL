import numpy as np
import sys
import os
# Suppress TF messages and Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)

NETWORK_NAME = str(sys.argv[1])
EPOCH_NUMBER = int(sys.argv[2])


layer_num = 0
layer_sizes = []


def read_network_conf(network_name):
    conf_file_name = network_name
    global layer_num, layer_sizes
    conf_file = open("../"+conf_file_name)
    if conf_file is None:
        print("Cannot read file!")
        exit()
    for _ in range(3): conf_file.readline()
    layer_num = int(conf_file.readline().split()[0])
    conf_file.readline()
    for _ in range(layer_num+1):
        layer_sizes.append(int(conf_file.readline().split()[0]))


def print_conf():
    print("\n\n**********************************************************\n")
    print("\t	" + str(NETWORK_NAME) + "  is going to be trained.")
    print()
    print("\t	Input : " + str(layer_sizes[0]))
    for i in range(layer_num):
        print("\t	Layer %d : %s", i, str(layer_sizes[i+1]))


def save_2d_array(w_array, name, w_type="ab"):
    file = open("../parameters/"+NETWORK_NAME[0:-5]+"_weights.bin", w_type)
    for i in range(w_array.shape[1]):
        tmp = w_array[:, i]
        for j in range(w_array.shape[0]):
            tmp[j].tofile(file)
    file.close()


def save_1d_array(w_array, name, w_type="ab"):
    file = open("../parameters/"+NETWORK_NAME[0:-5]+"_weights.bin", w_type)
    for i in range(w_array.shape[0]):
        tmp = w_array[i]
        tmp.tofile(file)
        #print(tmp)
    #print()
    file.close()


def save_parameters_binary():
    print("\nSaved weights and biases to: parameters/"+
            NETWORK_NAME[0:-5] +"_weights.bin\n")
    save_2d_array(weights[0].eval(), "w", w_type="wb")
    for i in range(1, layer_num):
        save_2d_array(weights[i].eval(), "w2")
    for i in range(0, layer_num):
        save_1d_array(biases[i].eval(), "b1")



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

    X = tf.placeholder("float", [None, layer_sizes[0]])
    Y = tf.placeholder("float", [None, layer_sizes[layer_num]])
    keep_prob = tf.placeholder(tf.float32)

    weights = []
    biases = []
    for n in range(layer_num):
        weights.append(tf.Variable(tf.truncated_normal([layer_sizes[n], layer_sizes[n+1]], stddev=0.1)))
        biases.append(tf.Variable(tf.constant(0.1, shape=[layer_sizes[n+1]])))
    print("Number of layers: " + str(layer_num))
    print("Layer 1 Size " + str(layer_sizes[0+1]))
    tmp = tf.nn.relu(tf.add(tf.matmul(X, weights[0]), biases[0]))
    for n in range(1, layer_num-1):
        print("Layer " + str(n+1) + " Size " + str(layer_sizes[n+1]))
        tmp = tf.nn.relu(tf.add(tf.matmul(tmp, weights[n]), biases[n]))
    layer_drop = tf.nn.dropout(tmp, keep_prob)
    print("Output " + str(layer_num-1) + " Size " + str(layer_sizes[layer_num]))
    output_layer = tf.nn.softmax(tf.matmul(tmp, weights[layer_num-1]) + biases[layer_num-1])

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
        print("Accuracy on MNIST test set:", test_accuracy)

    save_parameters_binary()
