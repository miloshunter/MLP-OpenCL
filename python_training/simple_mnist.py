import tensorflow as tf
import numpy as np
from skimage import io
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def save_2d_array_c(array, name):
    file = open("exported_c_files/"+name+".h", "w")
    file.writelines("//Testiranje ispisa\n")
    file.writelines("//Niz je: " + str(array.shape) + "\n")
    file.writelines("extern double "+name+"["+
                    str(array.shape[1])+"]["+str(array.shape[0])+
                    "];\n")
    file.close()

    file = open("exported_c_files/"+name+".c", "w")
    file.writelines("double "+name+"[" +
                    str(array.shape[1]) + "][" + str(array.shape[0]) +
                    "] = {\n")

    for i in range(array.shape[1]):
        file.writelines("{ ")
        tmp = array[:, i]
        for j in range(array.shape[0]):
            file.writelines(str(tmp[j]))
            if j < array.shape[0] - 1:
                file.writelines(",")
        file.writelines("}")
        if i < array.shape[1]-1:
            file.writelines(",")
        file.writelines("\n")
    file.writelines("};\n")
    file.close()


def save_c_weights():
    save_2d_array_c(weights["w1"].eval(), "w1")
    save_2d_array_c(weights["w2"].eval(), "w2")
    save_2d_array_c(weights["w3"].eval(), "w3")
    save_2d_array_c(weights["wout"].eval(), "wout")

    save_1d_array_c(biases["b1"].eval(), "b1")
    save_1d_array_c(biases["b2"].eval(), "b2")
    save_1d_array_c(biases["b3"].eval(), "b3")
    save_1d_array_c(biases["bout"].eval(), "bout")

def save_c_layers():
    save_1d_array_c(sess.run(layer_1, feed_dict={X: [y]})[0], "layer1")
    save_1d_array_c(sess.run(layer_2, feed_dict={X: [y]})[0], "layer2")
    save_1d_array_c(sess.run(layer_3, feed_dict={X: [y]})[0], "layer3")
    save_1d_array_c(sess.run(output_layer, feed_dict={X: [y]})[0], "output")

def save_1d_array_c(array, name):
    file = open("exported_c_files/"+name+".h", "w")
    file.writelines("//Testiranje ispisa\n")
    file.writelines("//Niz je: " + str(array.shape) + "\n")
    file.writelines("extern double "+name+"["+
                    str(array.shape[0])+
                    "];\n")
    file.close()
    file = open("exported_c_files/"+name+".c", "w")
    file.writelines("double " + name + "[" +
                    str(array.shape[0]) +
                    "] = {\n")
    for j in range(array.shape[0]):
        file.writelines(str(array[j]))
        if j < array.shape[0] - 1:
            file.writelines(",")
    file.writelines("};\n")
    file.close()


image_name = 'sedmica'
im = io.imread('test_pics/'+image_name+'.png', as_gray=True)
im = 1 - im


with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    n_train = mnist.train.num_examples
    n_validation = mnist.validation.num_examples
    n_test = mnist.test.num_examples

    # define neural network config
    layer_num = 4

    n_input = 784
    n_hidden1 = 4096
    n_hidden2 = 2048
    n_hidden3 = 2048
    n_output = 10

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

    # Training
    # train on mini batches
    for epoch in range(1):
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

    ulaz = np.array(im)
    y = ulaz.reshape(784)
    save_1d_array_c(y, image_name)

    save_c_weights()
    save_c_layers()
    t0 = time.time()
    # Create the Timeline object, and write it to a json
    prediction = sess.run(output_layer, feed_dict={X: [y]}, options=run_options, run_metadata=run_metadata)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    t1 = time.time()

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print("Time elapsed: ", (t1 - t0))
    print("Prediction for test image:", prediction)




