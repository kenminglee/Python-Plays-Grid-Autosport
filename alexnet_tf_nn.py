import tensorflow as tf
from decimal import Decimal, ROUND_HALF_UP

STRIDES = [1, 1, 1, 1]

def alexnet_tf_nn(image_width, image_height, input_placeholder, n_classes):


    weights = tf.Variable(tf.random_normal([11, 11, 1, 96]))
    biases = tf.Variable(tf.random_normal([96]))
    network = tf.nn.conv2d(input_placeholder, weights, strides=[1, 4, 4, 1], padding='SAME')
    network = tf.nn.relu(network + biases)
    network = tf.nn.max_pool(network, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    network = tf.nn.local_response_normalization(network)

    weights = tf.Variable(tf.random_normal([5, 5, 96, 256]))
    biases = tf.Variable(tf.random_normal([256]))
    network = tf.nn.conv2d(network, weights, strides=STRIDES, padding='SAME')
    network = tf.nn.relu(network + biases)
    network = tf.nn.max_pool(network, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    network = tf.nn.local_response_normalization(network)

    weights = tf.Variable(tf.random_normal([3, 3, 256, 384]))
    biases = tf.Variable(tf.random_normal([384]))
    network = tf.nn.conv2d(network, weights, strides=STRIDES, padding='SAME')
    network = tf.nn.relu(network + biases)

    weights = tf.Variable(tf.random_normal([3, 3, 384, 384]))
    network = tf.nn.conv2d(network, weights, strides=STRIDES, padding='SAME')
    network = tf.nn.relu(network + biases)

    weights = tf.Variable(tf.random_normal([3, 3, 384, 256]))
    biases = tf.Variable(tf.random_normal([256]))
    network = tf.nn.conv2d(network, weights, strides=STRIDES, padding='SAME')
    network = tf.nn.relu(network + biases)
    network = tf.nn.max_pool(network, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    network = tf.nn.local_response_normalization(network)

    new_height = int(Decimal(image_height/32).quantize(0, ROUND_HALF_UP))
    new_width = int(Decimal(image_width/32).quantize(0, ROUND_HALF_UP))
    flatten_dim = new_height*new_width*256
    network = tf.reshape(network, [-1, flatten_dim])

    weights = tf.Variable(tf.random_normal([flatten_dim, 4096]))
    biases = tf.Variable(tf.random_normal([4096]))
    network = tf.nn.tanh(tf.matmul(network, weights) + biases)
    network = tf.nn.dropout(network, 0.5)

    weights = tf.Variable(tf.random_normal([4096, 4096]))
    network = tf.nn.tanh(tf.matmul(network, weights) + biases)
    network = tf.nn.dropout(network, 0.5)

    weights = tf.Variable(tf.random_normal([4096, 3]))
    biases = tf.Variable(tf.random_normal([3]))
    raw_output = tf.add(tf.matmul(network, weights), biases, name="op_to_restore")

    return raw_output















