# Run this to create a model from
import numpy as np
from alexnet_tf_nn import alexnet_tf_nn
import tensorflow as tf
import cv2

#HUGE PROBLEM: we are getting images in the form of width*height, but tf only accepts height*width - therefore all of
# our images are squished when it is reshaped - therefore it is far less effective than tflearn - which uses width*height
WIDTH = 80
HEIGHT = 60
# LR = 1e-3
EPOCHS = 1
MODEL_NAME = 'ENTER_DIRECTORY_HERE'
BATCH_SIZE = 100

input_placeholder = tf.placeholder('float', [None, HEIGHT, WIDTH, 1], name="input_placeholder")
output_placeholder = tf.placeholder('float')

train_data = np.load('training_data_balanced.npy')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,HEIGHT,WIDTH,1)
Y = [i[1] for i in train]
print('The shape of X is ', np.shape(X))
print('The shape of Y label is ', np.shape(Y))
print('you have ', len(X), ' training dataset')

test_x = np.array([i[0] for i in test]).reshape(-1,HEIGHT,WIDTH,1)
test_y = [i[1] for i in test]

raw_output = alexnet_tf_nn(WIDTH, HEIGHT, input_placeholder, 3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=raw_output, labels=output_placeholder))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        epoch_loss = 0
        batch_start_index = 0
        batch_end_index = BATCH_SIZE - 1
        batch = 1
        no_of_training_batches = int(len(X)/BATCH_SIZE)
        for _ in range(no_of_training_batches):
            epoch_x = np.array(X[batch_start_index:batch_end_index])
            epoch_y = np.array(Y[batch_start_index:batch_end_index])
            _, c = sess.run([optimizer, cost], feed_dict={input_placeholder:epoch_x, output_placeholder:epoch_y})
            epoch_loss += c
            batch_start_index += BATCH_SIZE
            batch_end_index += BATCH_SIZE
            print('Epoch: ', epoch, '; Batch ', batch, ' completed out of ', no_of_training_batches, '; loss: ', c)
            batch += 1
        print('Epoch ', epoch, ' completed out of ', EPOCHS, ' epoch-loss: ', epoch_loss)

    correct = tf.equal(tf.argmax(raw_output, 1), tf.argmax(output_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy: ', accuracy.eval({input_placeholder : test_x, output_placeholder : test_y}))
    saver = tf.train.Saver()
    save_path = saver.save(sess, MODEL_NAME)
    print("Model saved in path: ", save_path)

