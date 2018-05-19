import tensorflow as tf
import numpy as np
import cv2
from mss import mss

WIDTH = 80
HEIGHT = 60
EPOCHS = 8
MODEL_NAME = 'NAME_OF_MODEL'

ACTUAL_W = 800
ACTUAL_H = 600

mon = {'top': 35, 'left': 0, 'width': ACTUAL_W, 'height': ACTUAL_H}

sess = tf.Session()
saver = tf.train.import_meta_graph(MODEL_NAME)
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
output = graph.get_tensor_by_name("op_to_restore:0")
prediction = tf.nn.softmax(output)
while(True):
    screen = np.array(mss().grab(mon))
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, (WIDTH, HEIGHT))
    screen = screen.reshape(1,HEIGHT,WIDTH,1)
    p = sess.run(prediction, feed_dict={input_placeholder:screen})
    output = list(np.around(p))
    if output[0][0] == 1:
        print('A')
    elif output[0][1] == 1:
        print('W')
    elif output[0][2] == 1:
        print('D')
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
model = alexnet(WIDTH, HEIGHT,  LR)
model.load(MODEL_NAME)










