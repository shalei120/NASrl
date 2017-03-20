import tensorflow as tf
import numpy as np

elems = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])
sum = tf.scan(lambda a, x:  x[0]+x[1], (elems,y))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print sess.run([sum])