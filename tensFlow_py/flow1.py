#
# simple graph with constants
#

import tensorflow as tf

a = tf.constant(3.0, tf.float32)
b = tf.constant(4.0)

sess = tf.Session()
c = a*b

loc = "C:\\Users\\ks_work\\Desktop\\Code\\tensFlow_py\\tsgraph"

fw = tf.summary.FileWriter(loc,sess.graph)

print(sess.run(c))

sess.close()


#  type the following in the file directory (one level up)
#  C:\Users\ks_work\Desktop\Code>tensorboard --logdir="tensFlow_py"