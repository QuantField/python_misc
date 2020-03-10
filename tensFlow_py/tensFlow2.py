#
# simple graph with variables
#

import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

sess = tf.Session()
add_node = a+b

print(sess.run(add_node,{a:[1,3], b:[2,4]}))

loc = "C:\\Users\\ks_work\\Desktop\\Code\\tensFlow_py\\tsgraph"

#fw = tf.summary.FileWriter(loc,sess.graph)

#print(sess.run(c))

sess.close()


#  type the following in the file directory (one level up)
#  C:\Users\ks_work\Desktop\Code>tensorboard --logdir="tensFlow_py"