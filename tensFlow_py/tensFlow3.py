#
# simple graph with variables
# in order to modify we need vairables
#

import tensorflow as tf

w = tf.Variable([0.3],tf.float32)
b = tf.Variable([-0.3],tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#model
model = w*x + b

#loss
sq = tf.square(model - y)
loss = tf.reduce_sum(sq)


#optimise
optimizer = tf.train.GradientDescentOptimizer(0.01)

myTrain  = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(myTrain, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([w,b]))


sess.close()


