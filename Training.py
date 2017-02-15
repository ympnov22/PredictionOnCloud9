import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import sys

args = sys.argv

INPUT = 54
HIDDEN_1 = int(args[1])
HIDDEN_2 = int(args[2])
OUTPUT = 2

TEST_SIZE = 0.1
start = time.time()
subdir = datetime.now().strftime("%Y%m%d%H%M%S")

data_x = np.genfromtxt( "TrainingData_x.csv", delimiter=",", filling_values=(0, 0, 0) )
data_y = np.genfromtxt( "TrainingData_y.csv", delimiter=",", filling_values=(0, 0, 0) )

train_x = []
train_y = []

test_x = []
test_y = []

batch_x = []
batch_y = []

training_loss = []
testing_loss =[]

acuracy = []

profit = []

laptime = []

for i in range(len(data_x)):
    
    if(np.random.rand() > TEST_SIZE):
        train_x.append(data_x[i])
        train_y.append(data_y[i])
        
    else:
        test_x.append(data_x[i])
        test_y.append(data_y[i])

x = tf.placeholder(tf.float32, [None, INPUT])

w1 = tf.Variable(tf.random_normal([INPUT, HIDDEN_1]))
b1 = tf.Variable(tf.zeros([HIDDEN_1]))

w2 = tf.Variable(tf.random_normal([HIDDEN_1, HIDDEN_2]))
b2 = tf.Variable(tf.zeros([HIDDEN_2]))

wy = tf.Variable(tf.random_normal([HIDDEN_2, OUTPUT]))
by = tf.Variable(tf.zeros([OUTPUT]))

h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
y = tf.matmul(h2, wy) + by

y_ = tf.placeholder(tf.float32, [None, OUTPUT])

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.0005)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()

ckpt = tf.train.get_checkpoint_state('./')

if(ckpt):
    last_model = ckpt.model_checkpoint_path
    print("load " + last_model)
    saver.restore(sess, last_model)

else: 
    print("inti variables")
    sess.run(init)

#summary_writer = tf.summary.FileWriter('tensorboard_log/' + subdir, graph_def=sess.graph_def)

for i in range(50000):
        
  sess.run(train, feed_dict={x: train_x, y_: train_y})
  
  if(i%1000 == 0):
    print("training..." + "step:" + str(i))
    print(sess.run(loss, feed_dict={x: train_x, y_: train_y}))
    training_loss.append(sess.run(loss, feed_dict={x: train_x, y_: train_y}))
    
    print("testing...")
    print(sess.run(loss, feed_dict={x: test_x, y_: test_y}))
    testing_loss.append(sess.run(loss, feed_dict={x: test_x, y_: test_y}))
  
    output = sess.run(y, feed_dict={x: test_x})
    
    temp_a = 0
    temp_p = 0
    for j in range(len(output)):
        if(output[j][1]*test_y[j][1]>0):
            temp_a = temp_a + 1
            temp_p += abs(test_y[j][1])
        else:
            temp_p -= abs(test_y[j][1])

    temp_a = float(temp_a) / len(output)
    
    print('acuracy is ')
    print(temp_a)

    print('profit is ')
    print(temp_p)

    acuracy.append(temp_a)
    profit.append(temp_p)
    
result_y = sess.run(y, feed_dict={x: test_x, y_: test_y})

np.savetxt("test_x.csv", test_x, delimiter=",")
np.savetxt("test_y.csv", test_y, delimiter=",")
np.savetxt("test_y_.csv", result_y, delimiter=",")

np.savetxt("training_loss.csv", training_loss, delimiter=",")
np.savetxt("testing_loss.csv", testing_loss, delimiter=",")

np.savetxt("acuracy.csv", acuracy, delimiter=",")
np.savetxt("profit.csv", profit, delimiter=",")

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
laptime.append(elapsed_time)
np.savetxt("elapsed_time.csv", laptime, delimiter=",")

saver.save(sess, "./model.ckpt")
