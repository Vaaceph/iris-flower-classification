from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf

dataset_path = Path(__file__).resolve().parent / 'iris.data'
print(dataset_path)
file_ = open(dataset_path)

data = []
label = []

# Loading and labeling data
for line in file_:
    s = line.split(',')
    data.append(list(map(float, s[:-1])))
    if s[-1] == 'Iris-setosa\n':
        label.append([1, 0, 0])
    elif s[-1] == 'Iris-versicolor\n':
        label.append([0, 1, 0])
    elif s[-1] == 'Iris-virginica\n':
        label.append([0, 0, 1])

# Shuffle data
indexes = np.arange(150)
np.random.shuffle(indexes)

# Dividing into train and test data
data_ = np.array(data)
label_ = np.array(label)

SPLIT = 105
TOTAL = 151

train_data = data_[indexes[:SPLIT]]
train_label = label_[indexes[:SPLIT]]

test_data = data_[indexes[SPLIT:TOTAL]]
test_label = label_[indexes[SPLIT:TOTAL]]

# Neural Network
sess = tf.Session()
train_data_ = tf.placeholder(dtype=tf.float32, shape=[None, 4])
train_label_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])

fd = {train_data_: train_data, train_label_: train_label}

# build model in TF
Z1 = 20
Z2 = 20
W1 = tf.Variable(tf.random_uniform([4, Z1], -0.01, 0.01, dtype=tf.float32))
W2 = tf.Variable(tf.random_uniform([Z1, Z2], -0.01, 0.01, dtype=tf.float32))
W3 = tf.Variable(tf.random_uniform([Z2, 3], -0.01, 0.01, dtype=tf.float32))
b1 = tf.Variable(tf.random_uniform([1, Z1], -0.01, 0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([1, Z2], -0.01, 0.01, dtype=tf.float32))
b3 = tf.Variable(tf.random_uniform([1, 3], -0.01, 0.01, dtype=tf.float32))

a1 = tf.nn.relu(tf.matmul(train_data_, W1) + b1)
a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
model_output = tf.matmul(a2, W3) + b3

sess.run(tf.global_variables_initializer())

# classification accuracy
nrCorrect = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(model_output, axis=1), tf.argmax(train_label_, axis=1)), tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=model_output, labels=train_label_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
update = optimizer.minimize(loss)
for iteration in range(0, 2000):
    sess.run(update, feed_dict=fd)
    correct, lossVal = sess.run([nrCorrect, loss], feed_dict=fd)
    testacc = sess.run(nrCorrect, feed_dict={
                       train_data_: test_data, train_label_: test_label})
    print("epoch ", iteration, "acc=", round(float(correct), 2),
          "loss=", round(lossVal, 2), "testacc=", round(testacc, 2))
