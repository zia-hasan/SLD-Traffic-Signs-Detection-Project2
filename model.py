# Load pickled data
import pickle
import numpy as np
import util
import os
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data
training_file  = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

### Preprocess the data here.
### Feel free to use as many code cells as needed.
## Normalization
## RGB to Grayscale
# Center Normalization
    
X_train = util.min_max_normalization(X_train)
X_valid = util.min_max_normalization(X_valid)
X_test  = util.min_max_normalization(X_test)

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 2
BATCH_SIZE = 128
tf.reset_default_graph()

def LeNet2(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    wc1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma), name="wc1")
    b1 = tf.Variable(tf.zeros(6), name="b1")
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID') + b1
    conv1 = tf.nn.relu(conv1) # Activation
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    wc2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="wc2")
    b2 = tf.Variable(tf.zeros(16), name="b2")
    conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='VALID') + b2
    conv2 = tf.nn.relu(conv2)  # Activation
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 1024.
    wc3 = tf.Variable(tf.truncated_normal(shape=(400, 1024), mean=mu, stddev=sigma), name="wc3")
    b3 = tf.Variable(tf.zeros(1024), name="b3")
    conv3 = tf.matmul(fc0, wc3) + b3
    conv3 = tf.nn.relu(conv3) # Activation
    conv3 = tf.nn.dropout(conv3, keep_prob) # Dropout

    # Layer 4: Fully Connected. Input = 1024. Output = 512
    wc4 = tf.Variable(tf.truncated_normal(shape=(1024, 512), mean=mu, stddev=sigma), name="wc4")
    b4 = tf.Variable(tf.zeros(512), name="b4")
    conv4 = tf.matmul(conv3, wc4) + b4
    conv4 = tf.nn.relu(conv4) # Activation
    conv4 = tf.nn.dropout(conv4, keep_prob) # Dropout

    # Layer 5: Fully Connected. Input = 512. Output = 43.
    wc5 = tf.Variable(tf.truncated_normal(shape=(512, 43), mean=mu, stddev=sigma), name="wc5")
    b5 = tf.Variable(tf.zeros(43), name="b5")
    logits = tf.matmul(conv4, wc5) + b5
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
### Train your model here.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet2(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("logits", logits)
tf.add_to_collection("correct_prediction", correct_prediction)
tf.add_to_collection("accuracy_operation", accuracy_operation)

saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet2.ckpt')
    print("Model saved")
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    # print([v.op.name for v in tf.global_variables()])
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    # print([v.op.name for v in tf.global_variables()])
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
        
tf.reset_default_graph()
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
X_test  = util.min_max_normalization(X_test)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./lenet2.ckpt.meta')
    saver.restore(sess, "./lenet2.ckpt")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))