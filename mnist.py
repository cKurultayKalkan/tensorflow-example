import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data('./datasets')

print(train_images, train_labels, test_images, test_labels)

print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))

tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

logits = tf.matmul(x, w) + b
y = tf.nn.softmax(logits)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 128


def get_paths_and_labels(path):
    """ image_paths  :  list of relative image paths
        labels       :  mix of alphanumeric characters """
    image_paths = [path + image for image in os.listdir(path)]
    labels = [i.split(".")[-3] for i in image_paths]
    labels = [i.split("/")[-1] for i in labels]
    return image_paths, labels


def encode_labels(train_labels, test_labels):
    """ Assigns a numeric value to each label since some are subject's names """
    found_labels = []
    index = 0
    mapping = {}
    for i in train_labels:
        if i in found_labels:
            continue
        mapping[i] = index
        index += 1
        found_labels.append(i)
    return [mapping[i] for i in train_labels], [mapping[i] for i in test_labels], mapping


train_labels, test_labels, mapping = encode_labels(train_labels, test_labels)


# numeric_train_ids = [labels[idx] for idx in train_labels]
# numeric_test_ids = [labels[idx] for idx in test_labels]

# one_hot_train_labels = tf.one_hot(indices=numeric_train_ids, depth=num_classes)
# one_hot_test_labels = tf.one_hot(indices=numeric_test_ids, depth=num_classes)


def train_step(iterations):
    for i in range(iterations):
        start = i * batch_size
        end = (batch_size * (i + 1)) - 1
        sess.run(optimize, feed_dict={x: train_images[start:end], y_true: train_labels[start:end]})


train_step(1)
