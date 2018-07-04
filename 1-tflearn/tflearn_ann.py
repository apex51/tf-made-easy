import random
import numpy as np
import tensorflow as tf
import tflearn


def create_feature_sets_and_labels(test_size=0.3):
    # known patterns (5 features) output of [1] of positions [0,4]==1
    features = [[[0, 0, 0, 0, 0], [0, 1]],
                [[0, 0, 0, 0, 1], [0, 1]],
                [[0, 0, 0, 1, 1], [0, 1]],
                [[0, 0, 1, 1, 1], [0, 1]],
                [[0, 1, 1, 1, 1], [0, 1]],
                [[1, 1, 1, 1, 0], [0, 1]],
                [[1, 1, 1, 0, 0], [0, 1]],
                [[1, 1, 0, 0, 0], [0, 1]],
                [[1, 0, 0, 0, 0], [0, 1]],
                [[1, 0, 0, 1, 0], [0, 1]],
                [[1, 0, 1, 1, 0], [0, 1]],
                [[1, 1, 0, 1, 0], [0, 1]],
                # [[0, 1, 0, 1, 1], [0, 1]],
                [[0, 0, 1, 0, 1], [0, 1]],
                [[1, 0, 1, 1, 1], [1, 0]],
                # [[1, 1, 0, 1, 1], [1, 0]],
                [[1, 0, 1, 0, 1], [1, 0]],
                [[1, 0, 0, 0, 1], [1, 0]],
                [[1, 1, 0, 0, 1], [1, 0]],
                [[1, 1, 1, 0, 1], [1, 0]],
                [[1, 1, 1, 1, 1], [1, 0]],
                [[1, 0, 0, 1, 1], [1, 0]]]

    # shuffle out features and turn into np.array
    random.shuffle(features)
    features = np.array(features)

    # split a portion of the features into tests
    testing_size = int(test_size * len(features))

    # create train and test lists
    train_x = list(features[:, 0])
    train_y = list(features[:, 1])

    return train_x, train_y


train_x, train_y = create_feature_sets_and_labels()

# reset underlying graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, 5])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=100, batch_size=16, show_metric=True)

print(model.predict([[0, 1, 0, 1, 1]]))
print(model.predict([[1, 1, 0, 1, 1]]))

