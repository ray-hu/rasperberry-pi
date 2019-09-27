import tensorflow as tf
import numpy as np
from protocol.protocol_utils import uplaod, receive

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x'] ## modify
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def local_solver(model_params=None):
    # load local dataset
    train_X = 0
    train_y = 0
    test_x = 0
    test_y = 0
    data_train = []


    # Parameters
    learning_rate = 0.01
    epochs = 25
    batch_size = 10
    display_step = 1
    num_classes = 10
    regu_param = 1e-3

    # initialize graph
    graph = tf.Graph()
    with graph.as_default():
        # tf Graph Input
        X = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

        logits = tf.layers.dense(inputs=X, units=num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(regu_param))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        predictions = {
                    "classes": tf.argmax(input=logits, axis=1),
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                    }

        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        eval_metric_ops = tf.count_nonzero(tf.equal(y, predictions["classes"]))
        saver = tf.train.Saver()
    sess = tf.Session(graph=graph)
    # initalization

    if model_params is not None:
        with graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, model_params):
                variable.load(value, sess)
    
    # train
    for epoch in range(epochs):
        for batch_X, batch_y in batch_data(data_train, batch_size):
            with graph.as_default():
                sess.run(train_op, feed_dict={X: batch_X, y: batch_y})
    soln =  sess.run(tf.trainable_variables())
    return soln

def main():
    # receive model_params from server
    model_params = receive('ip of server')
    
    # calculate local model
    local_model = local_solver(model_params)
    # upload local model to server
    upload(local_model, 'ip of server')

if __name__ == '__main__':
    main()
