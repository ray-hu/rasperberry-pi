import numpy as np 
import os
import json
import random
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from protocol_utils import receive, send
from model import Model

def read_data():
    ''' Get MNIST data, normalize, and divide by level
    Return: two datasets include X and y
    '''
    # 
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X, y, train_size=0.5, test_size=0.5)
    return X_train_1, X_train_2, y_train_1, y_train_2


def assign_data():
    ''' Assign datasets to corresponding devices
    '''

    # load datasets for two pis
    X_train_1, X_train_2, y_train_1, y_train_2 = read_data()
    print('_'*30 + ' check the consistency ' + '_'*30)
    print('len(X_1)=:{}, len(y_1)=:{}'.format(len(X_train_1), len(y_train_1)))
    print('len(X_2)=:{}, len(y_2)=:{}'.format(len(X_train_2), len(y_train_2)))
    print('-'*70)

    # send data to pis [X_train_1, y_train_1] is for pi1, [X_train_2, y_train_2] is for pi2

    data_1 = np.asarray(np.hstack((X_train_1, y_train_1.reshape(-1,1))))
    data_2 = np.asarray(np.hstack((X_train_1, y_train_1.reshape(-1,1))))

    send("172.24.6.253", "data_01", 12345, data_1[:100,:])
    send("172.24.6.253", "data_02", 12345, data_2[:100,:])

def model_initialier():
    '''
    Return: initialized model for devices to start training
    '''
    optimizer = tf.train.GradientDescentOptimizer(0.1)

    model = Model(num_classes=10, optimizer=optimizer, regu_param=1e-3)

    return model.get_params()

def aggregate(local_models):
    ''' Aggregate local models to update global_model
    Input: a list of local_models
    Output: global model
    '''
    num_devices = len(local_models)
    global_model = 0
    for i in range(num_devices):
        global_model += local_models[i]

    return global_model/num_devices



def main():

    # num of communication round
    num_round = 100

    # num of devices
    num_devices = 2

    # assign datasets to devices
    assign_data()

    # initialize model
    global_model_init = model_initialier()

    ### do not change model datatype here, change inside the send function
    # model_change = [model_init[0].tolist(),model_init[1].tolist()] 

    # boardcast model_init to all devices
    send("172.24.6.253", "global_model", 12345, global_model_init)
    
    # continue the training for 100 iterations
    for i in range(num_round):
        ## I don't know the meaning of self_name, pls specify in your code
        local_models = receive("172.24.6.253", "local_model", 12345, self_name=None,time1=None,count=num_devices)

        # aggregate local models
        # assume I receive a list of local models
        global_model = aggregate(local_model)

        # braodcast globel_model to devices
        send("172.24.6.253", "global_model", 12345, global_model)

        # evaluate using test dataset ( to be cont.)


if __name__ == '__main__':
    main()
