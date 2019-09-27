import numpy as np 
import os
import json
import random
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from protocol.protocol_utils import server_send, server_receive, server_braodcast

def read_data():
    # Get MNIST data, normalize, and divide by level
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X, y, train_size=0.5, test_size=0.5)
    return X_train_1, X_train_2, y_train_1, y_train_2


def assign_data():
    # load datasets for two pis
    X_train_1, X_train_2, y_train_1, y_train_2 = read_data()
    print('_'*30 + ' check the consistency ' + '_'*30)
    print('len(X_1)=:{}, len(y_1)=:{}'.format(len(X_train_1), len(y_train_1)))
    print('len(X_2)=:{}, len(y_2)=:{}'.format(len(X_train_2), len(y_train_2)))
    print('-'*70)

    # send data to pis [X_train_1, y_train_1] is for pi1, [X_train_2, y_train_2] is for pi2
    # suggest compression of data before sending
    np.save('./data/X_train_1.npy', X_train_1)

    server_send([X_train_1, y_train_1], 'ip of pi1')
    server_send([X_train_2, y_train_2], 'ip of pi2')

def receive_and_broadcast():
    ''' 
    receive uploads from workers and take the average of all uplaods,
     and then broadcast the average to all workers
    ''' 

def main():

    # assign datasets to workers
    assign_data()

    # continue the process for 100 iterations
    for i in range(100):
        receive_and_broadcast()  


if __name__ == '__main__':
    main()
