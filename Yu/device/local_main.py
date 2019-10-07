import tensorflow as tf
import numpy as np
from protocol_utils import receive, send
from model import Model


def main():

    # id of this device
    ID = "01"

    # define local model
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    model = Model(num_classes=10, optimizer=optimizer, regu_param=1e-3)

    # num of communication round
    num_rounds = 100
    # num of local iterations
    num_epochs = 10

    # receive dataset from server
    topic = "data_" + ID
    dataset = receive("172.24.6.253", topic,12345, "pi 1",0.1,0)[0]
    print("receive dataset successfully")


    for round in range(num_rounds):
        # receive model parameter from server
        model_params = receive("172.24.6.253", "global_model", 12345, "pi 1",0.1,0)
        print("receive dataset successfully")
        # check if received global model
        if len(model_params) == 0:
            print('='*60)
            print('[INFO] DID NOT RECEIVE GLOBAL MODEL !')
            return 0
        
        # update local model
        print(dataset)
        #print(model_params)
        local_model = model.solve_inner(data=dataset, num_epochs=num_epochs, batch_size=10)
        print("local model good")

        # upload local model to server
        send("172.24.6.253", "local_model", 12345, local_model)
        print("upload good")
        

if __name__ == '__main__':
    main()

    

