import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time
import numpy as np
import json

def on_connect(client, userdata, flags, rc):
    global mqtt_topic
    print("Connected with result code "+str(rc)) 
    client.subscribe(mqtt_topic)

def on_message(client, userdata, msg):
    global message
    global count1
    x = msg.payload ##test
    message.append(np.frombuffer(x))
    if len(message)>count:
        client.loop_stop()

def receive(server_ip,topic, port, self_name,time1,count):
    '''
    Input: time1: timer to be setted manually (use loop() other than loop_start())
    Output: convert the received string to the datatype of orignal data
    '''

    #### when writing functions, clarify the input and output using comment ''' '''
    #### rewrite the receive func for devices, do not use the same func of server


    global mqtt_topic
    global message
    global count1

    
    count1 = count
    message = []
    mqtt_topic = topic
    client = mqtt.Client(self_name)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(server_ip, port, 60)
    client.loop_start()
    time1 = float(time1)
    time.sleep(time1)
    exist = len(message)
    client.loop_stop()

    return message


def send(server_ip,topic, ports, data):
    '''
    Input:
        data: original data, needed to be converted to that can be transfered, accroding to the topic (use if topic, else if)
    '''
    data1 = data.tostring()
    publish.single(topic, data1, hostname=server_ip,port=ports)

