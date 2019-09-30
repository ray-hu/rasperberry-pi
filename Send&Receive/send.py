import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time
import numpy as np
import json


def send(server_ip,topic, ports, data):
    data1 = data.tostring()
    publish.single(topic, data1, hostname=server_ip,port=ports)
