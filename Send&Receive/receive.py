import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time
import numpy as np

def on_connect(client, userdata, flags, rc):
    global mqtt_topic
    print("Connected with result code "+str(rc)) 
    client.subscribe(mqtt_topic)

def on_message(client, userdata, msg):
    global message
    x = msg.payload ##test
    message.append(np.frombuffer(x))

def receive(server_ip,topic, port, self_name):
    global mqtt_topic
    global message
    message = []
    mqtt_topic = topic
    client = mqtt.Client(self_name)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(server_ip, port, 60)
    client.loop_start()
    time.sleep(10)
    client.loop_stop()
    return message
