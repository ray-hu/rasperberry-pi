import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time
import numpy as np
import json



def receive_data_type(data):
    datax = []
    datay = []
    if 'model' in mqtt_topic:
        d = json.loads(data)
        d2 = np.array(d[0],dtype=float)
        d3 = np.array(d[1],dtype=float)
        d1 =[d2,d3]
    elif 'data' in mqtt_topic:
        d = json.loads(data)   
        for i in d:
            x1 = [float(h) for h in i[:-1]]
            datax.append(x1)          
            x2 = [str(h) for h in i[-1]]
            datay.append(x2)    
            
        datax=np.array([np.array(xi,dtype=float) for xi in datax])
        datay=np.array([np.array(xi,dtype=object) for xi in datay])
        d1= np.asarray(np.hstack((datax, datay)))
        print(d1.shape)
        print("d1")    
    else: 
        print("ask Yu to add new data type")        
    return d1

def send_data_type(data):
    global mqtt_topic
    model = []
    data1 = []
 
    if 'data' in mqtt_topic:
        for i in data:
            data1.append(i.tolist())
        d1 = json.dumps(data1)
    elif 'model' in mqtt_topic:
        model = [data[0].tolist(),data[1].tolist()]
        d1 = json.dumps(model)
    else: 
        print("ask Yu to add new data type")        
    return d1
    

def on_connect(client, userdata, flags, rc):
    global mqtt_topic
    print("Connected with result code "+str(rc)) 
    client.subscribe(mqtt_topic)

def on_message(client, userdata, msg):
    global message
    global count1
    x = msg.payload.decode("utf-8") ##test
    h = receive_data_type(x)
    message.append(h)
    if len(message)>count:
        client.loop_stop()

def receive(server_ip,topic, port, self_name,count,time_block): #,time1
    global mqtt_topic
    global message
    global count1
    time = float(time_block)
    count1 = count
    message = []
    mqtt_topic = topic
    client = mqtt.Client(self_name)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(server_ip, port, 60)
    while len(message)<count1:
        client.loop(.1)
    return message



def send(server_ip,topic, ports, data):
    '''
    #Input: data: numpy array
    '''
    global mqtt_topic
    mqtt_topic = topic
    data1 = send_data_type(data)

    publish.single(topic, data1, hostname=server_ip,port=ports)

