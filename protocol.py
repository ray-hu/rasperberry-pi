import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish





# In[2]:


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    #print("Connected with result code "+str(rc)) 
    client.subscribe(MQTT_PATH)


# In[3]:


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    updates = []
    num_subscribers = 100
    
    update = msg.payload.decode("utf-8")
    updates.append(update)
    num = len(updates)
    if num == num_subscribers:
        model = mean(updates) # average
        publish.single(MQTT_PATH, model, hostname=MQTT_SERVER)



main():
    
    MQTT_SERVER = "172.24.6.253"
    MQTT_PATH = "Max"

    client = mqtt.Client("windows_subscriber")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_SERVER, 12345, 60)


    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_forever()


if __name__ == '__main__':
    main()
