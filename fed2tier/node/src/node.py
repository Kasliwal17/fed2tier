
import json
from queue import Queue
import torch
from io import BytesIO
import time

import grpc
from . import ClientConnection_pb2_grpc
from .ClientConnection_pb2 import ClientMessage

from .server_src.client_manager import ClientManager
from .server_src.client_connection_servicer import ClientConnectionServicer 
from concurrent import futures
import threading
import os
from datetime import datetime
from .node_lib import train, set_parameters

#start the client and connect to server
def node_runner(client_manager, configurations):
    
    print("\nStarted node runner")
    config_dict = {"message": "eval"}
    num_of_clients = configurations["num_of_clients"]
    fraction_of_clients=None
    clients = client_manager.random_select(num_of_clients, fraction_of_clients)
    accept_conn_after_FL_begin = configurations["accept_conn_after_FL_begin"]
    ip_address = configurations["ip_address"]

    keep_going = True
    wait_time = configurations["wait_time"]
    device = torch.device(configurations["device"])
    client_manager.accepting_connections = accept_conn_after_FL_begin

    

    while keep_going:
        #wait for specified time before reconnecting
        time.sleep(wait_time)
        
        #create new gRPC channel to the server
        with grpc.insecure_channel(ip_address, options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
                ]) as channel:
            stub = ClientConnection_pb2_grpc.ClientConnectionStub(channel)
            client_buffer = Queue(maxsize = 1)
            client_dicts=None
            #wait for incoming messages from the server in client_buffer
            #then according to fields present in them call the appropraite function
            for server_message in stub.Connect( iter(client_buffer.get, None) ):
                
                if server_message.HasField("trainOrder"):
                    train_order_message = server_message.trainOrder
                    train_response_message= train(train_order_message, device, configurations, client_manager)
                    message_to_server = ClientMessage(trainResponse = train_response_message)
                    client_buffer.put(message_to_server)

                if server_message.HasField("setParamsOrder"):
                    set_parameters_order_message = server_message.setParamsOrder
                    set_parameters_response_message = set_parameters(set_parameters_order_message, client_manager)
                    message_to_server = ClientMessage(setParamsResponse = set_parameters_response_message)
                    client_buffer.put(message_to_server)
                    print("parameters of clients successfuly set")

                if server_message.HasField("disconnectOrder"):
                    print("recieve disconnect order")
                    disconnect_order_message = server_message.disconnectOrder
                    message = disconnect_order_message.message
                    print(message)
                    print("disconnecting clients...")
                    for client in client_manager.random_select():
                        client.disconnect()
                    print("clients disconnected")
                    reconnect_time = disconnect_order_message.reconnectTime
                    if reconnect_time == 0:
                        keep_going = False
                        break
                    wait_time = reconnect_time

def node_start(configurations):
    client_manager = ClientManager()
    client_connection_servicer = ClientConnectionServicer(client_manager)

    channel_opt = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=channel_opt)
    ClientConnection_pb2_grpc.add_ClientConnectionServicer_to_server( client_connection_servicer, server )
    server.add_insecure_port('localhost:8214')
    server.start()

    server_runner_thread = threading.Thread(target = node_runner, args = (client_manager, configurations, ))
    server_runner_thread.start()
    server_runner_thread.join()

    server.stop(None)