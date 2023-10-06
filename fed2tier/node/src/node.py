
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
from .node_lib import train, set_parameters

#start the client and connect to server
def node_runner(client_manage, configurations):
 
    config_dict = {"message": "eval"}
    num_of_clients = configurations["num_of_clients"]
    algorithm = configurations["algorithm"]
    n_rounds = configurations["rounds"]
    epochs = configurations["epochs"]
    accept_conn_after_FL_begin = configurations["accept_conn_after_FL_begin"]
    ip_address = configurations["ip_address"]

    #create a directory to store the results
    fl_timestamp = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    save_dir_path = f"FL_checkpoints/{fl_timestamp}"
    os.makedirs(save_dir_path)
    #create new file inside FL_results to store training results
    with open(f"{save_dir_path}/FL_results.txt", "w") as file:
        pass
    print(f"FL has started, will run for {n_rounds} rounds...")

    keep_going = True
    wait_time = configurations["wait_time"]
    device = torch.device(configurations["device"])

    

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
                    train_response_message, client_dicts = train(train_order_message, device, configurations)
                    message_to_server = ClientMessage(trainResponse = train_response_message)
                    client_buffer.put(message_to_server)

                if server_message.HasField("setParamsOrder"):
                    set_parameters_order_message = server_message.setParamsOrder
                    set_parameters_response_message = set_parameters(set_parameters_order_message, client_dicts)
                    message_to_server = ClientMessage(setParamsResponse = set_parameters_response_message)
                    client_buffer.put(message_to_server)
                    print("parameters successfuly set")

                if server_message.HasField("disconnectOrder"):
                    print("recieve disconnect order")
                    disconnect_order_message = server_message.disconnectOrder
                    message = disconnect_order_message.message
                    print(message)
                    reconnect_time = disconnect_order_message.reconnectTime
                    if reconnect_time == 0:
                        keep_going = False
                        break
                    wait_time = reconnect_time

def node_start(configurations):
    node_manager = ClientManager()
    node_connection_servicer = ClientConnectionServicer(node_manager)

    channel_opt = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=channel_opt)
    ClientConnection_pb2_grpc.add_ClientConnectionServicer_to_server( node_connection_servicer, server )
    server.add_insecure_port('localhost:8214')
    server.start()

    server_runner_thread = threading.Thread(target = node_runner, args = (node_manager, configurations, ))
    server_runner_thread.start()
    server_runner_thread.join()

    server.stop(None)