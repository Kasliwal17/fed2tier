import torch
from io import BytesIO
import json
import time
from datetime import datetime
import os
from .net import get_net
from .net_lib import train_model, test_model, load_data, train_fedavg, train_scaffold, fedadam, train_fedprox, train_feddyn
from .create_datasets import make_client_datasets
from .data_utils import distributionDataloader
from .ClientConnection_pb2 import  TrainResponse
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from datetime import datetime
from codecarbon import  OfflineEmissionsTracker
from concurrent import futures
import threading
from .server_src.server_evaluate import server_eval


# def train(train_order_message, device, args):

#     data_bytes = train_order_message.modelParameters
#     data = torch.load( BytesIO(data_bytes), map_location="cpu" )
#     model_parameters, control_variate_server  = data['model_parameters'], data['control_variate']

#     config_dict_bytes = train_order_message.configDict
#     config_dict = json.loads( config_dict_bytes.decode("utf-8") )
#     print(config_dict["message"])

#     ##get accuracy threshold for early stopping
#     accuracy_threshold = config_dict['threshold']

#     algorithm = args['algorithm']####algorithm for aggregation at node
#     ## initialize control variate if scaffold is used
#     if algorithm == "scaffold":
#         dummy_modeldict = get_net(config_dict)
#         control_variate = [torch.zeros_like(param).to(device) for param in dummy_modeldict.parameters()]
#     else:
#         control_variate = None

#     client_dicts = prepare_dataset_models(config_dict, device, niid=args['niid'], num_of_clients=args['num_of_clients'], control_variate=control_variate)
  
#     exec(f"from .algorithms.{algorithm} import {algorithm}") # nosec
#     aggregator = eval(algorithm)() # nosec

#     state_dict = deepcopy(model_parameters)
#     rounds = args['rounds']
#     if rounds > config_dict['n_rounds']:
#         rounds = config_dict['n_rounds']
#     path = create_path(f"{save_dir_path}/server_round_0")
#     ##Create 2 dataframes for storing time and carbon emission of each client for eachround. column for clients and rows for rounds
#     time_df = pd.DataFrame(columns=[f"client_{i}" for i in range(len(client_dicts))])
#     carbon_df = pd.DataFrame(columns=[f"client_{i}" for i in range(len(client_dicts))])

#     ##set carbon to true if args['carbon'] is 1 else false
#     carbon = args['carbon']

#     ###run communication rounds for the node in a for loop
#     for round in range(rounds):
#         ###make path for the round(for saving models and results)
#         round_path = f"{path}/round_{round}"
#         os.makedirs(round_path)
#         #create new file inside model_checkpoints to store training results
#         with open(f"{round_path}/FL_results.txt", "w") as file:
#             pass
#         for i, client_dict in enumerate(client_dicts):
#             ##start calculating time and carbon emission for each client
#             start_time = time.time()
#             if carbon:
#                 carbon_tracker = OfflineEmissionsTracker(country_iso_code="IND", output_dir = round_path)
#                 carbon_tracker.start()
#             client_dict["model"].load_state_dict(state_dict)
#             epochs = args["epochs"]
#             if args["algorithm"] == "scaffold":
#                 client_dict['model'], client_dict['delta_c'], client_dict['control_variate']= train_scaffold(client_dict["model"], control_variate, client_dict['control_variate'], client_dict["trainloader"], epochs, device)
#             elif args["algorithm"] == "fedavg":
#                 client_dict['model']=train_fedavg(client_dict["model"], client_dict["trainloader"], epochs, device)
#             elif args["algorithm"] == "fedprox":
#                 client_dict['model']=train_fedprox(client_dict["model"], client_dict["trainloader"], epochs, device, args['mu'])
#             elif args["algorithm"] == "feddyn":
#                 client_dict['model'], client_dict['prev_grads']=train_feddyn(client_dict["model"], client_dict["trainloader"], epochs, device, client_dict['prev_grads'])
#             else:
#                 client_dict['model']=train_model(client_dict["model"], client_dict["trainloader"], epochs, device)
#             ##calculate time and log in time_df
#             end_time = time.time()
#             time_df.loc[round, f"client_{i}"] = end_time - start_time
#             if carbon:
#                 ##calculate carbon emission and log in carbon_df
#                 emission = carbon_tracker.stop()
#                 carbon_df.loc[round, f"client_{i}"] = emission

#         # save_model_states(client_dicts, round_path)
#         ###aggregate the client models
#         trained_models_state_dicts = [client_dict["model"].state_dict() for client_dict in client_dicts]

#         if control_variate:
#             updated_control_variates = [client_dict['delta_c'] for client_dict in client_dicts]
#             trained_model_parameters, control_variate = aggregator.aggregate(state_dict,
#                                             control_variate, trained_models_state_dicts, updated_control_variates)
    
#         else:
#             trained_model_parameters = aggregator.aggregate(state_dict,trained_models_state_dicts)

#         state_dict = trained_model_parameters

#         ###save the results for the round
#         ###eval results can be calculated on any one client as all clients share the same model architecture and testset
#         client_dicts[0]["model"].load_state_dict(state_dict)
#         eval_loss, eval_accuracy = test_model(client_dicts[0]["model"], client_dicts[0]["testloader"], device)
#         eval_result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
#         print("Eval results: ", eval_result)
#         eval_list.append(eval_result)
#         #store the results
#         with open(f"{round_path}/FL_results.txt", "a") as file:
#             file.write( str(eval_result) + "\n" )

#         ##check if the accuracy threshold is reached
#         if eval_accuracy > accuracy_threshold:
#             print("Accuracy threshold reached. Stopping training")
#             break

#     print("train eval")
#     # eval_results = []
#     # for client_dict in client_dicts:
#     #     client_dict["model"].load_state_dict(trained_model_parameters)
#     #     eval_loss, eval_accuracy = test_model(client_dict["model"], client_dict["testloader"])
#     #     eval_results.append( {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy} )
#     # response_dict = average(eval_results)

#     ##eval results can be calculated on any one client as all clients share the same model and testset
#     client_dicts[0]["model"].load_state_dict(state_dict)
#     eval_loss, eval_accuracy = test_model(client_dicts[0]["model"], client_dicts[0]["testloader"], device)
#     response_dict = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
#     response_dict_bytes = json.dumps(response_dict).encode("utf-8")

#     ###apply algorithms for server level aggregation
#     if config_dict['algorithm'] == "fedavg":
#         state_dict = state_dict
#     else:
#         state_dict = fedadam(model_parameters, state_dict)

#     data_to_send = {'model_parameters': state_dict, 'control_variate': control_variate_server}
#     buffer = BytesIO()
#     torch.save(data_to_send, buffer)
#     buffer.seek(0)
#     data_to_send_bytes = buffer.read()

#     train_response_message = TrainResponse(
#         modelParameters = data_to_send_bytes, 
#         responseDict = response_dict_bytes)
#     ##convert both the dataframes to csv and save them
#     time_df.to_csv(f"{save_dir_path}/time.csv")
#     carbon_df.to_csv(f"{save_dir_path}/carbon.csv")

#     return train_response_message, client_dicts

def train(train_order_message, device, args, client_manager):


    accept_conn_after_FL_begin = args['accept_conn_after_FL_begin']
    communication_rounds = args['rounds']
    fraction_of_clients=None

    #create a directory to store the results
    fl_timestamp = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    save_dir_path = f"FL_checkpoints_node/{fl_timestamp}"
    os.makedirs(save_dir_path)
    #create new file inside FL_results to store training results
    with open(f"{save_dir_path}/FL_results.txt", "w") as file:
        pass
    print(f"FL has started, will run for {args['rounds']} rounds...")

    data_bytes = train_order_message.modelParameters
    data = torch.load( BytesIO(data_bytes), map_location="cpu" )
    model_parameters, control_variate_server  = data['model_parameters'], data['control_variate']
    server_model_state_dict = model_parameters

    config_dict_bytes = train_order_message.configDict
    config_dict = json.loads( config_dict_bytes.decode("utf-8") )
    print(config_dict["message"])
    algorithm = args['algorithm']####algorithm for aggregation at node
    ## initialize control variate if scaffold is used
    if algorithm == "scaffold":
        dummy_modeldict = get_net(config_dict)
        control_variate = [torch.zeros_like(param).to(device) for param in dummy_modeldict.parameters()]
    else:
        control_variate = None

    exec(f"from .algorithms.{algorithm} import {algorithm}") # nosec
    aggregator = eval(algorithm)() # nosec

    client_manager.accepting_connections = accept_conn_after_FL_begin
    for round in range(1, communication_rounds + 1):
        clients = client_manager.random_select(client_manager.num_connected_clients(), fraction_of_clients) 
        
        print(f"Communication round {round} is starting with {len(clients)} node(s) out of {client_manager.num_connected_clients()}.")
        config_dict = {"algorithm":algorithm, "message":"train",
                   "dataset":config_dict['dataset'], "net":config_dict['net'], "resize_size":config_dict['resize_size'], "batch_size":config_dict['batch_size'], "epochs":args['epochs'], "carbon-tracker":args['carbon']}
        trained_model_state_dicts = []
        updated_control_variates = []
        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            result_futures = {executor.submit(client.train, server_model_state_dict, control_variate, config_dict) for client in clients}
            for client_index, result_future in zip(range(len(clients)), futures.as_completed(result_futures)):
                trained_model_state_dict, updated_control_variate, results = result_future.result()
                trained_model_state_dicts.append(trained_model_state_dict)
                updated_control_variates.append(updated_control_variate)
                print(f"Training results (client {clients[client_index].client_id}): ", results)
        print("Recieved all trained model parameters.")
        
        selected_state_dicts = trained_model_state_dicts

        if control_variate:
            server_model_state_dict, control_variate = aggregator.aggregate(server_model_state_dict,
                                        control_variate, selected_state_dicts, updated_control_variates)
        else:
            server_model_state_dict = aggregator.aggregate(server_model_state_dict,selected_state_dicts)

        torch.save(server_model_state_dict, f"{save_dir_path}/round_{round}_aggregated_model.pt")

        if args['eval']==1:
            #test on server test set
            print("Evaluating on server test set...")
            eval_result = server_eval(server_model_state_dict,config_dict)
            eval_result["round"] = round
            print("Eval results: ", eval_result)


        ###apply algorithms for server level aggregation
    if config_dict['algorithm'] == "fedavg":
        state_dict = server_model_state_dict
    else:
        state_dict = fedadam(model_parameters, server_model_state_dict)


    response_dict = eval_result
    response_dict_bytes = json.dumps(response_dict).encode("utf-8")

    data_to_send = {'model_parameters': state_dict, 'control_variate': control_variate_server}
    buffer = BytesIO()
    torch.save(data_to_send, buffer)
    buffer.seek(0)
    data_to_send_bytes = buffer.read()

    train_response_message = TrainResponse(
        modelParameters = data_to_send_bytes, 
        responseDict = response_dict_bytes)

    return train_response_message


def set_parameters(set_parameters_order_message, client_manager):
    model_parameters_bytes = set_parameters_order_message.modelParameters
    model_parameters = torch.load( BytesIO(model_parameters_bytes), map_location="cpu" )
    for client in client_manager.random_select():
        client.set_parameters(model_parameters)


