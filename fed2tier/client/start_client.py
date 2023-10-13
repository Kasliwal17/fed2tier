from .src.client import client_start
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default = "localhost:8214", help="IP address of the node")
parser.add_argument("--device", type=str, default = "cpu", help="Device to run the client on")
parser.add_argument('--wait_time', type = int, default= 5, help= 'time to wait before sending the next request')
parser.add_argument('--data_path', type = str, default= "./data/client_1", help= 'path to the data folder')

args = parser.parse_args()

configs = {
    "ip_address": args.ip,
    "wait_time": args.wait_time,
    "device": args.device,
    "data_path": args.data_path,
}

if __name__ == '__main__':
    client_start(configs)
