from binhex import openrsrc
import os

log_path = '/data2/guobo/01_SHIPRSDET/ShipDetv2/switch_vis/log/switch_log.txt'

with open(log_path,'r') as f :
    data = f.readline()