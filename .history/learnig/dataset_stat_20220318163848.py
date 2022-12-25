from binhex import openrsrc
import os
log_path = '/data2/guobo/01_SHIPRSDET/ShipDetv2/switch_vis/log/switch_log.txt'
temp = []
all = []
with open(log_path,'r') as f :
    for line in f.readlines():
        temp = []
        line = line.strip('\n')
        line_list = line.split(' ')
        if line_list[0][0] is '#':
            break
        for item in line_list:
            if item is not '':
                temp.append(item)
        temp[2] = temp[2][3:]
        all.append(temp)

print(all)