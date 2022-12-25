from binhex import openrsrc
import os
import matplotlib.pyplot as plt
import numpy as np

orignal_class = [1,2,8,9,10,11,16,40,4,5,6,7,12,13,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,47]
orignal_nums = [1012,942,258,171,169,395,423,618,28,69,82,13,88,68,48,140,69,61,53,40,31,77,29,14,95,126,88,60,22,43,31,37,28,48,42,107,104]
after_switch_nums = list(orignal_nums)

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

for item in all:
    more_class = int(item[1])
    switch_class = int(item[2])
    after_switch_nums[orignal_class.index(more_class)] -= 1
    after_switch_nums[orignal_class.index(switch_class)] += 1
    
ind = np.arange(len(after_switch_nums))
plt.bar(ind,after_switch_nums)
plt.xticks(ind,orignal_class)
plt.title('Before-Switch')
plt.savefig('/data2/guobo/01_SHIPRSDET/ShipDetv2/switch_vis/log/Before-Switch.png')
    
print(all)