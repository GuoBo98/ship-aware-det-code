'''
Author: Shuailin Chen
Created Date: 2022-03-30
Last Modified: 2022-04-02
	content: 
'''

# T = int(input())
# for _ in range(T):
#     n = int(input())
#     init = input()

#     abc = [(init.count(ch)-n, ch) for ch in ('A', 'B', 'C')]
#     abc.sort()


#     if all([abc[i][0]==0 for i in range(3)]):
#         print(0)
#         break
    
#     elif abc[1][0] >= 0:
#         ''' 只要刷一次的情况 '''
#         win_size = -abc[0][0]
#         flag = False
#         for i in range(3*n-win_size+1):
#             win = init[i:i+win_size]
#             cur_cnt = [0] * 2
#             for ch in win:
#                 if ch == abc[1][1]:
#                     cur_cnt[0] -= 1
#                 elif ch == abc[2][1]:
#                     cur_cnt[1] -= 1
            
#             cur_res = [ori[0] + cur for ori, cur in zip(abc[1:], cur_cnt)]
#             if all([val ==0 for val in cur_res]):
#                 print(1)
#                 flag = True
#                 break
#     if not flag:
#         print(2)
            