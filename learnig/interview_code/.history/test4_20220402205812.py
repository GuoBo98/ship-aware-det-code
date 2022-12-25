'''
Author: Shuailin Chen
Created Date: 2022-04-02
Last Modified: 2022-04-02
	content: 
'''
''' 第二题 '''
T = int(input())
for _ in range(T):
    n = int(input())
    init = input()

    abc = [(init.count(ch)-n, ch) for ch in ('A', 'B', 'C')]
    abc.sort()


    if all([abc[i][0]==0 for i in range(3)]):
        print(0)
        # break
    
    elif abc[1][0] >= 0:
        ''' 只要刷一次的情况 '''
        left = 0
        win_size = -abc[0][0] + 1
        right = win_size
        
        win = init[left: right]
        cur_cnt = [0] * 2
        for ch in win:
            if ch == abc[1][1]:
                cur_cnt[0] -= 1
            elif ch == abc[2][1]:
                cur_cnt[1] -= 1
        
        flag = False
        while right < len(init):
            if cur_cnt[0] > abc[]
        
        
        for i in range(3*n-win_size+1):
            win = init[i:i+win_size]
            cur_cnt = [0] * 2
            for ch in win:
                if ch == abc[1][1]:
                    cur_cnt[0] -= 1
                elif ch == abc[2][1]:
                    cur_cnt[1] -= 1
            
            cur_res = [ori[0] + cur for ori, cur in zip(abc[1:], cur_cnt)]
            if all([val ==0 for val in cur_res]):
                print(1)
                flag = True
                break
    if not flag:
        print(2)
            

'''
1
2
ABACBC

1
3
AAABBBBAC

1
3
CAACBCBCC
'''