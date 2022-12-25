M = int(input())
N = int(input())
tasks = list(map(str,input().split()))

chips = [4] * M
A_count = tasks.count('A')
B_count = tasks.count('B')
if B_count * 4 + A_count > 4 * M :
    print(0)
    print(0)
else:
    for i in range(len(tasks)):
        if tasks[i] == 'A':
            for j in range(len(chips)):
                if chips[j] >= 1 :
                    chips[j] = chips[j] - 1
                    break
        if tasks[i] == 'B':
            for j in range(len(chips)):
                if chips[j] == 4 :
                    chips[j] = chips[j] - 4 
                    break
    id_1 = j + 1
    id_2 = (4 - chips[j])
    # if id_2 == 0:
    #     id_2 = 1
    print(id_1,id_2)

    
        
    
                