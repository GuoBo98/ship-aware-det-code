M = int(input())
N = int(input())
tasks = list(map(str,input().split()))

chips = [4] * M

for i in range(len(tasks)):
    if tasks[i] == 'A':
        for j in range(chips):
            if chips[j] >= 1 :
                chips[j] = chips[j] - 1
                break
    if tasks[i] == 'B':
        for j in range(chips):
            if chips[j] == 4 :
                chips[j] = chips[j] - 4 
                break
id_1 = j + 1
id_2 = (4 - chips[j])
print(id_1,id_2)

    
        
    
                