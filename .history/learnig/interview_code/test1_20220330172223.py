# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(-1)
    p1 , p2 = l1 , l2
    len1, len2 = 0 , 0
    while p1:
        len1 = len1 + 1 
        p1 = p1.next
    print(len1)


list1 = [2,4,3]
list2 = [5,6,4]
l1 = ListNode(-1)
l2 = ListNode(-1)
p1,p2 = l1,l2
for i in range(len(list1)):
    p1.next = ListNode(list1[i])
    p2.next = ListNode(list2[i])
addTwoNumbers(l1.next,l2.next)
    
    
    
        
        
