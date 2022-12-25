# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    p1 , p2 = l1 , l2
    len1, len2 = 0 , 0
    while p1:
        len1 = len1 + 1 
        p1 = p1.next
    while p2:
        len2 = len2 + 1
        p2 = p2.next
    
    if len1 == len2:
        pass
    elif len1 > len2:
        for i in range(len1 - len2):
            p2.next = ListNode(0)
            p2 = p2.next
    elif len1 < len2:
        for i in range(len2 - len1):
            p1.next = ListNode(0)
            p1 = p1.next 
    
    dummy = ListNode(-1)
    p1 = l1
    p2 = l2
    p3 = dummy
    carry = 0
    while not p1 and not p2:
        _sum = p1.val + p2.val
        if _sum >= 10:
            carry = 1
        else:
            carry = 0
        p3.next = ListNode(_sum % 10)   
        p1 = p1.next
        p2 = p2.next 
        p3 = p3.next 
    
    if carry == 1 :
        p3.next = ListNode(1)
        p3 = p3.next
    return dummy.next
        
        
    
    
            
            


list1 = [2,4,3]
list2 = [5,6,4]
l1 = ListNode(-1)
l2 = ListNode(-1)
p1,p2 = l1,l2
for i in range(len(list1)):
    p1.next = ListNode(list1[i])
    p2.next = ListNode(list2[i])
    p1 = p1.next
    p2 = p2.next
addTwoNumbers(l1.next,l2.next)
    
    
    
        
        
