# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(-1)
        p1 , p2 = l1 , l2
        len1, len2 = 0 , 0
        while p1:
            len1 = len1 + 1 
            p1 = p1.next
        print(len1)

    addTwoNumbers([2,4,3],[2,4,3])
    
        
        
