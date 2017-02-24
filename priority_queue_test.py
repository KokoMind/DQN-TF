"""This File for testing the implementation of MAXheap"""

import random
from priority_queue import MaxHeap

arr = [random.randint(0, 1000) for i in range(20)]

my = MaxHeap(20, arr, True)
my.push(1)
my.pop()
my.push_pop(5)
my.pop_push(10)
print(my.sort_retrieve())
