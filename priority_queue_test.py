"""This File for testing the implementation of MAXheap"""

import random
from priority_queue import IndexedMaxHeap

arr = [(random.randint(0, 1000), i + 1) for i in range(20)]
sorted_arr = arr.copy()
sorted_arr.sort()
print(sorted_arr)
my = IndexedMaxHeap(30)
my.heapify(arr)

sorted = my._sort()

print(sorted)
