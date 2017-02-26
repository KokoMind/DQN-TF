"""This File for testing the implementation of MAXheap"""

import random
from priority_queue import IndexedMaxHeap

arr = [(random.randint(0, 1000), i + 1) for i in range(20)]
sorted_arr = arr.copy()
sorted_arr.sort()
print(arr)
print(sorted_arr)
my = IndexedMaxHeap(30)
my.heapify(arr)

ids = my.get_experience_ids([1, 5, 16, 18])
print(ids)

my.balance()

my.update(5, 300)

sorted = my._sort()
print(sorted)
