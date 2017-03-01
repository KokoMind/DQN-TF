"""This File for testing the implementation of MAXheap"""

import random
from priority_queue import IndexedMaxHeap

# Generate A Random Array and print it sorted by maximum
arr = [(random.randint(2, 1000), i + 1) for i in range(20)]
sorted_arr = arr.copy()
sorted_arr.sort(reverse=True)
print("The Sorted Array from random:")
print(sorted_arr)

# Create a priority Queue
my = IndexedMaxHeap(30)
my.heapify(arr)

# Get specific IDS
print("IDS before balancing:")
ids = my.get_experience_ids([1, 5, 16, 18])
print(ids)

# Then Rebalance################################
my.balance()

# Then Get it again YOU WILL OBSERVE THAT THE ERROR BECAME MINIMUM
print("IDS after balancing:")
ids = my.get_experience_ids([1, 5, 16, 18])
print(ids)

# TEST THE UPDATE
my.update(21, 400)
my.update(8, 300)
my.update(5, 1)
my.update(1, 2000)
print("After Updating Some variables:")
# OUTPUT the sorted Please check it carefully
sorted = my._sort()
print(sorted)


# ### Run the Testing Code many times to observe that rebalancing is more accurate
