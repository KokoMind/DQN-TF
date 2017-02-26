import math


class IndexedMaxHeap:
    """
    This class implements an IndexedMaxheap to be used in rank-based-prioritized-experience
    Note : this solution is a problem-solving technique
    So if you don't understand You are good (Y)
    """

    def __init__(self, heap_sz, arr=None):
        self._cur_size = 0
        self.max_size = heap_sz
        self._init_structure()

    def _init_structure(self):
        self._heap = [-1 for i in range(self.max_size + 1000)]  # My heap which will contain the nodes
        self._index = [-1 for i in range(self.max_size + 1000)]  # The index of the node in the heap
        self._keys = [0 for i in range(self.max_size + 1000)]  # The priorities
        self._rank = [-1 for i in range(self.max_size + 1000)]  # Nodes rank

    def update(self, e_id, priority):
        """Use it to push or update an element"""
        # Check if it is already exist or to insert a new one
        if self._index[e_id] == -1:
            self._insert(e_id, priority)
        else:
            self._update_node(e_id, priority)

    def get_max_priority(self):
        """Get the max priority, if there is no element return 1"""
        if self.size > 0:
            return self._keys[self._heap[1]]
        else:
            return 1

    def balance(self):
        """Rebalance the Priority Queue"""
        self._init_structure()
        sorted = self._sort()
        self._cur_size = 0
        while self._cur_size <= len(sorted):
            self._cur_size += 1
            id, priority = sorted[self._cur_size - 1]
            self._index[id] = self._cur_size
            self._heap[self._cur_size] = id
            self._rank[self._cur_size] = id
            self._keys[id] = priority
        for i in range(math.floor(len(sorted) / 2), 1, -1):
            self._heap_down(i)

    def get_experience_ids(self, priority_ids):
        """Get Experience ids by priority ids"""
        return [self._rank[i] for i in priority_ids]

    def _swap(self, i, j):
        """Function to use it in swapping between two nodes i and j"""
        # Pythonic way
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index[self._heap[i]], self._index[self._heap[j]] = i, j
        self._rank[i], self._rank[j] = self._heap[i], self._heap[j]

    def _heap_up(self, i):
        while i > 1 and self._keys[self._heap[i / 2]] < self._keys[self._heap[i]]:
            self._swap(i, i / 2)
            i /= 2

    def _heap_down(self, i):
        j = 0
        while 2 * i <= self._cur_size:
            j = 2 * i
            if j < self._cur_size and self._keys[self._heap[j]] < self._keys[self._heap[j + 1]]:
                j += 1
            if self._keys[self._heap[i]] <= self._keys[self._heap[j]]:
                break
            self._swap(i, j)
            i = j

    @property
    def size(self):
        return self._cur_size

    def exist(self, id):
        return self._index[id] != -1

    def _insert(self, id, priority):
        self._cur_size += 1
        self._index[id] = self._cur_size
        self._heap[self._cur_size] = id
        self._rank[self._cur_size] = id
        self._keys[id] = priority
        self._heap_up(self._cur_size)

    def _update_node(self, id, priority):
        """update depends on increasing the priority of decreasing it"""
        if priority > self._keys[id]:
            self._keys[id] = priority
            self._heap_down(self._index[id])
        else:
            self._keys[id] = priority
            self._heap_up(self._index[id])

    def remove(self, id):
        pass

    def _push(self):
        pass

    def _pop(self):
        """Pop from the indexedMaxHeap"""
        maxi = self._heap[1]
        self._cur_size -= 1
        self._swap(1, self._cur_size)
        self._heap_down(1)
        self._index[maxi] = -1
        self._heap[self._cur_size + 1] = -1
        self._rank[1] = self._heap[1]
        self._heap_down(1)
        return maxi, self._keys[maxi]

    def _sort(self):
        """Function that destroy and return a sorted array (id, priority) """
        sorted = []
        while self._cur_size > 0:
            sorted.append(self._pop())
        return sorted.reverse()
