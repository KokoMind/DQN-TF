class IndexedMaxHeap:
    """
    This class implements an IndexedMaxheap to be used in rank-based-prioritized-experience
    Note : this solution is a problem-solving technique
    So if you don't understand You are good (Y)
    """

    def __init__(self, heap_sz):
        self._cur_size = 0
        self.max_size = heap_sz
        self._init_structure()

    def _init_structure(self):
        """Initialize the Arrays of my structure"""
        self._heap = [-1 for _ in range(self.max_size + 1000)]  # My heap which will contain the nodes
        self._index = [-1 for _ in range(self.max_size + 1000)]  # The index of the node in the heap
        self._keys = [0 for _ in range(self.max_size + 1000)]  # The priorities

    def heapify(self, arr):
        """Function takes an array and heapify it"""
        # For debugging only
        # Arr must be (Priority, e_id)
        for i, item in enumerate(arr):
            self._insert(item[1], item[0])

    def update(self, e_id, priority):
        """Use it to push or update an element"""
        # Check if it is already exist or to insert a new one
        if self._index[e_id] == -1:
            self._insert(e_id, priority)
        else:
            self._update_node(e_id, priority)

    def get_max_priority(self):
        """Get the max priority, if there is no element return 1"""
        if self._cur_size > 0:
            return self._keys[self._heap[1]]
        else:
            return 1

    def balance(self):
        """Rebalance the Priority Queue"""
        sorted_arr = self._sort()
        self._init_structure()
        self._cur_size = 0
        while self._cur_size < len(sorted_arr):
            self._cur_size += 1
            e_id, priority = sorted_arr[self._cur_size - 1]
            self._index[e_id] = self._cur_size
            self._heap[self._cur_size] = e_id
            self._keys[e_id] = priority
        # sort the heap
        for i in range(self._cur_size // 2, 1, -1):
            self._heap_down(i)

    def get_experience_ids(self, priority_ids):
        """Get Experience e_ids by priority e_ids"""
        return [self._heap[i] for i in priority_ids]

    def _swap(self, i, j):
        """Function to use it in swapping between two nodes i and j"""
        # Pythonic way
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index[self._heap[i]], self._index[self._heap[j]] = i, j

    def _heap_up(self, i):
        while i > 1 and self._keys[self._heap[i // 2]] < self._keys[self._heap[i]]:
            self._swap(i, i // 2)
            i //= 2

    def _heap_down(self, i):
        while 2 * i <= self._cur_size:
            j = 2 * i
            if j < self._cur_size and self._keys[self._heap[j]] < self._keys[self._heap[j + 1]]:
                j += 1
            if self._keys[self._heap[i]] >= self._keys[self._heap[j]]:
                break
            self._swap(i, j)
            i = j

    @property
    def size(self):
        return self._cur_size

    def exist(self, e_id):
        """Check Existence of this key"""
        return self._index[e_id] != -1

    def _insert(self, e_id, priority):
        self._cur_size += 1
        self._index[e_id] = self._cur_size
        self._heap[self._cur_size] = e_id
        self._keys[e_id] = priority
        self._heap_up(self._cur_size)

    def _update_node(self, e_id, priority):
        """update depends on increasing the priority of decreasing it"""
        if priority > self._keys[e_id]:
            self._keys[e_id] = priority
            self._heap_up(self._index[e_id])
        else:
            self._keys[e_id] = priority
            self._heap_down(self._index[e_id])

    def remove(self, e_id):
        """Remove a specific key to be implemented if needed"""
        # TODO
        pass

    def _pop(self):
        """Pop from the indexedMaxHeap"""
        maxi = self._heap[1]
        self._swap(1, self._cur_size)
        self._cur_size -= 1
        self._index[maxi] = -1
        self._heap[self._cur_size + 1] = -1
        self._heap_down(1)
        return maxi, self._keys[maxi]

    def _sort(self):
        """Function that destroy and return a sorted_arr array (e_id, priority) """
        sorted_arr = []
        while self._cur_size > 0:
            sorted_arr.append(self._pop())
        return sorted_arr
