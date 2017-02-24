class IndexedMaxHeap:
    """
    This class implements an IndexedMaxheap to be used in rank-based-prioritized-experience
    Note : this solution is a problem-solving technique
    So if you don't understand You are good (Y)
    """

    def __init__(self, heap_sz, arr=None):
        self._cur_size = 0
        self.max_size = heap_sz

        self._heap = [-1 for i in range(self.max_size + 1000)]  # My heap which will contain the nodes
        self._index = [-1 for i in range(self.max_size + 1000)]  # The index of the node in the heap
        self._keys = [0 for i in range(self.max_size + 1000)]  # The priorities

    def update(self, priority, e_id):
        """Use it to push or update an element"""
        pass

    def get_max_priority(self):
        """Get the max priority, if there is no element return 1"""
        pass

    def balance(self):
        """Rebalance the Priority Queue"""
        pass

    def get_experience_ids(self, priority_ids):
        """Get Experience ids by priority ids"""
        pass

    def _swap(self, i, j):
        """Function to use it in swapping between two nodes i and j"""
        # Pythonic way
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index[self._heap[i]], self._index[self._heap[j]] = i, j

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

    def _insert(self, id):
        pass

    def remove(self, id):
        pass

    def _push(self):
        pass

    def _pop(self):
        pass
