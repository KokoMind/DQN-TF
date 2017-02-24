class IndexedMaxHeap:
    """This class implements a Maxheap to be used in rank-based-prioritized-experience"""

    def __init__(self, heap_sz, arr=None, replace=True):
        self._cur_size = 0
        self.max_size = heap_sz
        self._replace = replace

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

    def _insert(self, id):
        pass

    def remove(self, id):
        pass

    def _push(self):
        pass

    def _pop(self):
        pass

    def _heap_up(self):
        pass

    def _heap_down(self):
        pass
