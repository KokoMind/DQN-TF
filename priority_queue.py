from heapq import _heapify_max as heapify_max
from heapq import _siftdown_max, _siftup_max


# ################################################################################################
# Modified Version of heapq to be used as Max heap
# Source https://github.com/he-zhe/heapq_max/blob/master/heapq_max/heapq_max.py

def heap_pop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt


def heap_pop_push_max(heap, item):
    """Maxheap version of a heappop followed by a heappush."""
    returnitem = heap[0]  # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem


def heap_push_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)


def heap_push_pop_max(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] > item:
        # if item >= heap[0], it will be popped immediately after pushed
        item, heap[0] = heap[0], item
        _siftup_max(heap, 0)
    return item


# ################################################################################################


class MaxHeap:
    """This class implements a Maxheap to be used in rank-based-prioritized-experience"""

    def __init__(self, heap_sz, arr=None, replace=True):
        self._cur_size = 0
        self.max_size = heap_sz
        self._replace = replace

        # Construct the heap and heapify it
        if arr:
            self._heap = arr
            self.heapify()
        else:
            self._heap = []

    def heapify(self, arr=None):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        if arr:
            self._heap = arr
        heapify_max(self._heap)

    def heap_push(self, item):
        """Push into max heap"""
        heap_push_max(self._heap, item)

    def heap_pop(self, item):
        """Pop from max heap"""
        return heap_pop_max(self._heap)

    def heap_push_pop(self, item):
        """Push then Pop"""
        return heap_push_pop_max(self._heap, item)

    def heap_pop_push(self, item):
        """Pop then push (Replace)"""
        return heap_pop_push_max(self._heap, item)
