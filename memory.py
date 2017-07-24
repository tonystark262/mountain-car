import numpy as np
import random


class Memory(object):
    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size, dtype=object)
        self.tree = np.zeros(2 * size - 1, dtype=np.float32)
        self.current_size = 0
        self.last = 0

    def append(self, p, data):
        self.current_size = min(self.current_size + 1, self.size)
        cur = self.last + self.size - 1
        self.update_at_index(cur, p - self.tree[cur])
        self.data[self.last] = data
        self.last += 1
        if self.last >= self.size:
            self.last = 0

    def update(self, index, p):
        self.update_at_index(index, p - self.tree[index])

    def update_at_index(self, index, change):
        while (index >= 0):
            self.tree[index] += change
            index = (index - 1) // 2

    def get(self, index, s):
        left = index * 2 + 1
        if (left >= self.size):
            return (index, self.data[index + 1 - self.size])
        if (self.tree[left] >= s):
            return self.get(left, s)
        else:
            right = left + 1
            return self.get(right, s - self.tree[left])

    def sample(self, n):
        av_sum = self.tree[0] / n
        l = []
        m = []
        for i in range(n):
            min_sum = av_sum * i
            max_sum = av_sum * (i + 1)
            s = random.uniform(min_sum, max_sum)
            x = self.get(0, s)
            l.append(x[0])
            m.append(x[1])
        return l, m
