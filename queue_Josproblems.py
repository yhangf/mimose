'''
作者:杨航锋
时间:2016.7.29
'''

class Queue:

    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)

    def empty(self):
        return self.size() == 0

# solve Josephus problems
# method one
def josephus1(namelist, num):

    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)

    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())
        simqueue.dequeue()
    return simqueue.dequeue()

# method two
def josephus2(namelist, num):

    from collections import deque

    Queue = deque()
    for name in namelist:
        Queue.append(name)
    while len(Queue) > 1:
        for i in range(num):
            Queue.append(Queue.popleft())
        Queue.popleft()
    return Queue.popleft()

if __name__ == '__main__':
    print(josephus1(['yanghangfeng', 'wulin', 'xiaokai', 'jiangguoxing'], 3))
    print(josephus2(['yanghangfeng', 'wulin', 'xiaokai', 'jiangguoxing'], 3))
