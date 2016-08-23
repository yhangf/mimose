'''
作者：杨航锋
日期：20160823
'''

class Stack:

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def clear(self):
        del self.items[:]

    def empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)

    def top(self):
        return self.items[self.size()-1]

# 十进制转二进制 （模二取余,逆序排列）

def divideBy2(number):
    remstack = Stack()

    while number > 0:
        rem = number % 2
        remstack.push(rem)
        number = number // 2

    binString = ''
    while not remstack.empty():
        binString += str(remstack.pop())

    return binString

if __name__ == '__main__':
    print(divideBy2(233))
