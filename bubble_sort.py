'''
作者:杨航锋
时间:2016.7.29
'''

def bubble_sort(alist):

    for passnum in range(len(alist) - 1, 0, -1):
        for i in range(passnum):

            if alist[i] > alist[i+1]:
                alist[i+1], alist[i] = alist[i], alist[i+1]

    return alist

print(bubble_sort([1, 20, -5, 36, 12]))
