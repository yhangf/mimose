'''
作者:杨航锋
时间:2016.7.29
'''

def insert_sort(alist):

    for i in range(len(alist) - 1):
        for j in range(i + 1, len(alist)):
            if alist[i] > alist[j]:
                alist[i], alist[j] = alist[j], alist[i]

    return alist

print(insert_sort([2, 12, -5, 6]))
