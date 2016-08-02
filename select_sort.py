'''
作者:杨航锋
时间:2016.7.29
'''

def select_sort(alist):

    for i in range(len(alist) - 1):
        min_val = i
        for j in range(i + 1, len(alist)):
            if alist[min_val] > alist[j]:
                min_val = j

        alist[i], alist[min_val] = alist[min_val], alist[i]

    return alist

print(select_sort([14, 5, -56, 2, 7]))
