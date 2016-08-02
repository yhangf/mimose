'''
作者:杨航锋
时间:2016.7.29
'''

def quick_sort(alist):

    if alist == []:
        return []
    else:
        pivot = alist[0]
        lesser = quick_sort([x for x in alist[1:] if x <= pivot])
        greater = quick_sort([x for x in alist[1:] if x > pivot])

    return lesser + [pivot] + greater


print(quick_sort([56, -85, 5, 3.89, 9.1, 32]))
