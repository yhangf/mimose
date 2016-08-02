'''
作者:杨航锋
时间:2016.7.29
'''

def binary_search1(alist, item):
    first = 0
    last = len(alist) - 1
    found = False

    while first <= last and not found:

        midpoint = (first + last) // 2
        if alist[midpoint] == item:
            found = True
            print('Found at position: {}'.format(alist.index(item) + 1))
        else:

            if item < alist[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1

        if found is False:
            continue

def binary_search2(aseq, target):
    low = 0
    high = len(aseq) - 1

    while low <= high:

        mid = (low + high) // 2
        mid_val = aseq[mid]

        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            print('Found at position: {}'.format(mid + 1))
            break


print('Enter numbers seprated by space: ')
seq = input()
numbers = list(map(int, seq.split()))
target = int(input('Enter a single number to be found in the list: '))
# binary_search1(numbers, target)
binary_search2(numbers, target)
