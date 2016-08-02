'''
作者:杨航锋
时间:2016.7.29
'''

def dijkstra(graph, n):

    _ = float('inf')
    dis = [0] * n
    flag = [False] * n
    pre = [0] * n
    flag[0] = True
    k = 0

    # 初始化dis列表
    for i in range(n):
        dis[i] = graph[k][i]

    # 复制dis列表，不能直接tem_dis = dis
    tem_dis = dis[:]
    tem_dis[flag.index(True)] = _
    min_dex = tem_dis.index(min(tem_dis))
    if not flag[min_dex]:
        k = min_dex
        if k is 0:
            return

    flag[k] = True

    # 核心，修改dis列表的权值
    for i in range(1, n + 1):
        if dis[i] > dis[k] + graph[k][i]:
            dis[i] = dis[k] + graph[k][i]
            pre[i] = k

    return dis, pre

if __name__ == '__main__':

    _ = float('inf')
    n = 6
    graph = [
            [0, 6, 3, _, _, _],
            [6, 0, 2, 5, _, _],
            [3, 2, 0, 3, 4, _],
            [_, 5, 3, 0, 2, 3],
            [_, _, 4, 2, 0, 5],
            [_, _, _, 3, 5, 0],
    ]
    dis, pre = dijkstra(graph, n)
    print(dis)
    print(pre)
