class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, p):
        root = p
        while root != self.parent[root]:
            root = self.parent[root]
        while p != root:
            next_p = self.parent[p]
            self.parent[p] = root
            p = next_p
        return root

    def union(self, p, q):
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return
        if self.size[root_p] < self.size[root_q]:
            self.parent[root_p] = root_q
            self.size[root_q] += self.size[root_p]
        else:
            self.parent[root_q] = root_p
            self.size[root_p] += self.size[root_q]
        self.count -= 1

    def connected_components(self):
        return self.count

    def add_edges(self, edges):
        for u, v in edges:
            self.union(u, v)
