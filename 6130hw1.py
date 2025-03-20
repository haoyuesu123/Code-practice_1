import numpy as np
import time
from collections import deque, defaultdict

def kronecker_generator(SCALE, edgefactor):
    N = 2 ** SCALE
    M = N * edgefactor

    initiator = np.array([[0.57, 0.19],
                          [0.19, 0.05]])

    edges = np.zeros((M, 2), dtype=np.int64)
    for i in range(M):
        u, v = 0, 0
        for j in range(SCALE):
            r = np.random.rand()
            bit = 0 if r < initiator[0, 0] + initiator[0, 1] else 1
            u |= (bit << j)
            r = np.random.rand()
            bit = 0 if r < initiator[0, 0] + initiator[1, 0] else 1
            v |= (bit << j)
        edges[i] = [u, v]
    return edges

def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set()
    parent = {start_node: None}
    distance = {start_node: 0}
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return visited, parent, distance

def validate_bfs(visited, graph, start_node):
    # Kernel 2: Validate the BFS tree
    return len(visited) == len(graph)

def evaluate_graph500(SCALE, edgefactor):
    edges = kronecker_generator(SCALE, edgefactor)  # Kernel 1: Graph Generation
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    start_node = np.random.randint(0, 2 ** SCALE)
    start_time = time.time()
    visited, parent, distance = bfs(graph, start_node)  # Kernel 2: BFS Execution
    valid_bfs = validate_bfs(visited, graph, start_node)  # Kernel 2: Validation

    end_time = time.time()
    elapsed_time = end_time - start_time
    teps = len(edges) / elapsed_time if elapsed_time > 0 else float('inf')

    print(f"Graph500 BFS Evaluation: SCALE={SCALE}, edgefactor={edgefactor}")
    print(f"Elapsed Time: {elapsed_time:.6f} seconds")
    print(f"TEPS: {teps:.2f}")
    print(f"Validation of BFS tree: {'Passed' if valid_bfs else 'Failed'}")

SCALE = 10
edgefactor = 16
evaluate_graph500(SCALE, edgefactor)