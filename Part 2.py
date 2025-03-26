import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import math

#Helper function to plot experiment results 
def draw_plot(run_arr_dijkstra, run_arr_bellman, mean_dijkstra, mean_bellman, title):
    x = np.arange(len(run_arr_dijkstra))  # X-axis values based on number of trials

    fig = plt.figure(figsize=(20, 8))

    # Plot bars side by side
    width = 0.4  # Width of the bars
    plt.bar(x - width/2, run_arr_dijkstra, width=width, color="blue", alpha=0.6, label="Dijkstra’s")
    plt.bar(x + width/2, run_arr_bellman, width=width, color="green", alpha=0.6, label="Bellman-Ford")

    # Add horizontal lines for mean
    plt.axhline(mean_dijkstra, color="blue", linestyle="--", label=f"Dijkstra’s Avg = {mean_dijkstra:.3f} µs")
    plt.axhline(mean_bellman, color="green", linestyle="--", label=f"Bellman-Ford Avg = {mean_bellman:.3f} µs")

    # Adjust y-axis limits
    max_y = max(max(run_arr_dijkstra), max(run_arr_bellman))
    min_y = min(min(run_arr_dijkstra), min(run_arr_bellman))
    plt.ylim(min_y * 0.9, max_y * 1.1)  

    plt.xlabel("Number of Trials")
    plt.ylabel("Time taken (microseconds)")
    plt.title(title)
    plt.legend()
    plt.show()

# Directed Weighted Graph
class Graph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)

    def has_neg_cycle(self, nodes, src):
        dist = {node: float('inf') for node in self.adj}
        dist[src] = 0  
        
        for i in range(nodes):  
            for (u, v), w in self.weights.items():
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    if i == nodes -1:
                        return True
                    dist[v] = dist[u] + w
        
        return False
    
    def __str__(self):
        graph_str = ""
        for node in self.adj:
            graph_str += f"Node {node}: "
            for neighbor in self.adj[node]:
                graph_str += f"({neighbor}, weight: {self.weights[(node, neighbor)]}) "
            graph_str += "\n"
        return graph_str

# Helper Class for Heap
class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value

    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"

# Min Heap Class for Dijkstra
class Heap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)

        # add a map based on input node
        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

        self.build_heap()

    def find_left_index(self, index):
        return 2 * (index + 1) - 1

    def find_right_index(self, index):
        return 2 * (index + 1)

    def find_parent_index(self, index):
        return (index + 1) // 2 - 1

    def heapify(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[
            index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[
            smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]

            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # recursive call
            self.heapify(smallest_known_index)

    def build_heap(self, ):
        for i in range(self.length // 2 - 1, -1, -1):
            self.heapify(i)

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):

        while index > 0 and self.items[self.find_parent_index(index)].key < self.items[
            self.find_parent_index(index)].key:
            # swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], \
            self.items[index]
            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self, ):
        # xchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        # update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.heapify(0)
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]

    def is_empty(self):
        return self.length == 0

    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.items[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s

def dijkstra(graph, source, k):
    dist, path, shortest_paths, relax_count = {}, {}, {}, {}
    
    #Checking if source node exists in the graph
    if source not in graph.adj:
        return {}

    #Initializing all nodes distances to infinity and relaxation counter to 0
    for node in graph.adj: 
        dist[node] = float('inf') 
        path[node] = []
        relax_count[node] = 0
    
    dist[source] = 0
    path[source] = [source]

    #Creating initial heap from the list of items
    items = [Item(node, dist[node]) for node in graph.adj]
    min_heap = Heap(items)
    min_heap.decrease_key(source, 0)

    while not min_heap.is_empty():
        #Extracting the node with min distance 
        current_item = min_heap.extract_min()
        current_node = current_item.value
        current_dist = current_item.key

        #Skipping node if already relaxed k times or has dist == inf
        if current_dist == float('inf') or relax_count[current_node] >= k:
            continue

        #Relaxing its neighbours 
        for neighbour in graph.adj[current_node]:
            if relax_count[neighbour] >= k:
                continue

            #Calculating the new distance and updating if smaller 
            new_dist = current_dist + graph.w(current_node, neighbour)
            if new_dist < dist[neighbour]:
                dist[neighbour] = new_dist
                path[neighbour] = path[current_node] + [neighbour]

                #Decreasing key of the neighbour if its in the heap
                if neighbour in min_heap.map:
                    min_heap.decrease_key(neighbour, new_dist)

                #Increasing the neigbours relaxation count
                relax_count[neighbour] += 1

    #Adding the shortest distance and path for each node to one dictionary 
    for node in graph.adj:
        shortest_paths[node] = [ dist[node], path[node] ]

    return shortest_paths

# Bellman ford algorithm with k
# does not check for negative cycles
def bellman_ford(graph, source, k):
    result = {node: [float('inf'), []] for node in graph.adj}
    result[source] = [0, [source]]

    for _ in range(k):
        updated = False
        tmp_result = result.copy()

        for (u, v), w in graph.weights.items():
            if result[u][0] != float('inf') and result[u][0] + w < tmp_result[v][0]:
                tmp_result[v] = [result[u][0] + w, result[u][1] + [v]]
                updated = True

        result = tmp_result

        if not updated:
            break

    return result

def create_random_graph(nodes, edges, src, neg = False):
    graph = Graph()

    for node in range(nodes):
        graph.add_node(node)

    added_edges = set()

    # Ensure node 0 is connected to a random node
    if nodes > 1:  # Ensure there are other nodes to connect to
        node2 = random.randint(1, nodes - 1)  # Pick a random node from 1 to nodes-1
        weight = random.randint(-10, 10) if neg else random.randint(1, 10)
        graph.add_edge(0, node2, weight)
        added_edges.add((0, node2))

    # create a spanning tree
    for i in range(1, nodes):
        node1 = i
        node2 = random.randint(0, i - 1)  # connect node to any previously added node
        weight = random.randint(-10, 10) if neg else random.randint(1, 10)
        graph.add_edge(node1, node2, weight)
        added_edges.add((node1, node2))

    # add any random edges
    while len(added_edges) < edges:
        node1 = random.randint(0, nodes - 1)
        node2 = random.randint(0, nodes - 1)

        # ensure no self-loops + no duplicate edges
        if node1 != node2 and (node1, node2) not in added_edges:
            weight = random.randint(-10, 10) if neg else random.randint(1, 10)
            graph.add_edge(node1, node2, weight)

            if graph.has_neg_cycle(nodes, src):
                graph.adj[node1].remove(node2)
            else:
                added_edges.add((node1, node2))

    return graph

# test random graph for bellman
# print(create_random_graph(10,9, True)) 
# test for dijkstra
# print(create_random_graph(10,9)) 



def part2_experiment(node_vals, edge_vals, k_vals, title):
    trials = 40

    dijk_times, bell_times = [], []

    for n, e, k in zip(node_vals, edge_vals, k_vals):
        dijk_trial_times = []
        bell_trial_times = []

        for _ in range(trials):
            #generate graphs (for dijk and bell)
            #pick source node 

            start = time.time()
            dijkstra(graph, src, k)
            dijk_trial_times.append((time.time() - start) * 1e6)  # Convert to microseconds

            start = time.time()
            bellman_ford(graph, source, k)
            bell_trial_times.append((time.time() - start) * 1e6) 

        dijk_times.append(np.mean(dijk_trial_times))
        bell_times.append(np.mean(bell_trial_times))
        
        

        #Graph plots both graphs simultanously (in diff colours ) -> idk what values we should have in the legend or have one at all 
        draw_plot(dijk_times, bell_times, avg_dijk,avg_bell, f"Dijkstra's Vs Bellman Ford's Algorithms to test {title}")



    
    return


#Size vs density
part2_experiment([5, 10, 50, 100, 150], [5, 10, 50, 100, 150],  [ 6, 10, 50, 100, 150] , "Size vs Density" ) #density N sparse
part2_experiment([5, 10, 50, 100, 150], [15, 45, 1225, 4950, 11175 ],  [ 6, 10, 50, 100, 150] , "Size vs Density" ) #density medium
part2_experiment([5, 10, 50, 100, 150], [30, 90, 2450, 9900, 22,350 ],  [ 6, 10, 50, 100, 150] , "Size vs Density" ) #density (N(N-1)) dense

#K vs density
part2_experiment( [ 20, 20, 20], [20, 190, 380], [5]*3, "K value vs Density" ) #k = 5
part2_experiment( [ 20, 20, 20], [20, 190, 380], [10]*3, "K value vs Density" ) #k = 10
part2_experiment( [ 20, 20, 20], [20, 190, 380], [15]*3, "K value vs Density" ) #k = 15
part2_experiment( [ 20, 20, 20], [20, 190, 380], [15]*3, "K value vs Density" ) #k = 20


#K vs Size
part2_experiment( [ 5, 10, 20, 50, 100] , [10, 45, 190, 1225, 4950 ], [5]*5, "K value vs Size" ) #k = 5
part2_experiment( [ 5, 10, 20, 50, 100] , [10, 45, 190, 1225, 4950 ], [10]*5, "K value vs Size" ) #k = 10
part2_experiment( [ 5, 10, 20, 50, 100] , [10, 45, 190, 1225, 4950 ], [15]*5, "K value vs Size"  ) #k = 15
part2_experiment( [ 5, 10, 20, 50, 100] , [10, 45, 190, 1225, 4950 ], [20]*5, "K value vs Size"  ) #k = 20







