import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import math

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

class Queue:
    def __init__(self):
        self.q = []

def dijkstra(graph, source, k):

    dist, path = {}, {}
    relax_count = {}
    items = []

    #includes the shortest distance and path 
    shortest_paths = {}

    if source not in graph:
        return {}

    for node in graph: 
        dist[node] = float('inf') 
        path[node] = []
        relax_count[node] = 0
        #Adding each node and its distance to a list
        items.append(Item(dist[node], node))

    #Creating a min heap from the list of items 
    min_heap = Heap(items)

    #Initializing the source key distance to 0 and adding it to heap
    dist[source] = 0
    min_heap.decrease_key(source, 0)
    
    while not min_heap.is_empty():
        #extract ndoe with min distance 
        current_item = min_heap.extract_min()
        current_node = current_item.value
        current_dist = current_item.key

        #Skipping node if already relaxed k times 
        if relax_count[current_node] >= k:
            continue

        for neighbour in graph[current_node]:
            if relax_count[neighbour] >= k:
                continue

            #TO DO
            new_dist = 0

        return 


    return shortest_paths

def bellman_ford(graph, source, k):

    return

def part2_experiment():
    return

