# !/usr/bin/env python
# -*- coding: utf8 -*-

import itertools
import subprocess
import numpy as np
import networkx as nx
from collections import Counter


def get_device_id(cuda_is_available):
	if not cuda_is_available:
		return -1
	gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode("utf-8")
	gpu_stats = gpu_stats.strip().split("\n")
	stats = []
	for i in range(1, len(gpu_stats)):
		info = gpu_stats[i].split()
		used = int(info[0])
		free = int(info[2])
		stats.append([used, free])
	stats = np.array(stats)
	gpu_index = stats[:, 1].argmax()
	available_mem_on_gpu = stats[gpu_index, 1]
	device_id = gpu_index if available_mem_on_gpu > 2000 else -1
	print("Automatically selected device id %s (>= 0 for GPU, -1 for CPU)\n" % device_id)
	return device_id

class Constraints:
	""" Class to represent a fixed Constraint Representation """
	def __init__(self, domain_size, relations):
		self.domain_size = domain_size
		self.relations = relations
		self.relation_matrices = np.zeros((self.domain_size, self.domain_size), dtype=np.float32)
		idx = np.array(self.relations)
		self.relation_matrices[idx[:, 0], idx[:, 1]] = 1.0
		self.confidence_matrices = np.eye(self.domain_size, dtype=np.float32)

	@staticmethod
	def get_colorings(d):
		def get_relations(d):
			relations = [[i, j] for i in range(d) for j in range(d) if i!=j]
			return relations
		lang = Constraints(domain_size=d, relations=get_relations(d))
		return lang

class CSInstance:
	def __init__(self, adj, constraints, n_variables, conflicts, confidence):
		self.adj = adj
		self.constraints = constraints
		self.n_variables = n_variables
		self.conflicts = conflicts
		self.confidence = confidence

		all_conflicts = list(itertools.chain.from_iterable(conflicts)) #46
		variables, counts = np.unique(all_conflicts, return_counts=True) #nodes list, count
		# degrees = np.zeros(shape=(n_variables), dtype=np.int32)
		degrees = np.ones(shape=(n_variables), dtype=np.int32)
		for u, c in zip(variables, counts):
			degrees[u] = c
		self.degrees = degrees #[5 5 4 6 7 7 5 6 5 4 4 6 5 5 4 5 4 5]
		self.n_conflicts = len(all_conflicts) #46 number of edges
		# self.n_conflicts = len(conflicts)
		self.n_confidence = len(confidence)

	@staticmethod
	def graph_to_csinstance(negGraph, posGraph, constraints):
		adj = nx.adjacency_matrix(negGraph).todense() #sym
		n_variables = adj.shape[0] #50
		conflicts = np.int32(negGraph.edges()) #graph.edges()
		confidence = np.int32(posGraph.edges())

		instance = CSInstance(adj, constraints, n_variables, conflicts, confidence)
		return instance

	@staticmethod
	def merge(instances):
		adj = instances[0].adj
		constraints = instances[0].constraints
		conflicts = []
		n_variables = 0

		for instance in instances:
			shifted = instance.conflicts + n_variables
			conflicts.append(shifted)
			n_variables += instance.n_variables

		conflicts = np.vstack(conflicts)
		merged_instance = CSInstance(adj, constraints, n_variables, conflicts, instances[0].confidence)
		return merged_instance

	def count_conflicts(self, assignment):
		conflicts = 0
		matrices = self.constraints.relation_matrices
		valid = np.float32([matrices[assignment[u], assignment[v]] for [u, v] in self.conflicts])
		has_conflict = 1.0 - valid
		conflicts += np.sum(has_conflict)
		return int(conflicts)
