# !/usr/bin/env python
# -*- coding: utf8 -*-

import heapq
import statistics
import os,itertools,random,sys
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter


### LOAD PART
def load_SNP_Fragment_Matrix(args):
	filepath = 'data/'+args.data+'/Cov5/K4_Cov5_'+args.sample+'_SNV_matrix.txt'
	print(filepath)
	snp_mat = np.loadtxt(filepath)
	return snp_mat


def extract_NegEdges(snp_mat, thresA, thresB):
	def hamming_distance(read, haplo):
		overlap = (read!=0)&(haplo!=0)
		CNT = np.sum(overlap==True)
		if CNT >= 1:
			return sum((haplo-read)[overlap]!=0)
		else:
			return -1

	distances,pairs = [],[]
	num_reads,len_haplo = snp_mat.shape
	for i in range(num_reads):
		for j in range(i+1, num_reads):
			dis = hamming_distance(snp_mat[i],snp_mat[j])
			distances.append(dis)
			pairs.append([i,j])

	negEDGs = []
	for idx,val in enumerate(distances):
		if val>=thresA and val<thresB:
			negEDGs.append(pairs[idx])
	print("Edges in negGraph: %d" % len(negEDGs))
	return negEDGs


def construct_Graph(snp_mat, thres, ovlps):
	def hamming_distance(read, haplo):
		overlap = (read!=0)&(haplo!=0)
		CNT = np.sum(overlap==True)
		# return sum((haplo-read)[overlap]!=0),CNT if CNT>=1 else -1,CNT
		if CNT >= 1:
			return sum((haplo-read)[overlap]!=0),CNT
		else:
			return -1,CNT

	distances,pairs,overlaps = [],[],[]
	num_reads,len_haplo = snp_mat.shape
	for i in range(num_reads):
		for j in range(i+1, num_reads):
			dis,olp = hamming_distance(snp_mat[i],snp_mat[j])
			distances.append(dis)
			pairs.append([i,j])
			overlaps.append(olp)

	posEDGs,negEDGs = [],[]
	for idx,val in enumerate(distances):
		olp = overlaps[idx]
		if val>=thres:
			negEDGs.append(pairs[idx])
		if val==0 and olp>=ovlps:
			posEDGs.append(pairs[idx])
	print("posEDGs: %d, negEDGs: %d" % (len(posEDGs),len(negEDGs)))

	negGx = nx.Graph()
	negGx.add_nodes_from([val for val in range(num_reads)])
	negGx.add_edges_from(negEDGs)
	# negGx.add_weighted_edges_from(negEDGs)
	# print(nx.number_of_isolates(negGx)) #, list(nx.isolates(negGx)))
	# print("Conflict Graph: nodes: {:d}, edges: {:d}.".format(negGx.number_of_nodes(),negGx.number_of_edges()))

	posGx = nx.Graph()
	posGx.add_nodes_from([val for val in range(num_reads)])
	posGx.add_edges_from(posEDGs)
	# print(nx.number_of_isolates(posGx)) #, list(nx.isolates(posGx)))
	# print("Homophilic Graph: nodes: {:d}, edges: {:d}.".format(posGx.number_of_nodes(),posGx.number_of_edges()))
	return posGx,negGx

def refine_assignment(assignment, nodeMaps):
	n_nodes = max([v for k,v in nodeMaps.items()])+1
	assignment_new = np.zeros(n_nodes)
	for i in range(assignment.shape[0]):
		idx = nodeMaps[i]
		assignment_new[idx] = assignment[i]
	return assignment_new


def recon_haplotype(origins, SNVmatrix, n_cluster):
	def ACGT_count(submatrix):
		out = np.zeros((submatrix.shape[1], 4))
		for i in range(4):
			out[:, i] = (submatrix == (i + 1)).sum(axis = 0)
		return out
	V_major = np.zeros((n_cluster, SNVmatrix.shape[1])) #majority voting result
	ACGTcount = ACGT_count(SNVmatrix)
	
	for i in range(n_cluster):
		reads_single = SNVmatrix[origins == i, :] #all reads from one haplotypes
		single_sta = np.zeros((SNVmatrix.shape[1], 4))
		if len(reads_single) != 0:
			single_sta = ACGT_count(reads_single) #ACGT statistics of a single nucleotide position
		V_major[i, :] = np.argmax(single_sta, axis = 1) + 1

		uncov_pos = np.where(np.sum(single_sta, axis = 1) == 0)[0]
		for j in range(len(uncov_pos)):
			if len(np.where(ACGTcount[uncov_pos[j], :] == max(ACGTcount[uncov_pos[j], :]))[0]) != 1: #if not covered, select the most doninant one based on 'ACGTcount'
				tem = np.where(ACGTcount[uncov_pos[j], :] == max(ACGTcount[uncov_pos[j], :]))[0]
				V_major[i, uncov_pos[j]] = tem[int(np.floor(random.random() * len(tem)))] + 1
			else:
				V_major[i, uncov_pos[j]] = np.argmax(ACGTcount[uncov_pos[j], :]) + 1
	return V_major

def MEC(SNVmatrix, Recovered_Haplo):
	def _hamming_distance(read, haplo):
		return sum((haplo - read)[np.where(read != 0)] != 0)
	res = 0
	for i in range(len(SNVmatrix)):
		dis = [_hamming_distance(SNVmatrix[i, :], Recovered_Haplo[j, :]) for j in range(len(Recovered_Haplo))]
		res += min(dis)
	return res

