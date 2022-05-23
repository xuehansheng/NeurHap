# !/usr/bin/env python
# -*- coding: utf8 -*-

import os,torch,sys
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from torch.autograd import Variable
from torch.nn import functional as F
from loader import recon_haplotype,MEC
from utils import get_device_id,CSInstance


class Modeling:
	def __init__(self, constraints, args):
		self.args = args
		self.feature_size = args.feature_size
		self.lr = args.learning_rate
		self.epochs = args.epochs
		self.iterations = args.t_max
		self.set_device()
		self.model = NeuralModel(constraints, args=self.args, device=self.device)
		self.model.to(self.device)

	def set_device(self):
		device_id = get_device_id(torch.cuda.is_available())
		self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

	def run(self, instance, SNVmatrix):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		best_conflict_ratio = 1.0
		best_t,best_conflicts = 0,0
		best_assignment = None
		best_mec = 100000
		for epoch in range(self.epochs):
			self.model.train()
			optimizer.zero_grad()

			loss,conflict_ratio,_,_,assignment,n_conflicts = self.model(instance, self.iterations)
			haplotypes = recon_haplotype(assignment.cpu().detach().numpy(), SNVmatrix, self.args.n_colors)
			mec = MEC(SNVmatrix, haplotypes)
			loss_all = loss
			if epoch+1 == 1 or (epoch+1)%100 == 0:
				print("Epoch: {:d} loss={:.5f}, Ratio of violated constraints={:.4f}, MEC: {:d}".format(epoch+1,loss_all.item(),conflict_ratio,mec))

			if conflict_ratio < best_conflict_ratio or mec < best_mec:
				torch.save(self.model.state_dict(), 'models/best_model.pkl')
				best_t,best_conflict_ratio = epoch,conflict_ratio
				best_assignment = assignment
				best_conflicts = n_conflicts
				best_mec = mec

			loss_all.backward()
			optimizer.step()
		print('Saving {:d}-th epoch, conflict ratio: {:.4f} ({:d}).'.format(best_t+1, best_conflict_ratio, int(best_conflicts/2)))
		print("### Optimization Finished!")
		print('### Best MEC: %d' % best_mec)
		return best_assignment.cpu().detach().numpy()

	def evaluate(self, SNPMat, assignment):
		print('### Evaluating performance')
		pred_labels = np.array(assignment)
		ploidy = self.args.n_colors
		pred_haplotypes = recon_haplotype(pred_labels, SNPMat, ploidy)
		mec = MEC(SNPMat, pred_haplotypes)
		print('### MEC: %d' % mec)

	def refinement(self, SNPMat, assignment, negGraph, n_colors):
		n_nodes = assignment.shape[0]
		COLORS = [0,1,2,3]
		TAG = True
		negNeighbors = dict()
		for val in range(n_nodes):
			neighbors = list(negGraph.neighbors(val))
			negNeighbors[val]=neighbors

		pred_haplotypes = recon_haplotype(assignment, SNPMat, self.args.n_colors)
		best_mec = MEC(SNPMat, pred_haplotypes)
		_ASSIGNMENT = assignment.copy()

		while TAG==True:
			TAG = False
			for link in negGraph.edges():
				nodeA,nodeB = link[0],link[1]
				colorA,colorB = assignment[nodeA],assignment[nodeB]
				if colorA == colorB:
					_TMP_Assign = _ASSIGNMENT.copy()
					neigA,neigB = negNeighbors[nodeA],negNeighbors[nodeB]
					neigAcolors = list([assignment[v] for v in neigA])
					neigBcolors = list([assignment[v] for v in neigB])
					_CNTneigAcolors,_CNTneigBcolors = Counter(neigAcolors),Counter(neigBcolors)
					CNTneigAcolors,CNTneigBcolors = [_CNTneigAcolors[k] for k in range(len(_CNTneigAcolors))],[_CNTneigBcolors[k] for k in range(len(_CNTneigBcolors))]
					if colorA in range(len(CNTneigAcolors)) and colorB in range(len(CNTneigBcolors)):
						CNTcolorA,CNTcolorB = CNTneigAcolors[colorA],CNTneigBcolors[colorB]
						CNTcurrentConflict = CNTcolorA+CNTcolorB
						newColorA,newColorB = colorA,colorB
						for i in range(len(CNTneigAcolors)):
							for j in range(len(CNTneigBcolors)):
								if CNTneigAcolors[i]+CNTneigBcolors[j]<CNTcurrentConflict:
									CNTcurrentConflict = CNTneigAcolors[i]+CNTneigBcolors[j]
									newColorA,newColorB=i,j
						_TMP_Assign[nodeA] = newColorA
						_TMP_Assign[nodeB] = newColorB
						pred_haplotypes = recon_haplotype(_TMP_Assign, SNPMat, self.args.n_colors)
						mec = MEC(SNPMat, pred_haplotypes)
						if mec < best_mec:
							_ASSIGNMENT = _TMP_Assign
							best_mec = mec
							TAG = True
							print(best_mec)
		return _ASSIGNMENT


	def refine(self, SNPMat, assignment, negGraph, n_colors):
		TAG = True
		COLORS = set([0,1,2,3])
		n_nodes = assignment.shape[0]
		NODES = negGraph.nodes()
		NEIGHBORS = dict()
		for val in range(n_nodes):
			neighbor = list(negGraph.neighbors(val))
			NEIGHBORS[val]=neighbor

		pred_haplotypes = recon_haplotype(assignment, SNPMat, self.args.n_colors)
		best_mec = MEC(SNPMat, pred_haplotypes)
		# print('Best MEC score:', best_mec)
		_ASSIGNMENT = assignment.copy()

		while TAG==True:
			TAG = False
			for node in NODES:
				colorNODE = assignment[node]
				neighbors = NEIGHBORS[node]
				neigCOLORS = set([assignment[v] for v in neighbors])
				_colors = list(COLORS - neigCOLORS)
				if _colors != []:
					for i in range(len(_colors)):
						if _colors[i] != colorNODE:
							# print(_colors,colorNODE)
							_TMP_Assign = _ASSIGNMENT.copy()
							_TMP_Assign[node] = _colors[i]
							pred_haplotypes = recon_haplotype(_TMP_Assign, SNPMat, self.args.n_colors)
							mec = MEC(SNPMat, pred_haplotypes)
							if mec < best_mec:
								_ASSIGNMENT = _TMP_Assign
								best_mec  = mec
								TAG = True
								# print('New MEC score:', best_mec)
		return _ASSIGNMENT


class NeuralModel(nn.Module):
	def __init__(self, constraints, args, device=None):
		super(NeuralModel, self).__init__()
		self.args = args
		self.device = device
		self.constraints = constraints
		self.iterations = self.args.t_max #30
		self.domain_size = constraints.domain_size #4
		self.relations_matrices = constraints.relation_matrices #[4x4]
		self.confidence_matrices = constraints.confidence_matrices

		self.NeuralModelCell = NeuralModelCell(self.args.feature_size, self.domain_size, device=self.device)
		self.NeuralModelCell.to(self.device)
		self.decoder = nn.Linear(in_features=self.args.feature_size*self.iterations,out_features=self.domain_size)
		
	def forward(self, instance, iterations):
		self.relation_tensors = torch.Tensor(self.relations_matrices).to(self.device) #relations:[4x4]
		self.confidence_tensors = torch.Tensor(self.confidence_matrices).to(self.device)
		self.conflicts = torch.from_numpy(instance.conflicts).to(self.device) #edges:[46x2]
		self.confidence = torch.from_numpy(instance.confidence).to(self.device)
		self.adj = torch.Tensor(instance.adj).to(self.device)

		# conflicts
		self.idx_left = torch.reshape(self.conflicts[:, 0], [-1, 1]).type(torch.long) #[3351,1]
		self.idx_right = torch.reshape(self.conflicts[:, 1], [-1, 1]).type(torch.long) #[3351,1]

		# confidence
		self.idx_left_pos = torch.reshape(self.confidence[:, 0], [-1, 1]).type(torch.long) #[1019,1]
		self.idx_right_pos = torch.reshape(self.confidence[:, 1], [-1, 1]).type(torch.long) #[1019,1]

		self.degrees = torch.tensor(instance.degrees).to(self.device) #tensor([5, 5, 4, 6, 7, 7, 5, 6, 5, 4, 4, 6, 5, 5, 4, 5, 4, 5], dtype=torch.int32)
		self.n_variables = torch.tensor(instance.n_variables).to(self.device) #18
		self.n_conflicts = torch.tensor(instance.n_conflicts).to(self.device) #46
		self.n_confidence = torch.tensor(instance.n_confidence).to(self.device) #46

		### INITIAL LEARNABLE FEATURES
		state = nn.Parameter(F.normalize(torch.rand([self.n_variables,self.args.feature_size]), dim=1), requires_grad=True).to(self.device)

		logits,states = [],[]
		for t in range(iterations):
			logit,state = self.NeuralModelCell(state,instance.n_variables,self.degrees,self.idx_left,self.idx_right, self.adj)
			logits.append(logit) #torch.Size([18, 4])
			states.append(state)
		_logits = torch.stack(logits).transpose(1,0)
		# print(_logits.shape, self.states.shape) #torch.Size([398, 20, 4]) torch.Size([398, 20, 32])
		self.phi = nn.Softmax(dim=2)(_logits) #torch.Size([18, 30, 4])

		loss = self.build_loss()
		conflict_ratio,edge_conflicts,assignment,n_conflicts = self.build_predictions()
		return loss, conflict_ratio, self.phi, edge_conflicts,assignment, n_conflicts

	def build_loss(self):
		all_phi = torch.reshape(self.phi, (-1,self.iterations*self.domain_size)) #torch.Size([18, 120])

		phi_left = torch.reshape(all_phi[self.idx_left.squeeze(1),:],(-1,self.domain_size)) #torch.Size([1380, 4])
		phi_right = torch.reshape(all_phi[self.idx_right.squeeze(1),:],(-1,self.domain_size)) #torch.Size([1380, 4])

		conflicts_relation_loss = torch.sum(torch.matmul(phi_left, self.relation_tensors)*phi_right, axis=1)
	
		relation_loss = -torch.log(conflicts_relation_loss) #torch.Size([30, 46])
		relation_loss = torch.sum(relation_loss, axis=0) #torch.Size([30])		
		loss_conflict = relation_loss / self.n_conflicts #torch.Size([30])
		loss_conflict = torch.sum(loss_conflict)

		phi_left_pos = torch.reshape(all_phi[self.idx_left_pos.squeeze(1),:],(-1,self.domain_size)) #torch.Size([1380, 4])
		phi_right_pos = torch.reshape(all_phi[self.idx_right_pos.squeeze(1),:],(-1,self.domain_size)) #torch.Size([1380, 4])
		confidence_relation_loss = torch.sum(torch.matmul(phi_left_pos, self.confidence_tensors)*phi_right_pos, axis=1)
		confidence_loss = -torch.log(confidence_relation_loss) #torch.Size([30, 46])
		confidence_loss = torch.sum(confidence_loss, axis=0) #torch.Size([30])
		loss_confidence = confidence_loss / (10*self.n_confidence) #torch.Size([30])
		loss_confidence = torch.sum(loss_confidence)

		loss = loss_conflict + self.args.lamb*loss_confidence
		return loss

	def build_predictions(self):
		self.assignment = torch.argmax(self.phi, dim=2).int() #torch.Size([18, 30])
		assignment = self.assignment.unsqueeze(-1)
		val_left = assignment[self.idx_left.squeeze(1)] #torch.Size([46, 30, 1])
		val_right = assignment[self.idx_right.squeeze(1)] #torch.Size([46, 30, 1]) 
		valid = self.relation_tensors[val_left.type(torch.long),val_right.type(torch.long)].squeeze() #torch.Size([8260, 30])

		conflicts = 1.0 - valid
		n_conflicts = torch.sum(conflicts[:, -1])
		self.conflict_ratio = n_conflicts / self.n_conflicts
		# self.conflict_ratio = n_conflicts / self.n_confidence
		self.edge_conflicts = conflicts
		return self.conflict_ratio, self.edge_conflicts, self.assignment[:,-1], n_conflicts


class NeuralModelCell(nn.Module):
	def __init__(self, feature_size, domain_size, bias=True, device=None):
		super(NeuralModelCell, self).__init__()
		self.device = device
		self.feature_size = feature_size
		self.domain_size = domain_size
		self.normalize = nn.BatchNorm1d(num_features=feature_size)
		self.updater = nn.RNNCell(feature_size, feature_size)
		# self.updater = nn.GRUCell(feature_size, feature_size)
		self.decoder = nn.Linear(in_features=feature_size,out_features=domain_size)
		self.mpnn = MPNNLayer(feature_size)
		self.fusion = nn.Linear(in_features=2*feature_size,out_features=feature_size)
		self.linear = nn.Linear(in_features=feature_size,out_features=feature_size)

	def forward(self, states, n_variables, degrees, idx_left, idx_right, adj):
		conflicts_in_left = states[idx_left,:].squeeze(1) #[4971, 32]
		conflicts_in_right = states[idx_right,:].squeeze(1) #[4971, 32]
		msg_left, msg_right = self.mpnn(conflicts_in_left, conflicts_in_right) #[4971,32]

		variable_in_left = variable_in_right = torch.zeros(n_variables, self.feature_size).to(self.device)
		variable_in_left.index_add_(0, idx_left.squeeze(1), msg_left)
		variable_in_right.index_add_(0, idx_right.squeeze(1), msg_right)
		variable_in = variable_in_right + variable_in_left  #torch.Size([240, 32])

		rec = torch.div(variable_in, degrees.unsqueeze(1)) #[240, 32]
		rec = self.normalize(rec) #[240, 32]
		# states = self.updater(rec, states)
		states = rec
		logits = self.decoder(states) #[240, 4]
		return logits, states


class MPNNLayer(nn.Module):
	def __init__(self, feature_size):
		super(MPNNLayer, self).__init__()
		self.feature_size = feature_size
		self.MLP = nn.Sequential(
			nn.Linear(in_features=feature_size*2, out_features=feature_size),nn.ReLU(),
			nn.Linear(in_features=feature_size, out_features=feature_size),nn.ReLU(),
			nn.LayerNorm(feature_size))

	def forward(self, in_right, in_left):
		in_lr = torch.cat((in_left,in_right), axis=1)
		in_rl = torch.cat((in_right,in_left), axis=1)
		msg = torch.cat((in_lr,in_rl), axis=0)

		h = self.MLP(msg)
		n_edges = in_right.shape[0]
		msg_left = h[:n_edges, :]
		msg_right = h[n_edges:, :]
		return msg_left, msg_right

