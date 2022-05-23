# !/usr/bin/env python
# -*- coding: utf8 -*-

import argparse,sys
from utils import *
from loader import *
from model import Modeling

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--epochs', type=int, default=2000, help='Number of training epochs')
	parser.add_argument('-r', '--learning_rate', type=float, default=1e-3, help='Number of learning rate.')
	parser.add_argument('-t', '--t_max', type=int, default=10, help='Number of iterations t_max for which RUN-CSP runs on each instance')
	parser.add_argument('-f', '--feature_size', type=int, default=32, help='Feature size for training')
	parser.add_argument('-m', '--model_dir', type=str, default='models', help='Model directory in which the trained model is stored')
	parser.add_argument('-d', '--data', default='Semi-Potato', help='Dataset name')
	parser.add_argument('-s', '--sample', type=str, default='Sample1', help='Sample name')
	parser.add_argument('-k', '--n_colors', type=int, default=4, help='Number of colors')
	parser.add_argument('-p', '--overlap', type=int, default=6, help='Number of overlaps for consistent graphs')
	parser.add_argument('-q', '--threshold', type=int, default=2, help='Threshold to construct conflict graphs')
	parser.add_argument('-l', '--lamb', type=float, default=0.01, help='Weight in the loss function.')
	args = parser.parse_args()
	print(args)
	return args

def run(args):
	print('### Loading graphs...')
	SNPMat = load_SNP_Fragment_Matrix(args)
	print(SNPMat.shape)
	posGraph,negGraph = construct_Graph(SNPMat, thres=args.threshold, ovlps=args.overlap)

	print('### Generating constraints...')
	constraints = Constraints.get_colorings(args.n_colors)
	print('### Converting graphs to CS-Instances') #constraint satisfaction problem
	instances = CSInstance.graph_to_csinstance(negGraph, posGraph, constraints)

	model = Modeling(constraints=constraints, args=args)
	assignment = model.run(instances, SNPMat)
	model.evaluate(SNPMat, assignment)

	# np.save(args.sample+'.npy', assignment)
	# assignment = np.load(args.sample+'.npy')
	# model.evaluate(SNPMat, assignment)

	# assignment = model.refinement(SNPMat, assignment, negGraph, args.n_colors)
	assignment = model.refine(SNPMat, assignment, negGraph, args.n_colors)
	model.evaluate(SNPMat, assignment)
	# eval_AgainstNegEDGs(assignment, negGraph)


if __name__ == '__main__':
	run(parser())
