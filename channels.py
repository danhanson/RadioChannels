#!/usr/bin/env python

from matplotlib import pyplot
from math import sqrt

class Node:

	def __init__(self):
		self.left = None
		self.right = None
		self.topLeft = None
		self.topRight = None
		self.botLeft = None
		self.botRight = None
		self.channel = None
		self._neighbor_queries = dict()

	def neighbors(self,s):
		if s in self._neighbor_queries:
			return self._neighbor_queries[s]

		from collections import deque
		node = self
		queue = deque()
		neighbors = set()
		def addChildren(dist,parent):
			queue.extend((dist,node) for node in [
					parent.right,
					parent.topRight,
					parent.topLeft,
					parent.left,
					parent.botLeft,
					parent.botRight
				] if node != None and node not in neighbors and node != self
			)
		addChildren(1,self)
		while(len(queue) > 0):
			dist, parent = queue.popleft()
			if parent in neighbors or parent == self:
				continue
			neighbors.add(parent)
			if(dist < s):
				addChildren(dist+1,parent)
		ret = frozenset(neighbors)
		self._neighbor_queries[s] = ret
		return ret

class HexGraph:

	def __init__(self,size):
		self.center = Node()
		border = [self.center]
		for ring in xrange(1,size):
			nextBorder = [Node() for i in xrange(ring*6)]

			# add right-most node
			nextBorder[0].left = border[0]
			border[0].right = nextBorder[0]

			nextBorder[0].botLeft = nextBorder[-1]
			nextBorder[-1].topRight = nextBorder[0]

			# add top right nodes
			for i in xrange(1,ring):
				nextBorder[i].botRight = nextBorder[i-1]
				nextBorder[i-1].topLeft = nextBorder[i]

				nextBorder[i].left = border[i]
				border[i].right = nextBorder[i]

				nextBorder[i].botLeft = border[i-1]
				border[i-1].topRight = nextBorder[i]

			# add top right corner node
			nextBorder[ring].botRight = nextBorder[ring-1]
			nextBorder[ring-1].topLeft = nextBorder[ring]

			nextBorder[ring].botLeft = border[ring-1]
			border[ring-1].topRight = nextBorder[ring]

			# add top nodes
			for i in xrange(ring+1,2*ring):
				nextBorder[i].right = nextBorder[i-1]
				nextBorder[i-1].left = nextBorder[i]

				nextBorder[i].botRight = border[i-2]
				border[i-2].topLeft = nextBorder[i]
				
				nextBorder[i].botLeft = border[i-1]
				border[i-1].topRight = nextBorder[i]

			# add top left corner node
			nextBorder[2*ring].right = nextBorder[2*ring-1]
			nextBorder[2*ring-1].left = nextBorder[2*ring]

			nextBorder[2*ring].botRight = border[2*ring-2]
			border[2*ring-2].topLeft = nextBorder[2*ring]

			# add top left nodes
			for i in xrange(2*ring+1,3*ring):
				nextBorder[i].topRight = nextBorder[i-1]
				nextBorder[i-1].botLeft = nextBorder[i]

				nextBorder[i].right = border[i-3]
				border[i-3].left = nextBorder[i]

				nextBorder[i].botRight = border[i-2]
				border[i-2].topLeft = nextBorder[i]

			# add left corner node
			nextBorder[3*ring].topRight = nextBorder[3*ring-1]
			nextBorder[3*ring-1].botLeft = nextBorder[3*ring]

			nextBorder[3*ring].right = border[3*ring-3]
			border[3*ring-3].left = nextBorder[3*ring]

			# add bottom left nodes
			for i in xrange(3*ring+1,4*ring):
				nextBorder[i].topLeft = nextBorder[i-1]
				nextBorder[i-1].botRight = nextBorder[i]

				nextBorder[i].topRight = border[i-4]
				border[i-4].botLeft = nextBorder[i]

				nextBorder[i].right = border[i-3]
				border[i-3].left = nextBorder[i]

			# add bottom left corner
			nextBorder[4*ring].topLeft = nextBorder[4*ring-1]
			nextBorder[4*ring-1].botRight = nextBorder[4*ring]

			nextBorder[4*ring].topRight = border[4*ring-4]
			border[4*ring-4].botLeft = nextBorder[4*ring]

			# add bottom nodes
			for i in xrange(4*ring+1,5*ring):
				nextBorder[i].left = nextBorder[i-1]
				nextBorder[i-1].right = nextBorder[i]
				
				nextBorder[i].topLeft = border[i-5]
				border[i-5].botRight = nextBorder[i]

				nextBorder[i].topRight = border[i-4]
				border[i-4].botLeft = nextBorder[i]

			# add bottom right corner
			nextBorder[5*ring].left = nextBorder[5*ring-1]
			nextBorder[5*ring-1].right = nextBorder[5*ring]

			nextBorder[5*ring].topLeft = border[5*ring-5]
			border[5*ring-5].botRight = nextBorder[5*ring]

			# add bottom right nodes
			for i in xrange(5*ring+1,6*ring):
				nextBorder[i].botLeft = nextBorder[i-1]
				nextBorder[i-1].topRight = nextBorder[i]

				nextBorder[i].left = border[i-6]
				border[i-6].right = nextBorder[i]

				nextBorder[i].topLeft = border[(i-5)%(6*(ring-1))]
				border[(i-5)%(6*(ring-1))].botRight = nextBorder[i]

			border = nextBorder

	def plot(self, plot, midX=0, midY=0):
		node = self.center
		coord = (midX,midY)
		while(node.left != None):
			node = node.topLeft
			coord = (coord[0]-1,coord[1]+0.5+sqrt(1.25))

		# the loop traverses the graph by going left to right on each row starting at the top row
		stack = [(coord,node)]
		while(len(stack) > 0):
			coord, node = stack.pop()
			x,y = coord
			plot.text(x,y,str(node.channel),horizontalalignment='center',verticalalignment='center')

			# plot hexagon
			xs = [x-1,x,x+1,x+1,x,x-1,x-1]
			ys = [y-0.5,y-sqrt(1.25),y-0.5,y+0.5,y+sqrt(1.25),y+0.5,y-0.5]
			plot.plot(xs,ys,'-')

			# add next row
			if(node.left == None):
				if(node.botLeft != None):
					stack.append(((x-1,y-0.5-sqrt(1.25)),node.botLeft))
				elif(node.botRight != None):
					stack.append(((x+1,y-0.5-sqrt(1.25)),node.botRight))

			# add nodes to the right
			if(node.right != None):
				stack.append(((x+2,y),node.right))

	def nodes(self):
		node = self.center
		while(node.left != None):
			node = node.topLeft

		# the loop traverses the graph by going left to right on each row starting at the top row
		stack = [node]
		while(len(stack) > 0):
			node = stack.pop()
			if(node.left == None):
				if(node.botLeft != None):
					stack.append(node.botLeft)
				elif(node.botRight != None):
					stack.append(node.botRight)

			if(node.right != None):
				stack.append(node.right)
			yield node

	def computeChannels(self, upperBound, optimize=False):
		from constraint import Problem
		from constraint import SomeInSetConstraint
		found = False
		nodes = tuple(self.nodes())
		while(True): # this loop lowers the upper bounds each iteration to find the optimal solution
			p = Problem()
			p.addVariables([n for n in self.nodes()], range(upperBound-1,-1,-1)) # reverse ordering the domain biases the constraint solver towards smaller numbers
			p.addConstraint(SomeInSetConstraint([0]))
			def addConstraint(node1, node2, dist, diff):
				if(node1 in node2.neighbors(dist)):
					p.addConstraint(lambda x, y: abs(x - y) >= diff,(node1, node2))
					return True
				return False

			for i in xrange(len(nodes)-1):
				n1 = nodes[i]
				for j in xrange(i+1,len(nodes)):
					n2 = nodes[j]
					if(not addConstraint(n1, n2, 2, 2)): # each node pair needs no more than 1 constraint
						addConstraint(n1, n2, 4, 1)
			solution = p.getSolution()
			if(solution == None):
				return found
			found = True
			for node,channel in p.getSolution().iteritems():
				node.channel = channel
			if(not optimize):
				break
			upperBound = min(upperBound-1,max(solution.values()))
		return True

def main():
	from sys import argv
	from matplotlib import pyplot
	size = int(argv[1])
	lub = int(argv[2])
	g = HexGraph(size)
	g.computeChannels(lub)
	g.plot(pyplot)
	pyplot.show()

if __name__ == "__main__":
	main()

