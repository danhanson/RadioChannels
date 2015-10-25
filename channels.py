#!/usr/bin/env python

from matplotlib import pyplot
from math import sqrt

class Node:

	def __init__(self):
		self._pos = None
		self.left = None
		self.right = None
		self.topLeft = None
		self.topRight = None
		self.botLeft = None
		self.botRight = None
		self.channel = None
		self._neighbor_queries = dict()

	def distance(self,node):
		return sqrt(
			(sqrt(0.75)*(self._pos[0] - node._pos[0]))**2 + 
			(1.5*(self._pos[1] - node._pos[1]))**2
		)

	def x(self):
		return self._pos[0]*sqrt(0.75)

	def y(self):
		return self._pos[1]*1.5

	def neighbors(self,s):
		if s in self._neighbor_queries:
			return self._neighbor_queries[s]

		from collections import deque
		node = self
		queue = deque()
		neighbors = set()
		def addChildren(parent):
			queue.extend(node for node in [
					parent.right,
					parent.topRight,
					parent.topLeft,
					parent.left,
					parent.botLeft,
					parent.botRight
				] if node != None and node not in neighbors and node != self and self.distance(node) <= s
			)
		addChildren(self)
		while(len(queue) > 0):
			parent = queue.popleft()
			if parent in neighbors or parent == self:
				continue
			neighbors.add(parent)
			addChildren(parent)
		ret = frozenset(neighbors)
		self._neighbor_queries[s] = ret
		return ret
	
	def plot(self,ax):
		from numpy import array
		from matplotlib.patches import Polygon
		from matplotlib import colors
		hsv = ((hash(hash(self.channel)) % 10)/10.0, 0.8, 0.8)
		color = colors.hsv_to_rgb(hsv)

		x = self.x()
		y = self.y()
		ax.text(x, y, self.channel, horizontalalignment='center', verticalalignment='center')


		left = x - sqrt(0.75)
		right = x + sqrt(0.75)

		points = [(left, y - 0.5), (x, y - 1), (right, y - 0.5), (right, y + 0.5), (x, y + 1), (left, y + 0.5)]
		p = Polygon(array(points), color=color)
		ax.add_patch(p)

class HexGraph:

	def __init__(self,size):
		self.center = Node()
		self.size = size
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
		self.assignPositions()

	def assignPositions(self):
		node = self.center
		coord = (0,0) # relative indices for hexagon positions
		while(node.left != None):
			node = node.topLeft
			coord = (coord[0]-1,coord[1]+1)

		# the loop traverses the graph by going left to right on each row starting at the top row
		node._pos = coord
		stack = [node]
		while(len(stack) > 0):
			node = stack.pop()
			x,y = node._pos

			# add next row
			if(node.left == None):
				if(node.botLeft != None):
					node.botLeft._pos = (x - 1, y - 1)
					stack.append(node.botLeft)
				elif(node.botRight != None):
					node.botRight._pos = (x + 1, y - 1)
					stack.append(node.botRight)

			# add nodes to the right
			if(node.right != None):
				node.right._pos = (x + 2, y)
				stack.append(node.right)


	def plot(self, plot):
		fig = plot.figure()
		ax = fig.add_subplot(111)
		ax.set_xlim(-(2*self.size+1)*sqrt(0.75),(2*self.size+1)*sqrt(0.75))
		ax.set_ylim(-(self.size+1)*1.5, (self.size+1)*1.5)

		for node in self.nodes():
			node.plot(ax)

	def rows(self):
		def row(node):
			while(True):
				yield node
				if(node.right == None):
					return
				node = node.right

		node = self.center
		while(node.left != None):
			node = node.topLeft

		while(True):
			yield row(node)
			if(node.botLeft != None):
				node = node.botLeft
			elif(node.botRight != None):
				node = node.botRight
			else:
				return
	
	def columns(self):
		def column(node):
			while(True):
				yield node
				if(node.botLeft == None):
					return
				node = node.botLeft
		node = self.center
		while(node.left != None):
			node = node.topLeft
		while(True):
			yield column(node)
			if(node.right != None):
				node = node.right
			else:
				return

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

	def computeChannels(self, n, k=2):
		from constraint import Problem
		from constraint import SomeInSetConstraint
		found = False
		nodes = tuple(self.nodes())
		p = Problem()
		p.addVariables(list(self.nodes()), range(n,0,-1)) # reverse ordering the domain biases the constraint solver towards smaller numbers
		p.addConstraint(SomeInSetConstraint([1]))
		def addConstraint(node1, node2, dist, diff):
			if(node1 in node2.neighbors(dist)):
				p.addConstraint(lambda x, y: abs(x - y) >= diff,(node1, node2))
				return True
			return False

		for i in xrange(len(nodes)-1):
			n1 = nodes[i]
			for j in xrange(i+1,len(nodes)):
				n2 = nodes[j]
				if(not addConstraint(n1, n2, 2, k)): # each node pair needs no more than 1 constraint
					addConstraint(n1, n2, 4, 1)
		for rowIter in self.rows():
			row = tuple(rowIter)
			for i in xrange(len(row)-1):
				p.addConstraint(lambda x, y: y == (x + k) % n + 1,(row[i], row[i+1]))
		for colIter in self.columns():
			col = tuple(colIter)
			for i in xrange(len(col)-1):
				p.addConstraint(lambda x, y: y == (x + k - 1) % n + 1,(col[i], col[i+1]))
		solution = p.getSolution()
		if(solution == None):
			return found
		found = True
		for node,channel in p.getSolution().iteritems():
			node.channel = channel
		return True

def main():
	from sys import argv
	from matplotlib import pyplot
	size = int(argv[1])
	lub = int(argv[2])
	k = int(argv[3])
	g = HexGraph(size)
	g.computeChannels(lub,k)
	g.plot(pyplot)
	pyplot.show()

if __name__ == "__main__":
	main()

