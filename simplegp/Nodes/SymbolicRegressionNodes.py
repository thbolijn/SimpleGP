import numpy as np

from simplegp.Nodes.BaseNode import Node


class AddNode(Node):
	
	def __init__(self):
		super(AddNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '+'

	def GetOutput(self, X):
		X0 = self._children[0].GetOutput(X)
		X1 = self._children[1].GetOutput(X)
		return (X0 + X1) * self.weights[0] + self.weights[1]


class SubNode(Node):
	def __init__(self):
		super(SubNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '-'

	def GetOutput(self, X):
		X0 = self._children[0].GetOutput(X)
		X1 = self._children[1].GetOutput(X)
		return (X0 - X1) * self.weights[0] + self.weights[1]


class MulNode(Node):
	def __init__(self):
		super(MulNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '*'

	def GetOutput(self, X):
		X0 = self._children[0].GetOutput(X)
		X1 = self._children[1].GetOutput(X)
		return (np.multiply(X0, X1)) * self.weights[0] + self.weights[1]


class DivNode(Node):
	def __init__(self):
		super(DivNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '/'

	def GetOutput(self, X):
		X0 = self._children[0].GetOutput(X)
		X1 = self._children[1].GetOutput(X)
		return (np.multiply( np.sign(X1), X0) / ( 1e-2 + np.abs(X1) )) * self.weights[0] + self.weights[1]


class AnalyticQuotientNode(Node):
	def __init__(self):
		super(AnalyticQuotientNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return 'aq'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput(X)
		X1 = self._children[1].GetOutput(X)
		return (X0 / np.sqrt(1 + np.square(X1))) * self.weights[0] + self.weights[1]

	
class ExpNode(Node):
	def __init__(self):
		super(ExpNode,self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'exp'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return (np.exp(X0)) * self.weights[0] + self.weights[1]


class LogNode(Node):
	def __init__(self):
		super(LogNode,self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'log'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return (np.log( np.abs(X0) + 1e-2 )) * self.weights[0] + self.weights[1]


class SinNode(Node):
	def __init__(self):
		super(SinNode,self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'sin'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return (np.sin(X0)) * self.weights[0] + self.weights[1]

class CosNode(Node):
	def __init__(self):
		super(CosNode,self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'cos'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return (np.cos(X0)) * self.weights[0] + self.weights[1]


class FeatureNode(Node):
	def __init__(self, id):
		super(FeatureNode,self).__init__()
		self.id = id

	def __repr__(self):
		return 'x'+str(self.id)

	def GetOutput(self, X):
		return (X[:,self.id]) * self.weights[0] + self.weights[1]

	
class EphemeralRandomConstantNode(Node):
	def __init__(self):
		super(EphemeralRandomConstantNode,self).__init__()
		self.c = np.nan

	def __Instantiate(self):
		self.c = np.round( np.random.random() * 10 - 5, 3 )

	def __repr__(self):
		if np.isnan(self.c):
			self.__Instantiate()
		return str(self.c)

	def GetOutput(self,X):
		if np.isnan(self.c):
			self.__Instantiate()
		return (np.array([self.c] * X.shape[0])) * self.weights[0] + self.weights[1]
