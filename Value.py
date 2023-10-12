import math
class Value :

    def __init__(self, data, _childern=(), _op ='', label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        # prev stores the operands of the operation
        self._prev = set(_childern)
        # to store the operartion
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data})"

    def __rmul__(self,other): # for doing operartions like2*a
      return self * other

    def __radd__(self,other): # for doing operartions like2+a
      return self + other

    # this will get called when we add two objects of the Value class
    #  a + b will be converted to a.__add__(b) where a will be self and b will be other
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
          # for addition we just pass the derivatives
          self.grad += 1.0*out.grad
          other.grad += 1.0*out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
      other = other if isinstance(other,Value) else Value(other) # for doing operations like a + 1
      out = Value(self.data * other.data, (self, other), '*')

      def _backward():
          self.grad += other.data * out.grad  # d(self.data * other.data)/d(self) = other.data
          other.grad += self.data * out.grad  # d(self.data * other.data)/d(other) = self.data
          # Note: You don't need to print(out.grad) here.

      out._backward = _backward
      return out


    def __sub__(self, other):
        return self +(-other)

    def tanh(self): # activation function
      x = self.data
      t = (1 - math.exp(-2 * x)) / (1 + math.exp(-2 * x))
      out = Value(t, (self,), "tanh")
      def _backward():
          self.grad += (1 - t**2) * out.grad
      out._backward = _backward
      return out



    def exp(self):
      x = self.data
      out = Value(math.exp(x), (self,), 'exp')

      def _backward():
        self.grad += out.data * out.grad
      out._backward = _backward
      return out

    def __truediv__(self, other):
      self * other **-1

    def __pow__(self, other):
      assert isinstance(other, (int, float))
      out = Value(self.data ** other, (self,), f'**{other}')

      def _backward():
        self.grad +=out.grad*other*self.data ** (other -1)
      out._backward = _backward
      return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

