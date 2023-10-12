import random

from Value import Value
class Neuron:

  # nin is the number of inputs
  def __init__(self, nin): # the _ in the for just means repeat nin times dont think too much about it
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # will generate random weights for all the inputs
    self.b = Value(random.uniform(-1,1)) # random bias for all the input

  # x is the input values
  def __call__(self, x): # this will getcalled for x = Neuron, x(2)
    zip(self.w, x) # will match each xi with xi
    act = sum(wi*xi for wi,xi in zip(self.w,x)) + self.b # will mult the weights with the inputs and add bias to it and add them
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]


class Layer:

  def __init__(self, nin, nout): # nout is the number of neurons in an single layer nin is the number of inputs to each neuron
    self.neurons = [Neuron(nin) for i in range(nout)] # will create nout neurons with nin inputs

  def __call__(self, x): # x is a list of input values for  a single neuron
    outs = [n(x) for n in self.neurons] # calls n{x) for neuron in self.neurons array and creates an array of those outputs
    #
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
  def __init__(self, nin, nout): # nout is list containing the numbers of neurons in each layer of the mlp
    sz = [nin] + nout # we convert the number of ip to a list and add it to the list coresponding to the number of neurons in each layer
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nout))]
    # so this basically means that initially there will be the difined number of inputs for the
    # first layer then after that the number of neuron in the second layer will be the number of inputs
    # for each neuron in the second layer

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]


initialInput = [2.0,3.0, -1.0]
NeuralNet = MLP(3, [4,4,1])
