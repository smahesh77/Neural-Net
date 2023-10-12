from BinaryNeuralNet import NeuralNet
from SampleDataSet import xs, ys

for k in range (10): # 20 iterations to find that sweet spot spot for values

  # forward pass
  ypred = [NeuralNet(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

  # backward pass
  for p in NeuralNet.parameters():
    p.grad = 0.0 # reset the value of grads to prevent them from getting added on top of old valuees
    loss.backward()

  # update
  for p in NeuralNet.parameters(): # 0.01 is the learning rate
    p.data += -0.01 * p.grad # we modify the weights and bias as to decrease the loss

  print(k, loss.data)