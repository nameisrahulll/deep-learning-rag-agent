# Artificial Neural Networks

## Forward Propagation and Layered Computation
Artificial Neural Networks model a function by passing information through
layers of interconnected units. Each neuron receives an input vector, applies
weights and a bias, computes a weighted sum, and then passes that result through
an activation function. In a feedforward network, data moves from the input
layer through one or more hidden layers to an output layer without cycles. This
layered structure lets the model learn increasingly abstract representations as
depth increases. Early layers may capture simple combinations of input
features, while deeper layers combine those patterns into more useful
intermediate concepts. The forward pass is the process of producing a
prediction from current parameters. During training, that prediction is
compared with the target output using a loss function. The difference between
predicted and actual values becomes the signal that drives learning. A strong
interview answer should connect the forward pass to the idea that a neural
network is ultimately a parameterized function approximator.

## Backpropagation and Gradient-Based Learning
Backpropagation is the learning procedure that makes multilayer neural
networks practical. After the network computes an output, the loss function
quantifies error. Backpropagation then applies the chain rule from calculus to
compute how much each weight contributed to that error. Instead of estimating
weight updates independently, the algorithm efficiently propagates gradients
from the output layer backward through the network. Each parameter receives a
partial derivative showing how a small change in that weight would affect the
loss. Gradient descent or a related optimizer then updates the parameters in
the direction that reduces error. This is the core idea behind the classic
Rumelhart, Hinton, and Williams work that helped popularize deep learning.
When explaining backpropagation in an interview, it is important to emphasize
that it is not a separate model from the neural network. It is the mechanism
that allows the network to learn internal representations by reusing local
gradient computations across many layers.

## Activation Functions, Loss Functions, and Training Behavior
Activation functions give neural networks the nonlinearity needed to model
complex decision boundaries. Without nonlinear activations, stacking multiple
layers would collapse into a single linear transformation. Sigmoid and tanh
were historically important, but they can saturate and contribute to vanishing
gradients. ReLU became widely used because it is simple, computationally cheap,
and often improves optimization in deeper networks. The loss function defines
what the network is trying to optimize. Mean squared error is common for
regression, while cross-entropy is standard for classification because it
better matches probabilistic outputs. Training behavior emerges from the
interaction of architecture, activation choice, learning rate, and dataset
quality. Overfitting occurs when the network memorizes the training data rather
than generalizing. Techniques such as regularization, dropout, and validation
monitoring are therefore critical. A solid ANN explanation should link
activation choice to representational power and loss choice to the objective
the model is actually being trained to minimize.
