# Recurrent Neural Networks

## Hidden State and Sequence Modeling
Recurrent Neural Networks are built for ordered data where previous inputs
should influence later predictions. Instead of treating each example as
independent, an RNN maintains a hidden state that is updated at every time
step. The hidden state acts as a compressed memory of what the model has seen
so far. For a sequence such as text, audio, or time-series data, the network
combines the current input with the previous hidden state to produce a new
hidden state and, if needed, an output. Because the same transition weights are
shared across time, the network can process variable-length sequences with a
fixed set of parameters. This makes RNNs conceptually elegant for sequence
tasks. In an interview, it is important to explain that an RNN does not store
all past inputs explicitly. It learns a compact state representation that
summarizes history, which is powerful but also creates a bottleneck when long
range dependencies need to be preserved accurately.

## Backpropagation Through Time and the Vanishing Gradient Problem
Training an RNN uses Backpropagation Through Time, or BPTT, which unfolds the
recurrent computation across sequence steps and applies gradient descent to the
expanded graph. The same recurrent weights appear at each step, so gradients
from later outputs must pass through many repeated transformations when flowing
backward. This causes a major optimization challenge. If the recurrent
transitions repeatedly multiply values smaller than one, gradients shrink and
the model struggles to learn long-range dependencies. This is the vanishing
gradient problem. If values grow too much, exploding gradients can occur.
Standard RNNs therefore often learn short-term dependencies more easily than
long-term ones. This limitation is one reason gated architectures such as LSTM
and GRU became so important. A strong answer should connect BPTT to the idea
that sequence length directly affects optimization difficulty because the same
parameters are reused across many steps in the unfolded computational graph.

## Use Cases, Strengths, and Practical Limitations
RNNs were historically important for language modeling, speech processing, and
time-series forecasting because they naturally consume sequences one step at a
time. They are especially intuitive when order matters and past context should
shape future outputs. However, their sequential nature limits parallelism
during training, which can make them slower than architectures that process all
positions at once. Their hidden state can also struggle to preserve precise
information over long sequences, especially in the plain or vanilla RNN form.
Even when they perform well on short contexts, they may degrade when important
signals occur far apart in time. These limitations motivated more advanced
sequence models such as LSTMs, GRUs, and later transformers. In an interview,
an effective explanation should present RNNs as foundational sequence models:
they introduced the key idea of learned temporal state, but they also revealed
the optimization and memory challenges that later architectures were designed to
address more effectively.
