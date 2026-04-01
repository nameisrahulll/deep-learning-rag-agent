# Autoencoders

## Encoder, Decoder, and Latent Representation
An autoencoder is a neural network trained to reconstruct its own input. It is
usually divided into two main parts: an encoder that maps the input into a
compressed latent representation, and a decoder that reconstructs the original
input from that latent code. The latent space acts as a bottleneck, forcing the
network to learn a compact representation rather than simply copying the input
through an identity mapping. This makes autoencoders useful for representation
learning, dimensionality reduction, and feature extraction. The key interview
idea is that the model is not valuable because it reproduces the input exactly.
It is valuable because the compression pressure encourages the model to capture
the most informative structure in the data. A strong answer should explain that
the latent representation is the real objective: reconstruction is the training
signal that teaches the network what information should be preserved.

## Reconstruction Loss and Bottleneck Design
The training objective of an autoencoder is usually a reconstruction loss that
measures how close the output is to the original input. For continuous values,
mean squared error is common, while binary cross-entropy may be used when the
data is binary or normalized probabilistically. The bottleneck design matters
because it determines how much compression pressure the model experiences. If
the latent space is too large and the architecture is too flexible, the network
may learn a nearly trivial identity function that does not produce meaningful
features. If the bottleneck is too severe, the reconstruction may become poor
because too much information is discarded. Variants such as sparse or denoising
autoencoders intentionally modify the training setup to encourage stronger
representations. In interviews, it is useful to connect the reconstruction loss
to the idea that the model learns by preserving essential structure while
discarding less important detail.

## Use Cases and Relation to Other Models
Autoencoders are useful for unsupervised representation learning, anomaly
detection, denoising, and pretraining. In anomaly detection, a model trained on
normal examples may reconstruct familiar patterns well but perform poorly on
unusual inputs, making reconstruction error a useful signal. Denoising
autoencoders learn to recover clean inputs from corrupted versions, which helps
the model capture robust structure rather than memorize exact samples. There is
also a conceptual connection between autoencoders and encoder-decoder Seq2Seq
models because both compress information and reconstruct or generate outputs
from an intermediate representation. However, the goal differs: autoencoders
typically reconstruct the same input, while Seq2Seq models transform one
sequence into another. A complete interview answer should explain both the
shared encoder-decoder pattern and the difference in objective, since this
comparison often reveals whether the candidate understands why these models are
used in different settings.
