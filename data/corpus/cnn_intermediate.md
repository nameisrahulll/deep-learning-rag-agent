# Convolutional Neural Networks

## Convolution, Kernels, and Feature Maps
Convolutional Neural Networks are designed to process grid-like data such as
images by exploiting local spatial structure. Instead of connecting every input
pixel to every neuron as in a dense network, a convolution layer applies small
learned filters across the input. Each filter slides over local regions and
produces an activation map that responds strongly to specific patterns such as
edges, corners, or textures. Because the same filter weights are reused across
many positions, CNNs need fewer parameters than fully connected models on
images. This parameter sharing also gives them translation-aware behavior: a
useful pattern can be recognized in different parts of the image by the same
filter. The output of a convolution layer is often called a feature map because
it records where certain learned features appear. A good interview answer
should highlight that the power of CNNs comes from local connectivity, weight
sharing, and the ability to build hierarchical visual representations over many
layers.

## Pooling, Receptive Fields, and Invariance
Pooling layers reduce the spatial resolution of feature maps while retaining
the most important information. Max pooling, for example, keeps the strongest
activation within a small window, which helps the network become less sensitive
to small spatial shifts. As more convolution and pooling layers are stacked,
the effective receptive field grows. This means deeper neurons can respond to
larger and more complex regions of the image even though each individual filter
is local. Early layers often capture simple visual primitives, while deeper
layers detect combinations such as object parts or class-specific structures.
This progression is one reason CNNs perform so well on vision tasks. Pooling is
not the only way to reduce resolution, but it is a common interview topic
because it helps explain how CNNs trade exact location information for more
robust pattern detection. Strong answers connect pooling to invariance and to
the broader idea that deeper networks combine local cues into semantically
richer visual features.

## LeNet, AlexNet, and the Historical Impact of CNNs
LeNet showed that convolutional architectures could solve practical image
recognition tasks such as handwritten digit classification. Its design
demonstrated that alternating convolution and pooling layers could learn useful
visual features before a final classifier stage. Years later, AlexNet marked a
major turning point for deep learning in computer vision. It used a deeper
architecture, ReLU activations, dropout, and GPU acceleration to achieve a
dramatic improvement on ImageNet. AlexNet made it clear that learned features
could outperform hand-engineered vision pipelines when enough data and compute
were available. In interviews, these architectures are often discussed not only
as historical milestones but also as examples of design tradeoffs. LeNet is
compact and conceptually simple, while AlexNet shows how scaling depth,
regularization, and compute can unlock far better performance. A complete answer
should connect these models to the broader success of CNNs in classification,
detection, and feature extraction tasks.
