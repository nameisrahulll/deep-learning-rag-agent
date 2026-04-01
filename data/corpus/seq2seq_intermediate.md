# Sequence-to-Sequence Models

## Encoder-Decoder Structure
A Sequence-to-Sequence model maps one sequence into another by splitting the
task into an encoder stage and a decoder stage. The encoder reads the input
sequence and compresses its information into an internal representation. The
decoder then uses that representation to generate the target sequence step by
step. This setup is useful when the input and output lengths differ, such as in
machine translation, summarization, or question generation. Early Seq2Seq
systems frequently used recurrent networks, especially LSTMs, for both encoder
and decoder because they could process ordered data and maintain state over
time. The architecture is important because it separates understanding the
source from generating the target, rather than treating the whole problem as a
single fixed-size classifier. In an interview, a strong answer should emphasize
that Seq2Seq is a framework for variable-length sequence transformation rather
than a single specific layer type.

## Role of LSTM Memory in Seq2Seq Systems
LSTM memory helps Seq2Seq models because both encoding and decoding require
useful information to persist over many time steps. During encoding, the model
must summarize the input sequence into a state that preserves meaning relevant
for later generation. During decoding, the model must generate coherent output
conditioned on that summary and on previous generated tokens. Basic RNNs often
struggle with this because long-range dependencies are hard to maintain.
Replacing them with LSTMs improves the system's ability to retain important
context across longer sequences. This is especially important when the model
must connect early source tokens to much later output decisions. In a technical
discussion, it is helpful to explain that the LSTM does not change the encoder
decoder idea itself. Instead, it improves the quality of memory and optimization
inside that framework, making Seq2Seq training more effective for complex
sequence transformation tasks.

## Training, Inference, and Common Challenges
Seq2Seq models are typically trained with teacher forcing, where the decoder
receives the correct previous target token during training instead of its own
generated output. This speeds learning but creates a mismatch with inference,
where the model must rely on its own previous predictions. Errors can therefore
accumulate at generation time. Another challenge is the information bottleneck
in early Seq2Seq designs, where the encoder must compress the entire source
sequence into a limited representation before decoding begins. Attention later
reduced this bottleneck by letting the decoder focus on different source
positions dynamically, but the core Seq2Seq idea remains foundational. In an
interview, you should connect Seq2Seq to both architecture and training
behavior. It is not enough to say it maps input text to output text. You should
also explain how the encoder and decoder interact, why memory quality matters,
and what practical issues appear when generating outputs autoregressively.
