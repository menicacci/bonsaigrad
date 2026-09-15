# 15 — Character language model

> 🧪 Runnable companion: [`notebooks/15_character_language_model.ipynb`](../notebooks/15_character_language_model.ipynb)

A character language model learns to predict the next character from the
characters before it. Here we train on Italian names: predicting one character
at a time gives us a way to generate whole names.

## From text to training examples

`CharacterTokenizer` assigns each character an integer ID. Its vocabulary also
contains two special tokens: `<PAD>` fills missing context, and `<EOS>` marks
the end of a name. These IDs identify tokens; their numerical order carries
no meaning.

`make_examples` turns each name into context/target pairs. With a context
length of three, `anna` produces:

| Context | Next token |
| --- | --- |
| `<PAD> <PAD> <PAD>` | `a` |
| `<PAD> <PAD> a` | `n` |
| `<PAD> a n` | `n` |
| `a n n` | `a` |
| `n n a` | `<EOS>` |

Each context keeps only the most recent characters and pads on the left when
there are too few. The all-padding example teaches the model how names begin;
the final example teaches it when to stop. Padding has a learned embedding
like any other token.

## From context to prediction

`EmbedderCharacterModel` looks up an [embedding](12-embedding.md) for each
token, then concatenates the context's vectors into one feature vector.
For batch size $B$, context length $C$, and embedding dimension $D$:

$$
(B, C) \xrightarrow{\text{embedding}} (B, C, D)
\xrightarrow{\text{reshape}} (B, CD).
$$

`Leaf.reshape` changes the shape without changing the sequence of values.
Its backward pass reshapes the upstream gradient back to the input shape:
each value receives its corresponding gradient. This keeps the embedding
connected to the loss. Each context position occupies a distinct slice of
the flattened vector, so the following layer can distinguish character order.

Repeated `Linear → LayerNorm → Tanh` blocks turn that vector into hidden
features. `NextCharacterPredictor` adds a final linear layer producing one
logit per vocabulary token. [Cross-entropy](09-loss.md) trains those logits
against the next-token targets, using [Adam](11-optimizers.md) on randomly
sampled minibatches.

## Generating one token at a time

During training, contexts contain characters from the dataset. During
generation, the model feeds its own sampled characters back into the context.
This is **autoregressive generation**:

$$
P(x_1, \ldots, x_T)
= \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1}),
$$

where $x_T$ is `<EOS>`. Our model approximates each conditional using only
the last $C$ tokens.

`generate` starts from a prompt, left-pads its context, and converts the output
logits to probabilities with softmax. It excludes `<PAD>` from sampling,
then randomly draws the next token. Sampling allows several plausible
completions instead of always choosing the highest-scoring character.

The sampled character is appended and the process repeats until `<EOS>` or
`max_new_tokens` is reached. This is a fixed-context feed-forward model:
characters outside its context cannot influence the next prediction.
