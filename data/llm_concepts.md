# Large Language Models: Core Concepts

## Tokens
Large Language Models (LLMs) do not read raw characters; they read **tokens**.
A token is a chunk of text — often a word fragment of about four characters in
English. The sentence "RAG is powerful" might be split into the tokens
["RAG", " is", " powerful"]. Models have a maximum **context window** measured in
tokens, which bounds how much text (prompt plus generated output) they can
process at once. Both API cost and latency scale with the number of tokens.

## Embeddings
An embedding is a fixed-length vector of floating-point numbers that represents
the meaning of a piece of text. Texts with similar meaning map to vectors that
are close together in the embedding space. Semantic search works by embedding a
query and finding stored vectors that are nearest to it, usually by
**cosine similarity**. This is the retrieval backbone of a RAG system.

## Prompting
A **prompt** is the input given to an LLM. A well-structured prompt for a RAG
system typically contains three parts: a **system instruction** that sets the
model's role and rules (for example, "Answer only from the context"), the
**retrieved context**, and the **user question**. Telling the model to say
"I don't know" when the answer is not in the context is a key technique for
reducing hallucination.

## Temperature
**Temperature** controls randomness in generation. A temperature of 0 makes the
model nearly deterministic and is preferred for factual question answering, where
we want the same grounded answer every time. Higher temperatures (0.7–1.0) add
creativity and are useful for brainstorming or content generation.

## Hallucination
A hallucination is a confident but false or unsupported statement produced by an
LLM. Hallucinations arise because the model is trained to produce plausible text,
not necessarily true text. RAG mitigates hallucination by grounding answers in
retrieved evidence and by instructing the model to abstain when evidence is
missing.

## Evaluation
RAG systems are evaluated on two axes. **Retrieval quality** asks whether the
correct passages were fetched (measured with metrics such as recall@k and
context precision). **Generation quality** asks whether the final answer is
faithful to the retrieved context and actually answers the question
(measured with faithfulness and answer-relevance metrics). Faithfulness is the
fraction of an answer's claims that are supported by the retrieved context.
