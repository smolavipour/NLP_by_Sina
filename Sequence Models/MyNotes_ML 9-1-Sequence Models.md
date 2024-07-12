<p align="center">
<img src="../logo.png" alt="Alt text" width="100"/>
</p>

<h1 align="center">
    Sequence Models
</h1>

<h2 align="center">
    Machine Learning Self-study
</h2>

# 1 Applications of sequence models:
- Speech recognition
- Music generation
- Sentiment classification (give stars to a content)
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Name entity recognition
  
# 2 Notations
Consider a sentence as a sequence. We denote $X^{(i)&lt;t&gt;}$ to the $t$-th element of the $i$-uth sample, and we denote $T_X$ as the length of the input. 
Consider a dictionary (of size for instance 10k words). It is typical to have 30-40k size dictionary. So one way of representing words is using one-hot encoding which requires a vector of size 10k in our example.
# 3 Models developed for NLP
- **CBOW** (Continuous Bag of Word): The goal is to extract word embeddings. For every sequence, we take a fixed window as the context around each word. We build a vocabulary from all words and the representation for each word is one-hot encoding. Then we feed this representation in feed forward neural network to predict the central word’s embedding. This has a significant limitation which is the fixed length of the model. So recurrent models are suggested.
- **ELMo**: In this model, we use a bi-directional LSTM to predict the middle word and we can use all words before and after as the context.
	Transformer: It contains an encoder and decoder block based on attention mechanism
- **GPT**: It contains only a decoder and using that it predicts the next word. Originally it is uni-directional and it uses causal attention mask.
- **BERT**: It contains only encoder block and it uses bi-directional context. It is trained on two tasks, multi-mask language modeling and next sentence prediction which is binary decision for every query.
- **T5**: It contains an encoder decoder stack similar to the Transformer model and it uses bi-directional context. It is capable of multi-task such as both predicting the rate of a movie based on a review and answer a question and get an answer. It performs this by receiving a text as the indicator of the task.
