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
Consider a sentence as a sequence. We denote $X^{(i)[t]}$ to the $t$-th element of the $i$-uth sample, and we denote $T_X$ as the length of the input. 
Consider a dictionary (of size for instance 10k words). It is typical to have 30-40k size dictionary. So one way of representing words is using one-hot encoding which requires a vector of size 10k in our example.
# 3 Models developed for NLP
- **CBOW** (Continuous Bag of Word): The goal is to extract word embeddings. For every sequence, we take a fixed window as the context around each word. We build a vocabulary from all words and the representation for each word is one-hot encoding. Then we feed this representation in feed forward neural network to predict the central word’s embedding. This has a significant limitation which is the fixed length of the model. So recurrent models are suggested.
- **ELMo**: In this model, we use a bi-directional LSTM to predict the middle word and we can use all words before and after as the context.
	Transformer: It contains an encoder and decoder block based on attention mechanism
- **GPT**: It contains only a decoder and using that it predicts the next word. Originally it is uni-directional and it uses causal attention mask.
- **BERT**: It contains only encoder block and it uses bi-directional context. It is trained on two tasks, multi-mask language modeling and next sentence prediction which is binary decision for every query.
- **T5**: It contains an encoder decoder stack similar to the Transformer model and it uses bi-directional context. It is capable of multi-task such as both predicting the rate of a movie based on a review and answer a question and get an answer. It performs this by receiving a text as the indicator of the task.

# 4 Recurrent Neural Network
In this model, the parameters from input to the hidden units are called $w_{ax}$, while the parameters from hidden units of one step to next one is denoted by $w_{aa}$, and finally from hidden layer to output is shown with $w_{ya}$. 
![](images/1.png)

## 4.1 Forward Propagation
Consider the RNN structure above. The forward propagation is described as below:
```math
\begin{align}
	&a^{[0]}=0\\
	&a^{[t]}=g(w_{aa}  a^{[t-1]} + w_{ax}  X^{[1]}+b_a)\\
	&y^{[t]}=g(w_{ya}  a^{[t]}+b_y)
\end{align}
```

Usually the activation function tanh() is used and sometimes ReLU.

To simplify the notation, we can concatenate $w_{aa}$ and $w_{ax}$ and refer to it as $w_a$. Then by stacking $a^{[t-1]}$ and $X^{[t]}$ we can rewrite:
```math
\begin{align}
	&a^{[t]}=g(w_a  [a^{[t-1]}│X^{[1]} ]+b_a )\\
	&\hat{y}^{[t]}=g(w_y  a^{[t]}+b_y)
\end{align}
```

## 4.2 Back Propagation
Let us define the loss function as:
```math
\begin{align}
L^{[t]} (\hat{y}^{[t]},y^{[t]})=-y^{[t]}  log⁡(\hat{y}^{[t]})- (1-y^{[t]})log⁡(1-\hat{y}^{[t]})\\
L(\hat{y}, y)=\sum_{t=1}^T L^{[t]} (\hat{y}^{[t]},y^{[t]} )
\end{align}
```

## 4.3 Various Types of RNN
It is possible that in some applications, the length of input and output are different. For instance, in sentiment classification, we may want to map a text to a score for instance (many-to-one relation). Then the architecture can be:

![](images/2.png)

In another example such as machine translation, the architecture could be:

![](images/3.png)

## 4.4 Language Model
To create a language model, we need a large corpus (text body) of text for training. To give it more structure, in a sentence we call each word (also called token) as  $y^{[1]},y^{[2]},\dots,y^{[10]}$ and we can allocate the end of a sentence `<EoS>` as a token as well. If a word is not in our dictionary, conventionally we can map that to unknown token `<UNK>`.
Consider that we are interested to train a model that gives a score to each sentence. For example, we want to give a score to the sentence:

<p align="center">
Cats average 15 hours of sleep a day. &lt EOS &gt
</p>

We have $P(y^{[1]},y^{[2]},\dots,y^{[9]} )=P(y^{[1]} )P(y^{[2]}│y^{[1]} )…P(y^{[9]}│y^{[1]}…y^{[8]})$. Then we can compute each of these scores using the following model:

![](images/4.png)

It is also possible to design the model in character model. Then the dictionary would consist of characters and space and punctuations. 

To generate a sample from our model, simply we begin with an input for instance 0, and generate the output and use that output as the input of the next step. We continue this until we reach to `<EoS>`.

## 4.5 Vanishing/exploding Gradient Problem
In language applications, we may need long term dependencies. For instance, consider the following example where early words influence on the last words.

<p align="center">
The cat which always ate …., was full.
The cats which always ate …., were full.
<\p>
	
So, it is hard in back propagation to capture this long term dependencies and we face vanishing gradient. 
In some cases, when the derivatives become larger than 1, we may see exploding gradient and we need to address that by gradient clipping technique for instance. To address vanishing gradient, identity RNNs or skip connections are suggested. The best known approach to mitigate this is LSTM.

One conclusion is that we need to some how store the memory!

## 4.6 Gated Recurrent Unit
GRU is introduced to address the memory issue of RNN. Below is a schema of each RNN unit:
$a^{[t]}=tanh⁡(W_a [a^{[t-1]},x^{[t]}]+b_a)$

Let us define the memory as $c^{[t]}$. In RNN, we consider $c^{[t]}=a^{[t]}$.
For Gated Recurrent Unit (GRU), let us define a variable called candidate memory:
$\tilde{c}^{[t]}=tanh⁡(W_c [c^{[t-1]},x^{[t]}]+b_c)$
Then let us define the update gate function as:
$\Gamma_u=\sigma⁡(W_u [c^{[t-1]},x^{[t]}]+b_u)$
This gate somehow controls if we need to apply what we have stored in the memory or not. Like in the example above Gate tells us when it is important to care about was/were.
Then we can define: 
$c^=\Gamma_u *\tilde{c}^{[t]}+ 〖(1-\Gamma_u) *c^{[t-1]}$
The fill GRU has another gate (relevant gate) function $\Gamma_r$. So, the complete formulas become:

```math
\begin{align}
c^{[t]}=tanh⁡(W_c[\Gamma_r* c^{[t-1]},x^{[t]}]+b_c )
\Gamma_u=\sigma⁡(W_u [c^{[t-1]},x^{[t]} ]+b_u )
\Gamma_r=\sigma⁡(W_r [c^{[t-1]},x^{[t]} ]+b_r )
c^{[t]}=\Gamma_u  *\tilde{c}^{[t]}+ (1-\Gamma_u) *c^{[t]}
\end{align}
```
