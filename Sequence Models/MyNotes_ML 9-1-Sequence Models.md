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
c^{[t]}&=tanh⁡(W_c[\Gamma_r* c^{[t-1]},x^{[t]}]+b_c )\\
\Gamma_u&=\sigma⁡(W_u [c^{[t-1]},x^{[t]} ]+b_u )\\
\Gamma_r&=\sigma⁡(W_r [c^{[t-1]},x^{[t]} ]+b_r )\\
c^{[t]}&=\Gamma_u  *\tilde{c}^{[t]}+ (1-\Gamma_u) *c^{[t-1]}
\end{align}
```

In academic literature, people may use the following literature:
$\tilde{c}=\tilde{h}, \Gamma_u=u, \Gamma_r=r, c=h$

![](images/5.png)

## 4.7 Long Short-Term Memory (LSTM)
It contains of the following gates: Update, Forget, Output.

![](images/6.png)

```math
\begin{align}
c^{[t]}&=tanh⁡(W_c[\Gamma_r*a^{[t-1]},x^{[t]}]+b_c )\\
\Gamma_u&=\sigma⁡(W_u [a^{[t-1]},x^{[t]} ]+b_u )\\
\Gamma_r&=\sigma⁡(W_r [a^{[t-1]},x^{[t]} ]+b_r )\\
\Gamma_o&=\sigma⁡(W_o [a^{[t-1]},x^{[t]} ]+b_o )\\
c^{[t]}&=\Gamma_u*\tilde{c}^{[t]}+ (1-\Gamma_u) *c^{[t-1]}\\
a^{[t]}&=\Gamma_o *tanh⁡(c^{[t]})
\end{align}
```
Forget gate decides what information is keep or discarded. Input gate decides what information to be added in the cell state. 

## 4.8 Bidirectional RNN
In the bidirectional RNN, each direction is completely independent of the other one. After the hidden state is computed, the final output would be computed using:
$\hat{y}^{[t]}=g(W_y [{\overrightarrow{a^{[t]}}},{\overleftarrow{a^{[t]}}}]+b_y)$

![](images/7.png)

## 4.9 Deep RNN
Like deep NN, we can add more hidden layer to RNN/GRU/LSTM as well. Unlike deep RNN that may have many layers, deep RNNs have 2 3 layers and that is already a lot of parameters.

![](images/8.png)

$a^{[2]&lt3&gt}=g(W_a^{2}  [a^{[2]&lt2&gt},a^{[1]&lt3&gt}]+b_a^{2} )$

# 5 Word Embedding
Suppose we have a dictionary $V$ with 10k samples. The one-hot representation of words is that it is a vector of dimension 10k with all zeros except one at the index of the corresponding word. We denote it by $O_{421}$ as an example for 421th word.

One way to do embedding is to use methods such as t-SNE. However, t-SNE is more to visualize the data in lower dimension. 

There are several ways to learn word embeddings:
- Have a large corpus of text 1-100 B words
- Download pre-trained embeddings
Then one can use transfer learning to a new task with smaller training set (100k words)

In this area we deal with context. Context could be 4 words, or for example 4 left words and 4 right words, and we want to predict the next word or the word in the middle respectively.

## 5.1 Analogy reasoning
It answers to questions such: the relation between `Man->Woman` is similar to the relation between `King->`? And the model should give the answer “Queen”. The accuracy of the current models is around 30-75%.
The conventional way is to compare the embedding vector using proper distance metrics. In t-SNE, due to non-linearities one should not expect to identify analogies, especially complicated parallel relationships as below:

One typical distance metric is cosine similarity which is basically:
```math
\begin{equation}
similarity=\frac{u^T v}{\lvert\lvert u\rvert\rvert_2  \lvert\lvert v\rvert\rvert_2}$
\end{equation}
```

![](images/9.png)

This is much more used comparing to Euclidian distance.

## 5.2 Skip-gram (Word2Vec)
Let $c$ be the context (one word in this case) and t to be the target word in our dictionary of 10k words. The idea is that the target word is not far than a few words distance to the context word. It is called skip-gram since we skip some words from the context to look for the target. The goal is to upon a given context word, generate a  $\hat{y}$ that is likely to be the target. By using the softmax we have:
```math
\begin{equation}
p(t\lvert c)=\frac{e^{\Theta_t^T e_c}}{\sum_{j=1}^{10k} e^{\Theta_j^T e_c } }
\end{equation}
```

Where $e_c=E o_c$, $E$ is the embedding matrix mapping each word in the vocab to a vector, and $o_c$ is the one-hot vector corresponding to word $c$. 

**Problems**
Skipgram is computationally heavy since every time we have to some over the dictionary (10k). There are techniques to overcome this issue. Instead we can train layers of binary classifier as hierarchy that tells if the prediction is in the first half or the second half of the vocab. So instead of linear complexity $o(d)$ we would have logarithmic complexity $o(log⁡d)$.

Another thing to note is that words like **it**, **of**, **the**,… are very common and by uniformly sampling the context word in training, we may get biased training set. So we should take care of this as well.

## 5.3 Negative Sampling
Consider creating a dataset as following. Given a pair of words for example (“Orange”) and (“Juice”), we add k random words (negative examples) from dictionary (that probably it is not associated with these words). Then we assign a target value 1 to “Juice” and 0 to all other k samples. So we can train a model to tell if two words are close or not. $k$ is 2-5 for large datasets and 5-20 for smaller datasets.

Now to use this as a predictive model for predicting a target word given a context, we can train 10k binary classifiers for each target word. However, since for each context word, we only have $k+1$ pairs (i.e., k+1 classifiers), we are not updating all classifiers at once. This is much cheaper than training one softmax model to classify 10k groups.

**Problems**
Similar to skip-gram, there is a problem of getting biased when sampling negative examples. So, a heuristic way to sample words is to consider the distribution
```math
\begin{equation}
p(w_i)=\frac{f(w_i )^{3/4}}{\sum_{j=1}^{10k} f(w_i)^{3/4}}
\end{equation}
```
where $f(w_i)$ is the frequency of the word in English corpse. Then we sample according to this.

## 5.4 GLoVe Word Vectors
GLoVe stands for Global Vectors for Word Representation. Let us define $X_ct$ be #times $t$ appears in the context of $c$. Typically we assume $X_ct=X_tc$.
Let is define the loss function:
```math
\begin{equation}
Minimize \sum_{i=1}^{10k}\sum_{j=1}^{10k} f(X_{ij}) (\Theta_i^T e_j+b_i+b_j-\log⁡ X_{ij})^2 〗  
\end{equation}
```
Where $0\log⁡0=0$. 
This method is fairly easy and much faster than previous methods.

## 5.5 Embedding layer in libraries
The embedding layer in libraries is not a fixed lookup table, rather is trained during the training phase. Let us assume the desired output dimension is 10. Then the model initializes a table of dimension 10 for each word. Then for 1000 words, the matrix is 1000×10. So to access the embedding for each word, we can multiply a one-hot vector to this matrix and extract the corresponding vector. So the differentiation engine can use this multiplication to take the gradient and compute the matrix. It is possible to load a pre-trained model such as word2vec or GLoVe if the vocabulary and output dimension matches. In practice though, for fast implementation, the computer is not multiplying matrices and only takes the corresponding row from the matrix.

# 6 Sentiment Classification Problem
One famous use case of this category of problem is to understand the rating stars based on the review.
![image](https://github.com/user-attachments/assets/85a9ae72-b50c-416c-8bdd-c5c3bb8f2d37)

So $E$ is obtained using a large database 100B words beforehand.
We can then take the average over the feature vectors e and pass it through a Softmax to predict the rating. So it does not matter to deal with a short or long review. There is a problem with the current model which is that a word NOT can change the whole context, so if we have so many “Good” words in the review the average feature vector of “good” dominates and we give it a five star while we missed the important “NOT” word. So it is suggested to use RNN sentiment analysis instead.

![image](https://github.com/user-attachments/assets/bdb9f25b-29ba-46bb-b17f-3cdb544acd8a)

# 7 De-biasing 
There are biases in our society and accordingly in our corpse of text. This holds true in other applications as well when we need to keep a balance and de-bias our method. Let us look at this from word embedding perspective. So the first step is to identify the direction of bias.

## 7.1 Identify the bias direction
To learn about the direction of bias given the word embeddings, we can collect examples such as 
```math
\begin{align}
e_{he}&-e_{she}
e_{male}&-e_{female}
e_{father}&-e_{mother}
\end{align}
```

And take an average over these vectors to identify the direction of bias.
In more complicated approaches, we can use PCA, t-SNE or SVD to extract eigen vectors and identify which one is related to the corresponding bias.

## 7.2 Neutralize
Some words are biased intrinsically: male, female, father, mother and words like doctor, nurse, leader should be neutralized. For the words that are not intrinsically biased, we project them on the space orthogonal to the identified bias vector.

## 7.3 Equalize pairs
For the intrinsically biased words, such as male and female, we set the distance between male and the orthogonal space equal to the distance between female and the orthogonal space.
![image](https://github.com/user-attachments/assets/29e3942e-d6bf-4738-bdd1-57670a468a33)

# 8 LSTM for classification
In case we want to use LSTM for classification, one first problem is that the length of words could be different. Then we can use padding to make all the lengths the same.
![image](https://github.com/user-attachments/assets/8372f22d-e7ce-496c-8a5c-ed5bad7ae5be)

# 9 Sequence to Sequence Model (seq2seq)
## 9.1 Text translation
The input goes into an RNN/LSTM and the output is a vector. Then the vector is treated as the input to another RNN/LSTM to generate a sequence. The model was first proposed by Google in 2014. By using LSTM or GRU we overcome the vanishing/exploding gradient problems. In the encoding section, the input tokens are passed through an embedding layer first. The hidden state between encoder and decode captures the meaning of the sentence.

![image](https://github.com/user-attachments/assets/38a3f1b9-ebbb-4de2-a0f8-0c882562e661)

One main problem of this model is that although input and output may have different length, the size of hidden state is fixed and that forms a bottleneck for conveying the message from encoder to decoder. It is not possible to use all hidden states in the encoder due to memory issue as the input sequence becomes larger. The solution is **Attention models**.

One main difference of this model with random generation model is that we do not want the output translation be different every time a same input comes in. So in the decoding part, what we need is to take most probable translation. 
```math
\begin{align}
\arg⁡max_{y^{&lt 1 &gt},…,y^{&lt T_y &gt}}⁡ p(y^{&lt 1 &gt},…,y^{&lt T_y &gt} |x)
\end{align}
```

We do not want to do a greedy approach to pick each word separately solely depending on the previously selected words. The reason is that in many occasions just by following the most probable word at each step, we may drift apart from the best possible translation of the whole sentence.

**Beam Search Algorithm**

Given the vector from encoding part, the algorithm 
- computes $B$ most likely words as the first word for translation.
- Next for each $B$ (beam width) word as $y^{&lt 1 &gt}$, we give it as input to the next block and compute the likelihoods of the outcome. Then we can compute the joint likelihood as $p(y^{&lt 1 &gt} |X)p(y^{&lt 2 &gt} |X,y^{&lt 1 &gt})$ for all words in the vocabulary and pick the top B combinations $y^{&lt 1 &gt} y^{&lt 2 &gt}$.
- The algorithm continues to extract next words until it reaches `<EoS>`.

There is a concern that the product of all these probabilities becomes very small. So it is recommended to use logarithm of probabilities and sum over them.
It is recommended also to normalize the length with respect to $T_y$. In the algorithm there is a parameter α and the argmax is modified as:

```math
\begin{align}
\arg⁡max⁡ \frac{1}{T_y^α}  \sum_{t=1}^{T_y} \log⁡ P(y^{&lt t &gt}│x,y^{&lt 1 &gt},…,y^{&lt t-1 &gt}) 
\end{align}
```

The algorithm repeats the optimization above for some choices of $T_y$ and among all $B$ sequences for all $T_y$ choices, we pick the highest score and report as the output translation.

- High $B$ results better results but slower ($B=10$ is a good choice for a product, $B=100$ or higher is good for research)
- Unlike BFS and DFS, beam search is much faster but there is no guarantee to find exact match
![image](https://github.com/user-attachments/assets/092ae08f-5c58-4d54-a80e-47548f16a28d)

**Beam Search error**

Suppose we have made a translation using Beam search and an RNN. Consider a sentence being translated by human to be $y^{\*}$, and the generated translation by the model to be $\hat{y}$ which is not a very accurate translation. 
To understand the source of error we can compute $P(\hat{y}|x)$ and $P(y^{\*} |x)$ and compare them (To do this, one computes the likelihood of each word when fixing the input). Two cases may occur:

- $P(\hat{y}│x)&lt P(y^{\*} |x)$: then Beam search is failing and causing the error
- $P(\hat{y}│x)&gt P(y^{\*} |x)$: then RNN is not accurate.

We do this comparison for a set of examples to draw a conclusion.

**BLEU Score (Bilingual evaluation understudy)**

In a translation task, there could be several valid translations. To score the translation we may use BLEU metric. If we look into each word (unigram) in the machine translation, we have:
- **Precision** for a word: # of times that appear in references, over the # of words in the machine translation
- **Modified Precision** for a word: Maximum # of times that appear in references (count_clip), over the # of words in the machine translation (count)
Similarly, we can define the metrics above for a combination of two words (bigram). Now let us define Bleu score on m-grams:
```math
\begin{align}
P_m=\log⁡ \frac{\sum_{m_{gram} \in \hat{y}} count_{clip}(m_{gram})}{\sum_{m_{gram}\in \hat{y}} count(m_{gram})}
\end{align}
```
Then we can define combined bleu score and compute precision scores as:
```math
\begin{align}
P=BP \exp⁡ \frac{1}{M} \sum_{m=1:M} P_m
\end{align}
```

Where BP is a coefficient as brevity penalty to penalize short translations:
```math
\begin{align}
BP=min ⁡(1,e^{1-\frac{ref}{cand}})
\end{align}
```

And `ref` and `cand` are number of words in the reference and candidate translation. BLEU score is a precision score.

**ROUGE-N Score**

Assume we have multiple references and one candidate translation. For each reference we compute the counts and take the maximum among them. To compute the score for each reference, we start with each word and if the word exists in the reference, we add the counter. Note that we only add the counter for one recall. ROUGE-N score can be assumed as a recall metric.
One can combine BLEU score and ROUGE-N score to compute a F1 score:
```math
\begin{align}
F1=2\frac{precision \times recall}{precision + recall}
\end{align}
```

## 9.2 Image captioning
Using models such as AlexNet, we can take the last layer to an RNN as the input and generate a sequence of captions. (2014-15)

![image](https://github.com/user-attachments/assets/97043de3-3f08-49c3-b63f-66cccc3c56f0)

## 9.3 Teacher forcing
While training the model, the translation may make a mistake only for one word. If the training is very sensitive to such mistakes, the training becomes very slow. Instead, we allow the model to make such tiny mistakes and we move on. So, when feeding the outputs to the next stage in the decoder section, we feed the target word instead to help the model proceed.
What is Teacher Forcing?. A common technique in training… | by Wanshun Wong | Towards Data Science

[A related article](https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c)

## 9.4	Attention Model (2014)
For short sentence translation the method above works fine. But as the length of sentence increases, the performance drops dramatically since in the simple seq2seq model all the information had to be stored in the hidden state between encoder and decoder. One workaround is to add the hidden state from earlier states. However, if store them separately we run out of memory. So, we can combine them by adding values pointwise:
![image](https://github.com/user-attachments/assets/126f9947-3e62-40ff-ad06-10779488a046)

But this is not optimal. To resolve it, we can use weights and give less weight to hidden states that are more important. In the attention model, we use decoder hidden state to predict the importance weights of the next step. 
Consider the bidirectional LSTM/RNN model. In the first step, we want to extract features from forward and backward recurrences.

Let us define the features $a^{&lt t' &gt}=(\overrightarrow{a}^{&lt t' &gt}, \overleftarrow{a}^{&lt t' &gt})$. 
Next, we can define a forward RNN as below where attention parameters $\alpha^{&lt i,j &gt} construct the context variable $C$, where:
```math
\begin{align}
\sum_{t'} \alpha^{&lt 1,t' &gt}=1\\
\alpha^{&lt 1,t' &gt} \geq 0\\
C^{&lt 1&gt}=\sum_{t'} \alpha^{&lt 1,t' &gt} a^{&lt t' &gt}
\end{align}
```

![image](https://github.com/user-attachments/assets/081338ef-c10b-4cf9-8976-4bbf6e3bee1c)

Same as previous, the outputs y are generated until `<EoS>` appears. The dashed lines are not mandatory specially in cases where there is no strong relation between the characters/tokens in the output.

**How to compute attention weights**

Attention weight $α^(<t,t^'>)$ is defined as the amount of attention $y^{&lt t &gt}$ should pay to $a^{&lt t' &gt}$ and is computed as:
```math
\begin{align}
\alpha^{&lt t,t' &gt}=\exp⁡({e^{&lt t,t' &gt}})/{\sum_{\tau=1}^{T_x} e^{&lt t,\tau &gt}}
\end{align}
```

Where $e^{&lt t,t' &gt}$ are “energies” generated by a neural network with inputs $S^{&lt t-1 &gt}$ and $a^{&lt t' &gt}$.
![image](https://github.com/user-attachments/assets/ff14428e-e4c9-4c11-9990-05f7387ce7bf)

One problem of attention mechanism is that we have $T_x \times T_y$ number of attention coefficients and the computation becomes quadratic in terms of computing these coefficients.

## 9.5 Transcript machine
In speech recognition, it has been shown that working with spectrograms are the best way. The other approach is a 2d spectrogram where x axis is time and the y axis is the spectrum of frequency. Conventionally, linguistic scientists use phonemes where they had lookup tables. But in recent methods spectrogram is the best input. One method that is used is CTC cost (connectionist temporal classification). 

**CTC Cost**

In this use case, the number of inputs is much larger than the transcript output. So many inputs are mapped to the same character. For example, the speech is “the quick brown fox” and the output can be “ttt_h_eee____-____qqq___...”. So it is required to collapse the characters that are nor separated by blank character “-”. That is the intuition behind CTC cost.
![image](https://github.com/user-attachments/assets/f516fe6c-2527-4cd4-8b4e-b49bf6d907f4)

## 9.6 Decoding techniques
In the seq2seq model, we can take several approaches. In a greedy decoding, the decoder takes the most probable word from the softmax output. In practice, this has limitations:
For example a translation instead of “I am hungry” can become “I am am am am …”. Another option is random sampling, meaning that we take a sampling according to the output distribution of softmax. However, some trivial translations become too random. To adjust this we can use a method called temperature. It puts more weights on the probable words and less weight on less probable words. However, a more stable solution is Beam search.

**Beam search decoding**

In this method we keep B (Beam width) number of sequences and compute the probability of those sequences. Note that the softmax output at each stage is conditional probability on the previous stages. This was introduced in previous sections.

**Minimum Bayes Risk (MBR)**

In this method we create multiple candidate translations and to select one candidate among them, we use the following rule:
```math
\begin{align}
E^*=\arg max_{E\in{candidate set}}⁡ \frac{1}{n} \sum_{E'\in{candidate set}} ROUGE(E,E')
\end{align}
```

## 9.7 Tricks for training the seq2seq model
It is difficult to implement a feedback loop to feed in the decoder hidden states back into the attention block in the encoder. Instead, we can use a pre-attention decoder.

![image](https://github.com/user-attachments/assets/a1ada823-23ab-4715-8699-fd5e9dc54164)

To implement this, we concatenate the input sequence tokens and pre-attention decoder containing target sequence and make a shift right and add `<SOS>` to it. Since we use the target vectors for the pre-attention decoder, it means we are using teacher forcing.
The hidden states from encoder is used for both key and value, while the hidden states from the pre-decoder is used as the query in the attention model. Then the context vector is constructed and is fed into the decoder to predict the sequence.
To handle the padding mask, we make a copy of input and target, and use it to determine the paddings before the attention layer is computed. Note that the mask is used to avoid using padding tokens impact computing the probabilities.

![image](https://github.com/user-attachments/assets/f7be0676-ea74-43e5-894a-5e6f0e766542)

Then the context vector is computed and is passed to the decoder to compute the predictions. We use the copy of target for comparison. Note that the mask is discarded before feeding into the decoder. To evaluate the performance, we use BLEU score.

![image](https://github.com/user-attachments/assets/e21e5a05-021d-4d2b-adaa-d7035e7d416a)

The attention block may contain several dense layers which increases the risk of vanishing gradient. To overcome this, we can use a residual network.

![image](https://github.com/user-attachments/assets/4f727b87-3995-41b1-a788-37121d969ef9)

In the implementation of the Attention model, a dense layer is added before and after the attention head and we parallelize and concatenate the outputs according to the structure below:

![image](https://github.com/user-attachments/assets/cbff20d9-9780-4597-b9fd-9fdd70243cb5)

For clarity, note that the parameters that we train are the weights of dense layers and similar to the LSTM block (which is parametrized by the number of units), it does not matter how lengthy is the input. Every token is passed through same block of LSTM and attention is applied when all tokens of an input is passed through the linear layer.

# 10 Transformer
As the models from RNN to LSTM become more complex, it also increases the complexity of learning them. The structure of these models enforces us to compute one input before being able to operate on the next one. Transformer models are introduced to enable us to parallelize part of computations and reduce complexity accordingly. It utilizes attention mechanism and CNN jointly.
A transformer block consists of an attention layer and a one feedforward hidden layer.

## 10.1 Self-Attention
Consider inputs $x_1 \dot x_N$ each vector of dimension d. The attention layer at core is a linear combination of $x_i$s. Then with attention matrix $A$ the outputs are $Z=A^T X$. The limitation for $A\inR^{N×N}$ is that the rows of $A^T$ are normalized having values in $[0,1]$ and sum to 1. The matrix is not fixed size as the input length N can vary. We can perform the normalization row-wise using softmax function, so:
$Z=softmax(B)$, $X=A^T X$

Looking at element $a_ij$ of the attention matrix, we refer to the output $z_i$ as query and the $x_j$ as the key. In self-attention, we want to know how much each words are contributing in the context to their own. So it makes sense to construct matrix $B$ using $\frac{1}{\sqrt{d} XX^T}. However, this gives a high bias towards diagonal.

![image](https://github.com/user-attachments/assets/552d3b81-ee35-4931-999e-bcd0ea88e1d4)

This is not helpful in the transformer since gives weight only to self-tokens. Instead, we multiply matrices $W_Q$, $W_K$ to obtain $Q=X W_Q$ and $K=$XW_K$ to get query and key matrices. $W_Q$  $W_K$ are learnable during training.

The goal is to compute for each word an attention-based vector representation denoted by $A(q,K,V)$ where $q$ is the query, $K$ is key (quality of word given the query) and $V$ is value (specific representation of the word given the query). In self-attention queries and keys come from the same sentence and the goal is to capture a meaning for each word within the sentence. To give an intuition consider the sentence:

<p align=center>
	“Jane is visiting Africa in September.”
</p>

Then for the word “Africa”, query represents a question like “What is happening there?” and the key represents an answer to that question (e.g. “Visiting”). So when $q^{&lt 4 &gt}$. $k^{&lt 3 &gt}$ gets a high value which means these words are relevant to a context. So instead, if treating “Africa” as a single word, it is treating it finds the representation of that as a destination to visit called Africa. The key and value can be seen as a lookup table and we align queries with keys using alignment scores (see $q.k^{&lt i &gt}$ in the formula below). Then we can use alignment scores to compute weights for the weighted sum of the attention model using value vectors.
So, for example for word $x^{&lt 3 &gt}$ we have the representation $A^{&lt 3 &gt}$ where it encapsulates the context as well by looking at the other words around $x^{&lt 3 &gt}$. This is done by computing the representations as:
```math
\begin{align}
A(q,K,V)= \sum_i \frac{exp⁡(q.k^{&lt i &gt}}/{\sum_j exp⁡(q.k^{&lt i &gt} ) v^{&lt i &gt} 
\end{align}
```
where

```math
\begin{align}
q^{&lt i &gt}&=W^Q.x^{&lt i &gt}\\
k^{&lt i &gt}&=W^k.x^{&lt i &gt}\\
v^{&lt i &gt}&=W^V.x^{&lt i &gt}
\end{align}
```

![image](https://github.com/user-attachments/assets/fb11465b-d7ec-444c-b43c-0db4c22609ac)

In matrix form one can only write:
```math
\begin{align}
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\end{align}
```

![image](https://github.com/user-attachments/assets/84248693-728b-4564-b1a3-b917437d7ace)

**Padding**

**Padding Mask**: The length of the sequence is limited and should be the same for all inputs. So we truncate lengthy sentences and pad the short ones. So when we index the words and vectorize them, we can pad them with zeros. However, for softmax, these zeros are problematic. We can build a Boolean masking matrix that tells us which elements we should choose and later force these values to -1e9 to make sure they do not account in the softmax output in the attention blocks. For example, in a causal attention, we don’t want queries to attend to future, so we choose M to be an upper triangular matrix.
```math
\begin{align}
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}}+M)V
\end{align}
```

Where $M$ is the mask matrix with -1e9 values for padded elements and zero otherwise.

**Look-Ahead Mask**: This masking pretends that the model has predicted correctly part of the output and checks if it can correctly predict the next word and after.

## 10.2 Multi-head Attention
Intuitively, we allow the model to learn multiple questions and answers for each word. To implement this for “question-1” consider three matrices: $W_1^Q, W_1^K, W_1^V$, then we build a new set for “question-2”. In the multi-head attention setup, we have h heads. Then for each attention head we compute the value of $Attention(W_i^Q Q,W_i^K K,W_i^V V)$. By concatenation of these heads and multiplying by a matrix $W_o$, we can compute the final attentions:
$Multihead(Q,K,V)=concat(head_1,…,head_h ) W_o$

Note that this operation can be done in parallel which makes t appealing for GPUs. Also note that in the setup bellow, if q,k,v are the same, then this boils down to self-attention

![image](https://github.com/user-attachments/assets/dbb5ba64-ba8e-4d9a-b264-4161992483ad)

Below is a suggestion for matrix sizes:
![image](https://github.com/user-attachments/assets/4677951e-1d30-4d39-a79e-c609338c4e35)

In the transformer model, we add the positional encoding to keep track of order of tokens.

## 10.3 Transformer Decoder model
This structure is also known as GPT-2. It uses the input to predict the next word, for instance it can be used for the application of text summarization. 
![image](https://github.com/user-attachments/assets/ef5a66a3-6b14-4db7-94b8-1f1555710758)
 
Note that the output is a likelihood on the next word. So in the inference to summarize `[Article]<EOS>[Summary]<EOS>`, we provide `[Article]<EOS>` to predict the first word of Summary. Then we give the first predicted word of summary to predict next one and so on. We can use a cross entropy loss to compare the predicted summary and real summary as:
```math
\begin{align}
J=-\frac{1}{m} \sum_{i \in Summary} \sum_{j \in batch \quad elements} y_j^i  \log⁡ \hat{y}_j^i
\end{align}
```

## 10.4 Transformer Detailed Model
In practice and the original paper, the architecture of transformer to do a translation job is described as follow.
In the encoding, we compute the multi-head attention where we can pass the matrix through a feed forward neural network. So it contains contextual semantic embedding (and also positional encoding information according to the complete transformer model).

Then the next block is decoder block which outputs the translation. It starts with `<SoS>` in the output and we feed in <SoS> in the beginning to the decoder. Then using the output of encoder, we generate matrices $K$ and $V$. Together with a $Q$ generated by the multi-head attention on `<SoS>`, we feed it into another multi-head and finally a FF-NN and finally generates the next word in the translation using a softmax activation. To emphasize, in the decoder side, the first multi-head attention is a self-attention and the second one is cross-attentions since it takes $K$,$V$ from encoder side.

Next, we feed in the last translated word to the decoder and generate the next translation.
![image](https://github.com/user-attachments/assets/137e8909-e2f5-4e04-89fb-265a1891d0eb)

There are further blocks in the introduced model which one can read them through the paper. For instance a positional encoding would be provided to encode and decoder to give insight about the position of the words in the sentence, or Add & Norm blocks (similar to batch-norm) that helps speeding up the training. 

![image](https://github.com/user-attachments/assets/82407c78-1d3c-4e9c-9acb-800c5e43c821)

In the original model $N_x$ was 6, but more recent models constitute of 100 or more blocks.
Transformer encoder has bidirectional self-attention while Transformer decoder has unidirectional (left-only-context) self attention.
Positional encoding
In positional encoding we want to store somehow the position of words withing the sentence. To do so, we need a method that can treat any length of sentence and output a vector of desired dimension. To do so, we can use the following positional encoding:
```math
\begin{align}
PE(pos,2i)&=sin⁡(\frac{pos}{10000^{2i/d}})\\
PE(pos,2i+1)&=cos⁡⁡(\frac{pos}{10000^{2i/d}})
\end{align}
```
Where $d$ is the desired output dimension and pos is the position of the word in the sentence. The benefits are:
- The norm of this vector is always the same.
- The norm of the difference between two words with k position difference remains the same.

## 10.5 Transformer vs RNN
The problems with RNN are:
- Parallel computing is difficult.
- For long sequences we have loss of information (even with attention mechanism?)
- RNN has the problem of vanishing gradient
Transformer addresses all three problems above. The model is not handling information sequentially which enables using GPUs.

10.6	Well-known transformer architectures
- GPT-2 Generative pre-training for transformer (developed by OpenAI 2018)
- BERT (Bi-directional Encoder Representation from Transformers) (developed by Google AI Language 2019)
- T5 (Text to Text Transfer Transformer Google 2019)
  * It can perform tasks such as translation, classification, and question and answering.
![image](https://github.com/user-attachments/assets/6c43d4ea-b643-4720-a1a0-e7f99c4310bc)
  * It uses same model to perform all tasks and only the input determines which task is intended.

# 11 Transfer Learning
In NLP area, since the models are very huge and pre-trained on huge datasets (Wikipedia English ~14GB, C4 800 GB), it is more convenient to use the pre-trained model, adapt it to the new use case and run a few more training epochs (fine-tuning) for the new downstream task. Transfer learning can be performed in two ways:
- Feature-based: We store features learned from a previous training (e.g., word2vec embeddings). An  example is ELMo where a contextual representation is built for every token using concatenation of left-to-right and right-to-left representations.
  ![image](https://github.com/user-attachments/assets/fcc23766-168b-4d02-9e50-e71e776689fd)

- Fine-tuning: We use a pre-trained model and use the weights for a new task by adding minimal task-specific parameters and train all parameters for the downstream task (e.g., GPT). The tasks that have been used for pre-training a language model can be predicting masked words, or next sentence.
![image](https://github.com/user-attachments/assets/04a3c25b-6561-4010-a2ec-f635a7ecfbea)

# GPT (Generative Pre-trained Transformers)
It uses stacks of decoder blocks from the transformer model, so given a sequence of tokens, it generates the next token. In practice, it generates the output sequence until it reaches to the stop token. GPT architecture uses unidirectional structure meaning that every token can attend only to its previous tokens in the self-attention layer. The left-context-only version is often referred to as Transformer decoder. 
For pre-training, given every sequence of tokens u_1,…,u_n, the model predicts the next token and the objective function is defined as:
```math
\begin{align}
L_1(U) = \sum_i \log P(u_i|u_{i-k},\dots,u_{i-1};\Theta)
\end{align}
```
After pre-training, they fine-tune the model on classifying a task (such as classification, entailment, similarity, Q&A, multiple-choice). So a labeled dataset C is provided and the objective is:
```math
\begin{align}
L_2(C) = \sum_{(x,y)} \log P(y|x^1,\dots,x^m)
\end{align}
```
Where  
```math
\begin{align}
\log P(y|x^1,\dots,x^m) = softmax(h_l^m W_y)
\end{align}
```

# 13 BERT (Bi-directional Encoder Representation from Transformers)- 2019
It is designed to provide deep bidirectional representation for text input. Then the model can be fine-tuned for other tasks by adding one additional layer on the output. For example, question and answering, language inference. 
BERT uses bidirectional transformer (i.e., Transformer Encoder). Transformer model is able to learn the language context in two “separate” blocks of encoder and decoder. As a result, it enables us to use the encoder separately to build language models that understand context, language, grammar and meaning. This is the idea behind BERT. In short is a stack of encoders:

![image](https://github.com/user-attachments/assets/25d96de2-f146-408d-972f-c8d3b236db8f)

It leverages from
- Transformers model which is using self-attention mechanism
- It is bidirectional.

BERT uses the concept of transfer learning. First it is pre-trained on two tasks and then the model is fine-tuned.

## 13.1 Pre-training
This is done by two objectives. It is relatively expensive procedure in terms of computation. Using 16 TPUs BERT_Base takes 4 days to train. We should note that attention is quadratic to the sequence length.

**Masked Language Model (MLM)**
This objective enables the model to pre-train with a bi-directional attention structure. In previous methods they could train a transformer model using either left-to-right or right-to-left. In ELMo they perform both and finally concatenate both representations. But the difference in BERT is that in each layer of transformer, it uses bidirectional structure (transformer-encoder) and they introduced randomly masking tokens to prevent a token to “see itself”. Without masking, since the architecture is multi-layer, in layer 2 for every token has some information about itself using the output of other tokens in the previous layer.

In this task, we give the model sentences with masked words and the model should be able to predict masked words. The strategy is to mask 15% of the words at random. With 80% probability we use the <mask> token, and 10% we use a random token and 10% we choose the original token. Then we use the cross-entropy loss.

The reason for not having all masked token be <mask> is that during fine-tuning such tokens won’t exist and the model’s performance would be low. Having intact tokens when masking is justified because we want the model to learn the true context representation as well. On the other hand, random tokens try to show the model that it fails if it uses wrong token in the context. Since the percentage of such occurrences is very low, it is not hampering the performance much.

**Next Sentence Prediction (NSP)**

In this task, we give sentence A and B to the model and see if sentence B follows sentence A or not. The loss function is a binary loss.
This process is slow.
![image](https://github.com/user-attachments/assets/1f8f35c7-dd8f-4742-9138-9cdd61113089)

In practice these tasks are performed simultaneously and we add the loss functions. So, we input the model two sentence with maksed words. The output C then represents the binary classification whether Sentence B follows sentence A or not (NSP task). Es are embeddings and T are word vectors which is used for MLM task. 
The embeddings are in fact constructed from 3 different embeddings:
- Token Embedding: Uses an embedding like WordPieces which consists of 30k vocab)
![image](https://github.com/user-attachments/assets/85d7dd47-66d2-41f8-9bd4-a8a739a6e16e)
- Segment Embedding: indicates the sentence 
 ![image](https://github.com/user-attachments/assets/b2cc37e5-36ba-490f-bc7e-786d458011f5)
- Position Embedding:
 ![image](https://github.com/user-attachments/assets/96215faf-ad86-4465-ace1-ce4bdb989480)

The word vectors $T_i$ have the same size and are generated simultaneously. To define the loss function, we pass the output through a fully connected layer with output dimension 30k and Softmax activation. Then we can compare this likelihood vector with a one-hot code using the true sentence and compute cross entropy loss. In this computation we only consider the prediction for masked words and ignoring the other values.
![image](https://github.com/user-attachments/assets/86714052-c73c-4aee-bb6b-6650525c2ebd)

BERT_base model contains 12 layers and 12 attention heads with 110 million parameters. 

## 13.2 Fine tuning
We can fine tune BERT models with a more specific language task, for instance Q&A. So, we can replace the output layer with a fresh set and train the weights of the last layer while very slightly changing the weights of the rest (or maybe even fix them?!). Then the BERT model receives a question and a passage and marks the output with start and end of span of text that has the answer. So the objective is how accurate the start an end are predicted. We can train this using a Q&A dataset. This process is fast and can be done in about 1h Google cloud TPU (Tensor Processing Unit) or few hours GPU.
![image](https://github.com/user-attachments/assets/7ac6743e-8383-41df-a460-b3a52d18c77e)

In practice, this can be done by giving the `[question<SEP>passage]` and mark the start and end in the output word vectors that covers the answer. 
![image](https://github.com/user-attachments/assets/6447c575-8074-423e-9cb0-8317875165e2)

BERT was initially released in two models: BERT_base with 110M parameters and BERT_large with 340M parameters.

# 14 T5 model
This model (220 million parameters) can be used for tasks:
- Classification
- Q&A
- Machine Translation
- Summarization
- Sentiment analysis
See the paper [“Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer”](https://arxiv.org/pdf/1910.10683.pdf).
In the model, we have a fully visible masking encoder (not caual) and it contains a causal decoder.
![image](https://github.com/user-attachments/assets/671a7b9c-8648-45db-8853-09c49c95786d)
The model includes a language model with causal links and a Prefix model with mixed fully visible and causal links:
![image](https://github.com/user-attachments/assets/e028f31f-ef27-4620-8e34-b3c3681e86d5)

## 14.1 Multi-task training strategies
To train one model on multiple tasks, we can use tags to identify which task we are intended.

**Data sampling**

We can either use proportional mixing (taking a fixed percentage of different data sets), or we can fix the final sample size taken from each dataset. 

**Training phase**

We can use gradual unfreezing, meaning that at every step of training, we only unfreeze part on layer and the rest are fixed.
Adaptive layer is also another concept where we add feed-froward neural networks to adapt the output size to the input and during the fine tuning, we can only learn those parameters.

# 15 LLM downstream tasks
## 15.1 Named Entry Recognition NER
In many applications we need to make a differentiation between the names and other words in the document. 
![image](https://github.com/user-attachments/assets/9e25a342-87d2-4162-be18-5ba839725dc0)

## 15.2	Question Answer
Question answering (QA) is a task of natural language processing that aims to automatically answer questions. The goal of extractive QA is to identify the portion of the text that contains the answer to a question. For example, when tasked with answering the question 'When will Jane go to Africa?' given the text data 'Jane visits Africa in September', the question answering model will highlight 'September'.

For this purpose, there is a good dataset called “bAbI” developed by Facebook.
[https://research.facebook.com/downloads/babi/](https://research.facebook.com/downloads/babi/)

# 16 LLM evaluation
## 16.1 GLUE General Language Understanding Evaluation
It is a benchmark to evaluate the performance of a language model. It is a collection of train, evaluate, and analyze natural language understanding systems. It has different datasets of different sizes.

# 17 Reformer model (Long Sequence handling)
Reformer model is a modification on transformer model to handle long text inputs and save memory issues. The model uses:
- Local sensitivity hashing (LSH) to efficiently compute attention values
- Reversible residual layers to better use the memory

When dealing with use cases like writing a book or chatbot, we need to be able to handle long sequences. The problem is that we run out of memory.
- Attention on a sequence of length $L$ takes $L^2$ time and memory (since we are comparing words with each other)
- $N$ layers takes $N$ times as much memory (GPT-3 has 96 layers)

The attention inputs $Q,K,V$ are of dimension $[L,d_model ]$. Note that $Attention=softmax(QK^T)V$ and $QK^T$ is $[L,L]$. We may not need to consider all $L^2$ values and we can save memory by computing only part of these multiplications. Activations need to be stored for backpropagation which takes a lot of memory. So instead, we can re-compute them.

## 17.1 Parallel computing using LSH
The trick is to use local sensitivity hashing. In computing attention, we want to find $q$ and $k$ to be close. Instead of the inner product, we can bucket them, use local sensitivity hashing. This also allows us to parallelize the process.
![image](https://github.com/user-attachments/assets/44c57c04-e7ed-45ed-8a03-b9848c228dc7)

![image](https://github.com/user-attachments/assets/4607c5c7-07be-4780-a704-8332e47efee8)

![image](https://github.com/user-attachments/assets/72b35fb8-5596-4f79-a739-f2ec1d277cce)

![image](https://github.com/user-attachments/assets/3ef6f414-16c0-4535-993c-4131daee9949)

## 17.2 Memory saving in back propagation (Reversible residual layer)
In a simple transformer model, we have residual links that in backpropagation, we need to store the value to compute the backpropagation by subtracting them.
 
Instead, we can compute the residuals again in backpropagation to mitigate the memory issue. The trick is the architecture below that at each iteration we update only one of the values in columns:
 
```math
\begin{align}
y_1&=x_1+Attention(x_2)\\
y_2=&x_2+feedforward(y_1)
\end{align}
```

So we save $y_1$ and $y_2$ only and we don’t need to store the activation outputs. Then to recompute $x_1$ and $x_2$:
```math
\begin{align}
x_2&=y_2-Feedforward(y_1)\\
x_1&=y_1-Attention(x_2)
\end{align}
```

# 18 References
- Coursera Specialization course for seqence models (by DeepLearning.AI)
- Natural Language Processing Specialization (by DeepLearning.AI)
- Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy (GitHub: karpathy) [link](https://gist.github.com/karpathy/d4dee566867f8291f086)
- The Unreasonable Effectiveness of Recurrent Neural Networks (Andrej Karpathy blog, 2015) [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [deepjazz (GitHub: jisungk)](https://github.com/jisungk/deepjazz) 
- [Learning Jazz Grammars (Gillick, Tang & Keller, 2010)](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
- A Grammatical Approach to Automatic Improvisation (Keller & Morrison, 2007) [link](http://smc07.uoa.gr/SMC07 Proceedings/SMC07 Paper 55.pdf)
- [Surprising Harmonies (Pachet, 1999)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)
- Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (Bolukbasi, Chang, Zou, Saligrama & Kalai, 2016) [Link](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)
- GloVe: Global Vectors for Word Representation (Pennington, Socher & Manning, 2014) [Link](https://nlp.stanford.edu/projects/glove/)
- [Woebot](https://woebothealth.com/)
- Attention Is All You Need (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, 2017) [Link](https://arxiv.org/abs/1706.03762)
- “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer” 1910.10683.pdf (arxiv.org) [Link](https://arxiv.org/pdf/1910.10683.pdf)


