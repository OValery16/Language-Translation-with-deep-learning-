# Language Translation with deep learning 

## Project purpose

For this project we build a RNN sequence-to-sequence learning in Keras to translate a language A to a language B.

## Language and Dataset

Since I am french, I choose to translate english to french. However our system is pretty general and accepts any other language pair (e.g. english/french). By defauft, we use ANKI dataset which can be easy download [there](http://www.manythings.org/anki/)

## What is Sequence-to-sequence learning ?

Sequence-to-sequence learning (Seq2Seq) is about training models to convert sequences from one domain to sequences in another domain. It works as following:

1. We start with input sequences from a domain (e.g. English sentences) and correspding target sequences from another domain
    (e.g. French sentences).
2. An encoder LSTM turns input sequences to 2 state vectors (we keep the last LSTM state and discard the outputs).

3. A decoder LSTM is trained to turn the target sequences into the same sequence but offset by one timestep in the future,     a training process called "teacher forcing" in this context.  Is uses as initial state the state vectors from the encoder.     Effectively, the decoder learns to generate `targets[t+1...]` given `targets[...t]`, conditioned on the input sequence.
	
4. In inference mode, when we want to decode unknown input sequences, we:
    * Encode the input sequence into state vectors
    * Start with a target sequence of size 1 (just the start-of-sequence character)
    *	Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character
    * Sample the next character using these predictions (we simply use argmax).
    * Append the sampled character to the target sequence
    * Repeat until we generate the end-of-sequence character or we hit the character limit.
	
For more information, please check these papers:

	* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
    * [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

## How to use it

1. Downlow you training dataset
2. Update path and the number of training example
3. Run ```python3 training.py ```	
4. Prediction with ```python3 predictionTranslation.py```
	
## LSTM or GRU

By default, the model runs with LSTM cell (long short term memory), but we also provide the user the opportunity to use instead GRU cell. (GRU cell only include 1 gate which is meke the training faster)

## Downloading weights

We trained this model on the complete English/French dataset. The all training takes weeks. But we got promising results after 18 h of training (20 epoch). You can download our weights [there](https://drive.google.com/open?id=12s5KVDXex1Icy5FeFMLtQ2ADuWupzG_u)

## Our result 

For sure, our system is far from being as accurate as Google Transle. But after 20 epoch only, it reconnizes accurately short sentences.

Example of output:

``` Input sentence: I love you.``` 

``` Decoded sentence: Je t'aime !``` 

It is accurate.

``` Input sentence: We studied.``` 

``` Decoded sentence: Nous étudions.``` 

It is accurate.

``` Input sentence: I slept well. ```

``` Decoded sentence: J'ai dormi toute la journée. ```

Same meaning, but the translation is not fully accurate. The right translation would be "j'ai bien dormi"

``` Input sentence: He worked a lot.``` 
``` Decoded sentence: Il a travaillé pour un homme riche.``` 

The translation is not correct.

## Conclusion

To conclude, our network learnt the basic concept of english/french, but it still requires two things:

1. A longer training time
2. A deeper architecture, such as more LSTM cells









