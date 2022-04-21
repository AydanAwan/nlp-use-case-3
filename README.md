# nlp-use-case-3

project overview

this project is about text summurivation through abstractive summarization, taking a wall of text and extracting the most important information from it. we will be doing this using 5 different models as follow:
• Pipeline API
• T5 Transformer
• BERT
• GPT2
• XLNet


plan 

1. first thing we wnat to do is install the packages we need for this project, they are as follow:
install python 3.7.2 as some of the packages dont work for other versions of python
on this note we will be setting up a virtual enviroment but you dont have to and if you dont skip the next step

2. i set up my enviroment with venv  which requires you to have only one version of python installed but you can also use the virtualenv package which works with multiple versions

3. next you want to install the requirements.txt
the key packages are as follows
 bert-extractive-summarizer
 spacy
 transformers
 torch
 sentencepiece
 numpy (needed to install gensim)
 gensim 3.8.3 (as this is the last version that has the summarize function that we will be using)
 tensorflow

4. next we want to import the packages we need for this which will be the following:
from gensim.summarization import summarize
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from summarizer import Summarizer, TransformerSummarizer

5. now we are ready to start our summarization. first thing we need to do is get our data we are going to work on this has  just been taken as a block of text contained within a string

6. next we can try out some simple summarizations using gensims sumarize function which simply ranks words by how rfrequently they appear in the text then determins that the sentances with the most commen words are the most important and summarizes it that way
we can then chose to summarize buy word count or ratio showing the top sentances up to a word limit either with a set number or a % of total words

7. with that done we can move on to some more advanced models starting with the pipeline api from transformers package [2] which is a simple api which we can use to get and use a number of summarization models the first one we will use is the standard model then we will use t5 method [3] with t5 base [4] that in very simple terms is a model that is pre trained on a hdge data rich task making it applicable in many other text to text transformations and the t5 base we are using is a model trained using a multi task mixture of both supervised and unsupervised tasks see resources for more info
we create are model with the task of summarization using the t5-base model for both the tokenization and the task and we use tensorflow framework to carry out the mathermatiacal computations then we set the max and min words for the summarization

8. nest lets make a t5 model using the t5 transformer instead of pipeline this means doing each part ourselves so first we have to generate the tokenizer and the model adn we are using t5-small [5] for this. the search method we use is beam search [6] which reduces the probability of missing hidden high priority words and the parameters work as follow:
• max_length: The maximum number of tokens to generate.
• min_length: This is the minimum number of tokens to 
generate.
• length_penalty: Exponential penalty to the length, 1.0 
means no penalty, increasing this parameter, will increase 
the length of the output text.
• num_beams: Specifying this parameter, will lead the model to 
use beam search instead of greedy search, setting 
num_beams to 4, will allow the model to lookahead for 4 
possible words (1 in the case of greedy search).
• early_stopping: We set it to True, so that generation is 
finished when all beam hypotheses reached the end of 
string token (EOS). 

9. next we will be using the bert-extractive-summarizer to create a bert, gpt2 and XLNet model these are writen basicly the same way but the methods work as follows
• BERT (Bidirectional transformer) is a transformer used to 
overcome the limitations of RNN and other neural networks as Long 
term dependencies. It is a pre-trained model that is naturally 
bidirectional. This pre-trained model can be tuned to easily perform 
the NLP tasks as specified, Summarization in our case.[7]

• Generative Pre-trained Transformer (GPT) models by OpenAI have 
taken natural language processing (NLP) community by storm by 
introducing very powerful language models. These models can 
perform various NLP tasks like question answering, textual 
entailment, text summarization etc. without any supervised training.[8]

• XLNet is a generalized autoregressive language model that learns 
unsupervised representations of text sequences. This model 
incorporates modelling techniques from Autoencoder(AE) 
models(BERT) into AR models while avoiding limitations of AE.[9]

 resources

1. sentencepiece
https://github.com/google/sentencepiece#c-from-source

2. transformers pipeline
https://huggingface.co/docs/transformers/main_classes/pipelines

3. transformers t5 model
https://huggingface.co/docs/transformers/model_doc/t5

4. transformers t5 base
https://huggingface.co/t5-base

5. t5-small
https://huggingface.co/t5-small

6. search methods
https://huggingface.co/blog/how-to-generate

7. bert model
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

8. gpt2 model
https://huggingface.co/gpt2
https://en.wikipedia.org/wiki/GPT-2

9. XLNet model
https://huggingface.co/docs/transformers/model_doc/xlnet