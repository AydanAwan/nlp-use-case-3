# nlp-use-case-3

project overview

this project is about text summurivation through abstractive summarization

plan 

1. first thing we wnat to do is install the packages we need for this project, they are as follow:
install python 3.7 as some of the packages dont work for later versions of python
 bert-extractive-summarizer
 spacy
 transformers
 torch
 sentencepiece
 numpy (needed to install gensim)
 gensim 3.8.3 (as this is the last version that has the summarize function that we will be using)

2. next we want to import the packages we need for this which will be the following:
import gensim
from gensim.summarization import summarize
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, T5Config
from summarizer import Summarizer, TransformerSummarizer

 resources

 1. sentencepiece
 https://github.com/google/sentencepiece#c-from-source