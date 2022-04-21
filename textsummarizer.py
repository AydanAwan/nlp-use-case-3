

from gensim.summarization import summarize
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from summarizer import Summarizer, TransformerSummarizer

# pipeline api methods

text = "Tesla reported second-quarter earnings after the bell Monday, and it’s a beat on both the top and bottom lines. Shares rose about 2% after-hours. Here are the results. Earnings: $1.45 vs 98 cents per share adjusted expected, according to Refinitiv. Revenue: $11.96 billion vs $11.30 billion expected, according to RefinitivTesla reported $1.14 billion in (GAAP) net income for the quarter, the first time it has surpassed $1 billion. In the year-ago quarter, net income amounted to $104 million. Overall automotive revenue came in at $10.21 billion, of which only $354 million, about 3.5%, came from sales of regulatory credits. That’s a lower number for credits than in any of the previous four quarters. Automotive gross margins were 28.4%, higher than in any of the last four quarters. Tesla had already reported deliveries (its closest approximation to sales) of 201,250 electric vehicles, and production of 206,421 total vehicles, during the quarter ended June 30, 2021. The company also reported $801 million in revenue from its energy business, including solar photovoltaics and energy storage systems for homes, businesses and utilities, an increase of more than 60% from last quarter. While Tesla does not disclose how many energy storage units it sells each quarter, in recent weeks CEO Elon Musk said, in court, that the company would only be able to produce 30,000 to 35,000 at best during the current quarter, blaming the lag on chip shortages. Tesla also reported $951 million in services and other revenues. The company now operates 598 stores and service centers, and a mobile service fleet including 1,091 vehicles, an increase of just 34% versus a year ago. That compares with an increase of 121% in vehicle deliveries year over year. A $23 million impairment related to the value of its bitcoin holdings was reported as an operating expense under “Restructuring and other.”"
summary_by_ratio = summarize(text, ratio=0.15)
print("ratioSummary : \n" + summary_by_ratio)

summary_by_count = summarize(text, word_count=60)
print("countSummary : " + summary_by_count)

summarization = pipeline("summarization")
abstract_text = summarization(text)[0]['summary_text']
print("pipelineSummary:", abstract_text)

t5summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
print(t5summarizer(text, min_length=5, max_length=60))

# t5 transformer

t5model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

t5tokenized_text = t5tokenizer.encode("summarize:" + text, return_tensors="pt").to(device)
t5summary_ids = t5model.generate(input_ids=t5tokenized_text, num_beams=3, min_length=20, max_length=70, repetition_penalty=2.0, early_stopping=True)

output = t5tokenizer.decode(t5summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("T5Summary:", output)

# bert model
bert_model = Summarizer()
bert_summary = ''.join(bert_model(text, min_length=60))
print("bertSummary: " + bert_summary)

# gpt2 model
GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
gpt_summary = ''.join(GPT2_model(text, min_length=60))
print("GPT2Summary: " + gpt_summary)

# XLNet model
xlnet_model = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
xlnet_summary = ''.join(xlnet_model(text, min_length=60))
print("xlnetSummary: " + xlnet_summary)
