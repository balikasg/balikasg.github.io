---
title:  "Introducing tf-exporter"
categories: ["Python", "transformers", "sentence-transformers", "tensorflow"]
tags: ["Python", "machine learning"]
mathjax: false
---

# What is `tf-exporter`?
In this post I am introducing my new open-source library: `tf-exporter` [available at Github](https://github.com/balikasg/tf-exporter)!  

Simply put, 
its goal is to persist a sentence-transformer model *and* its tokenizer as a single tensorflow 
graph. This is a nice thing to have especially in production settings where we want to serve the model
because instead of maintaining and ensuring alignment of two artifacts (tokenizer and actual model) we care for one. 
For an NLP model for example, `tf-exporter` saves a single tensorflow graph that we can query with string tensors directly: 

```python
tf_graph = tf.saved_model.load("path-where-tf-exporter-saved-a-model")
serving_signature = tf_graph.signatures["serving_default"]
serving_signature(input_sequence=tf.constant([["This is a test"]]))
# returns the model output directly
```


# Some Context
Transformer models are deep learning neural network models that transform an input to either a 
vector or a category or a score etc. A transformer model comes with a dedicated tokenizer. As a result,
to get predictions for an NLP task for example does two steps: 
1. Calls the tokenizer 
2. Feeds the tokenizer output to the actual deep learning model

In terms of code for example using the popular (sentence-transformers)[https://github.com/UKPLab/sentence-transformers] 
package we need: 
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

features = model.tokenize(["this is a test sentence"])
# features are 
# {'input_ids': tensor([[ 101, 2023, 2003, 1037, 3231, 6251,  102]]),
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]]),
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

model.forward(features)
# Returns the sentence embeddings by doing the transformers forward pass
```
Of course, in terms of python this can be easily done in a single function call but in a 
production setting someone would need to maintain both separately (unless it is a fastapi implementation
where python controls everything). But if is to be served from JVM or from a triton cluster for example, 
having fewer things to maintain is something we want! 

# How is it done? 
One of the most rewarding parts of the project was to deep dive on the sentence-transformers code
and understand its architecture. It is easier to describe it if we first look at what a model directory 
consists of: 
```bash
ls ~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2
1_Pooling                         config.json                       data_config.json                  pytorch_model.bin                 special_tokens_map.json           tokenizer_config.json             vocab.txt
README.md                         config_sentence_transformers.json modules.json                      sentence_bert_config.json         tokenizer.json                    train_script.py
```
Among these files and directories `tokenizer_config.json` and `modules.json` contain important information 
about the task. The first:
```bash
# tokenizer_config.json
{"do_lower_case": true, "unk_token": "[UNK]", "sep_token": "[SEP]",
 "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]",
  "tokenize_chinese_chars": true, "strip_accents": null, 
  "name_or_path": "nreimers/MiniLM-L6-H384-uncased", "do_basic_tokenize": true,
  "never_split": null, "tokenizer_class": "BertTokenizer", "model_max_length": 512}
```
shows all the important information to create a tokenizer. The idea is to get the information from 
this config and the vocabulary of the tokenizer to create the same functionality with tensorflow. 

The second defines the model's architecture (it is a Transformer model and some normalization
or further dense layers on top of it).
```bash
# modules.json
[
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Normalize",
    "type": "sentence_transformers.models.Normalize"
  }
]
```
We see here for example that the `all-MiniLM-L6-v2` consists of a Transformer whose outputs are pooled and normalized. 
Also, the `path` of each element in the json shows where further specifications and artifacts for a given module reside. 
See for example is the `ls` above that there is a directory named `1_Pooling` that contains the pooler's information. 

So what sentence-transformers does to get predictions it forward's sequentially an input over the modules 
listed in the `modules.json` file. Which is exactly what we need to implement in the conversion code: 
We need to parse these configs and instantiate the same modules as tensorflow code and then persist it. A challenge is that, often, 
the weights of these individual modules can be in pytorch. Getting this to tensorflow can require some work.
