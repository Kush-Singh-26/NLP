# Transformer Implementation

This is the implementation of the [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762).

- Import the required libraries

```sh
pip install -r requirements.txt
```

- Import the model and the tokenizer file

```sh
python download.py
```

- Perform Inference

```sh
python inference.py -g "Climate change is one of the biggest challenges facing humanity today."
python inference.py -b 5 "Climate change is one of the biggest challenges facing humanity today."
```

- `-g` : greedy decoding
- `-b 5` : beam search with beam size 5