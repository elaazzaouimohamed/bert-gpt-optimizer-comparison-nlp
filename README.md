# BERT & GPT Optimizer Benchmarking for NLP

This project evaluates and compares multiple optimization algorithms — **AdamW**, **LAMB**, **SGD**, and **AdaFactor** — with learning rate warmup for fine-tuning transformer-based models (**BERT** and **GPT**) on the **SST-2 sentiment classification** task.

While testing, **SGD consistently yielded poor accuracy and instability** on the SST-2 dataset. To further investigate, we attempted fine-tuning the **GPT model** using **SGD** on the **SQuAD (Stanford Question Answering Dataset)**. Unfortunately, the results confirmed the trend — **accuracy remained low**, suggesting that SGD is not well-suited for fine-tuning large transformer models in this context.



## 🚀 Project Highlights

- 📚 Models: BERT, GPT (HuggingFace Transformers)
- ⚙️ Optimizers: AdamW, LAMB, SGD, AdaFactor (with Warmup)
- 📊 Evaluation Criteria:
  - Precision (Accuracy)
  - Training Stability
  - Resource Consumption (GPU/Memory/Time)
- 🧪 Datasets: GLUE SST-2 (Stanford Sentiment Treebank v2), SQuAD (Stanford Question Answering Dataset)

## 🧰 Requirements

- Python 3.8+
- Transformers
- Datasets
- PyTorch
- NumPy, pandas, matplotlib

Install them via:

```bash
pip install torch transformers datasets numpy pandas matplotlib
```

## 📝 How to Run

```python
# Load dataset and tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

Then follow each notebook (`BERT MODEL/`, `GPT MODEL/`) to run fine-tuning with a specific optimizer.

## 🤝 Contributing
Pull requests and issue discussions are welcome. If you'd like to add more optimizers or other datasets, feel free to fork the repo!
