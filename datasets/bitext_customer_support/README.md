# Bitext Customer Support

Real customer-support utterances with intent labels. Pulled from HuggingFace.

- **Source:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **License:** check the HuggingFace dataset card before redistributing.
- **Size:** ~27K utterances; we cache a subset for the lab.
- **Labels:** the dataset has 27 fine-grained intents grouped into 11 coarse **categories**. The L1 lab uses the `category` column and restricts it to **5 categories** (`ACCOUNT, ORDER, REFUND, PAYMENT, DELIVERY`, set in `cache_datasets.BITEXT_CATEGORIES`) to keep the confusion matrix readable and the task non-trivial.
- **Why this dataset:** closest thing to ITSM that's openly available. Real customer-support flavor, real intent labels.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
```

A cached subset is in `data.csv` for offline use.
