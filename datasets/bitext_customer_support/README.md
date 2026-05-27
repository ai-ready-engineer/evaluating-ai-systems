# Bitext Customer Support

Real customer-support utterances with intent labels. Pulled from HuggingFace.

- **Source:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **License:** check the HuggingFace dataset card before redistributing.
- **Size:** ~27K utterances; we cache a subset for the lab.
- **Labels:** multi-class intent (27 categories: billing, cancellation, refund, delivery, account, change order, contact, recover password, …).
- **Why this dataset:** closest thing to ITSM that's openly available. Real customer-support flavor, real intent labels.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
```

A cached subset is in `data.csv` for offline use.
