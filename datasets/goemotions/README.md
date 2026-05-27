# GoEmotions

58K Reddit comments labelled with emotions. Pulled from HuggingFace.

- **Source:** `go_emotions` (Google Research)
- **License:** Apache 2.0 (check the dataset card before redistributing)
- **Size:** ~58K comments; we cache a subset for the lab.
- **Labels:** multi-label, 27 emotions + neutral. Heavy-tailed — `joy` and `neutral` are very common, `grief` and `pride` very rare.
- **Why this dataset:** truly multi-label, naturally skewed. Perfect for the rare-label-noise demo.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("go_emotions", split="train")
```

A cached subset is in `data.csv` for offline use.
