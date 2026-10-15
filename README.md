# CottonDoctor-PK

Cotton leaf disease classifier from a phone photo. Upload a leaf image — get the predicted disease, confidence, and farmer-friendly treatment advice.

Cotton is Pakistan's key cash crop, and leaf diseases (especially Cotton Leaf Curl Virus) repeatedly devastate yields in Punjab. Small farmers can't afford agronomists, so a photo-diagnosis tool is genuinely useful.

## Dataset

[Cotton Leaf Disease Classification Bangladesh 2](https://huggingface.co/datasets/Project-AgML/cotton_leaf_disease_classification_bangladesh_2) — public, ungated, **CC-BY-4.0**. Raw split: 2,137 images (800×800 RGB, 0 corrupt), 7 classes:

| Class | Images |
|---|---|
| Bacterial Blight | 250 |
| Curl Virus | 431 |
| Healthy Leaf | 257 |
| Herbicide Growth Damage | 280 |
| Leaf Hopper Jassids | 225 |
| Leaf Redding | 578 |
| Leaf Variegation | 116 |

Honest note: the dataset is **Bangladesh-grown cotton** (conditions similar to Punjab/Sindh) — the method transfers, but this is not Pakistani field data.

## Method

Transfer learning with EfficientNet-B0 (ImageNet pretrained, PyTorch): frozen backbone + new 7-class head trained 15 epochs, then the last blocks unfrozen for 6 more epochs (Adam, class-weighted loss for the imbalance). Stratified split: 1,495 train / 321 val / 321 held-out test. The test set was evaluated once, at the end.

## Results (held-out test, n=321)

| Model | Accuracy |
|---|---|
| Majority-class baseline | 27.1% |
| Color-histogram + logistic regression | 80.7% |
| **EfficientNet-B0 (this model)** | **89.7%** |

Per-class performance:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Bacterial Blight | 0.825 | 0.892 | 0.857 | 37 |
| Curl Virus | 0.969 | 0.969 | 0.969 | 65 |
| Healthy Leaf | 0.775 | 0.974 | 0.864 | 39 |
| Herbicide Growth Damage | 0.977 | 1.000 | 0.988 | 42 |
| Leaf Hopper Jassids | 0.903 | 0.824 | 0.862 | 34 |
| Leaf Redding | 0.971 | 0.782 | 0.866 | 87 |
| Leaf Variegation | 0.696 | 0.941 | 0.800 | 17 |

Weakest spot: Leaf Variegation precision (0.696) — only 17 test samples, so it gets confused with other spotted-leaf classes. More field photos would fix this.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

`model.pt` (16.3 MB, state_dict) must sit next to `app.py`.

Retrain from scratch (Colab-ready, GPU optional): open `train.ipynb` — it downloads the data straight from Hugging Face, no Google Drive needed. `train.py` is the same pipeline as a script.

## Disclaimer

AI guidance only — confirm with a local agronomist before treating a crop.

**Author:** [Rayyan Khan](https://rayyankhan.online) — Machine Learning Engineer, Pakistan
