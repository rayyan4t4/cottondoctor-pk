import io, json, os, random, shutil, sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CLASSES = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage",
           "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]


def extract():
    t = pq.read_table(os.path.join(ROOT, "raw-train.parquet"))
    labels, imgs = t.column("label").to_pylist(), t.column("image").to_pylist()
    raw = os.path.join(DATA, "raw")
    for c in range(7):
        os.makedirs(os.path.join(raw, str(c)), exist_ok=True)
    counts = Counter()
    for i, (lab, img) in enumerate(zip(labels, imgs)):
        im = Image.open(io.BytesIO(img["bytes"])).convert("RGB").resize((256, 256))
        im.save(os.path.join(raw, str(lab), f"{i}.jpg"), quality=90)
        counts[lab] += 1
    print("extracted:", dict(sorted(counts.items())))
    return raw


def split(raw):
    items = [(c, f) for c in os.listdir(raw) for f in os.listdir(os.path.join(raw, c))]
    y = [int(c) for c, _ in items]
    tr, te = train_test_split(items, test_size=0.15, random_state=SEED, stratify=y)
    ytr = [int(c) for c, _ in tr]
    tr, va = train_test_split(tr, test_size=0.1765, random_state=SEED, stratify=ytr)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        for c, f in part:
            d = os.path.join(DATA, name, c)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(raw, c, f), os.path.join(d, f))
    sizes = {n: sum(len(fs) for _, _, fs in os.walk(os.path.join(DATA, n))) for n in ["train", "val", "test"]}
    print("split sizes:", sizes)


def files_labels(split_name, size=64):
    xs, ys = [], []
    base = os.path.join(DATA, split_name)
    for c in sorted(os.listdir(base)):
        for f in os.listdir(os.path.join(base, c)):
            im = Image.open(os.path.join(base, c, f)).convert("RGB").resize((size, size))
            xs.append(np.asarray(im)); ys.append(int(c))
    return np.array(xs), np.array(ys)


def baselines():
    xtr, ytr = files_labels("train"); xte, yte = files_labels("test")
    maj = Counter(ytr).most_common(1)[0][0]
    maj_acc = accuracy_score(yte, np.full_like(yte, maj))
    def hist(x):
        return np.stack([np.histogramdd(p.reshape(-1, 3), bins=8, range=((0,256),)*3)[0].ravel() for p in x])
    htr, hte = hist(xtr), hist(xte)
    sc = StandardScaler().fit(htr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(htr), ytr)
    hist_acc = accuracy_score(yte, clf.predict(sc.transform(hte)))
    print(f"baseline majority acc: {maj_acc:.4f} | hist+logreg acc: {hist_acc:.4f}")
    return {"majority": round(float(maj_acc), 4), "hist_logreg": round(float(hist_acc), 4)}


class LeafDS(Dataset):
    def __init__(self, split_name, tf):
        self.items = [(os.path.join(DATA, split_name, c, f), int(c))
                      for c in sorted(os.listdir(os.path.join(DATA, split_name)))
                      for f in os.listdir(os.path.join(DATA, split_name, c))]
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def train():
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tr_tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    ev_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    tr = DataLoader(LeafDS("train", tr_tf), batch_size=32, shuffle=True, num_workers=2)
    va = DataLoader(LeafDS("val", ev_tf), batch_size=64, num_workers=2)
    te = DataLoader(LeafDS("test", ev_tf), batch_size=64, num_workers=2)

    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 7)
    net.to(DEVICE)
    counts = Counter(y for _, y in LeafDS("train", ev_tf))
    w = torch.tensor([len(tr.dataset) / counts[c] for c in range(7)], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)

    def run(epochs, lr, params):
        opt = torch.optim.Adam(params, lr=lr)
        best, best_state, wait = 0, None, 0
        for ep in range(epochs):
            net.train()
            for x, y in tr:
                opt.zero_grad()
                crit(net(x.to(DEVICE)), y.to(DEVICE)).backward()
                opt.step()
            net.eval(); correct, total = 0, 0
            with torch.no_grad():
                for x, y in va:
                    p = net(x.to(DEVICE)).argmax(1).cpu()
                    correct += (p == y).sum().item(); total += len(y)
            acc = correct / total
            print(f"epoch {ep+1}/{epochs} val_acc {acc:.4f}", flush=True)
            if acc > best:
                best, best_state, wait = acc, {k: v.cpu().clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 4: break
        net.load_state_dict(best_state)
        return best

    for p in net.features.parameters(): p.requires_grad = False
    print("phase 1: head only"); run(15, 1e-3, net.classifier.parameters())
    for p in list(net.features.parameters())[-20:]: p.requires_grad = True
    print("phase 2: last blocks"); run(6, 1e-4, [p for p in net.parameters() if p.requires_grad])

    net.eval(); allp, ally = [], []
    with torch.no_grad():
        for x, y in te:
            allp += net(x.to(DEVICE)).argmax(1).cpu().tolist(); ally += y.tolist()
    acc = accuracy_score(ally, allp)
    rep = classification_report(ally, allp, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(ally, allp).tolist()
    metrics = {
        "test_accuracy": round(float(acc), 4),
        "per_class": {CLASSES[i]: {"precision": round(rep[CLASSES[i]]["precision"], 4),
                                   "recall": round(rep[CLASSES[i]]["recall"], 4),
                                   "f1": round(rep[CLASSES[i]]["f1-score"], 4),
                                   "support": int(rep[CLASSES[i]]["support"])}
                      for i in range(7)},
        "confusion_matrix": cm,
    }
    torch.save(net.state_dict(), os.path.join(ROOT, "model.pt"))
    json.dump(metrics, open(os.path.join(ROOT, "metrics.json"), "w"), indent=2)
    print("TEST ACCURACY:", round(float(acc), 4))
    for i in range(7):
        m = metrics["per_class"][CLASSES[i]]
        print(f"{CLASSES[i]:22s} p={m['precision']:.3f} r={m['recall']:.3f} f1={m['f1']:.3f} n={m['support']}")
    return metrics


if __name__ == "__main__":
    raw = extract() if not os.path.isdir(os.path.join(DATA, "train")) else os.path.join(DATA, "raw")
    if not os.path.isdir(os.path.join(DATA, "train")):
        split(raw)
    base = baselines()
    m = train()
    m["baselines"] = base
    json.dump(m, open(os.path.join(ROOT, "metrics.json"), "w"), indent=2)
    print("model.pt bytes:", os.path.getsize(os.path.join(ROOT, "model.pt")))
