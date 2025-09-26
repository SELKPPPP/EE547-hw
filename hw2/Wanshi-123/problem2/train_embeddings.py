import argparse, json, os, re, time,datetime 
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone

# 1. Text cleaning
def clean_text(text):
    # Convert to lowercase
    # Remove non-alphabetic characters except spaces
    # Split into words
    # Remove very short words (< 2 characters)
    text = text.lower()
    text = re.sub(r'[^a-z ]+', ' ', text)   # 
    words = text.split()
    words = [w for w in words if len(w) >= 2]  
    return words

# 2. Vocabulary building
def build_vocab(docs, max_vocab=5000):
    # Extract all unique words from abstracts
    # Keep only the top 5,000 most frequent words (parameter budget constraint)
    # Create word-to-index mapping
    # Reserve index 0 for unknown words

    counter = Counter()
    total_words = 0
    # Count word frequencies
    for words in docs:
        counter.update(words)
        total_words += len(words)
    most_common = counter.most_common(max_vocab - 1)  # -1 because we reserve index 0 for <UNK>
    
    word2idx = {"<UNK>": 0}
    idx2word = ["<UNK>"]
    for i, (w, _) in enumerate(most_common, start=1):
        word2idx[w] = i
        idx2word.append(w)
    return word2idx, idx2word, total_words

# 3. Sequence encoding
def encode_sequence(words, word2idx, fixed_length=150):
    # Convert abstracts to sequences of word indices
    # Pad or truncate to fixed length (e.g., 100-200 words)
    # Create bag-of-words representation for autoencoder input/output
    unk = 0
    ids = [word2idx.get(w, unk) for w in words[:fixed_length]]
    if len(ids) < fixed_length:   # padding
        ids.extend([unk] * (fixed_length - len(ids)))
    return ids

# Convert list of sequences to bag-of-words matrix
def seqs_to_bow(seqs, vocab_size):
    N, V = len(seqs), vocab_size
    mat =torch.zeros((N, V), dtype=torch.float32)
    for i, ids in enumerate(seqs):
        seen = set()
        for idx in ids:
            if 0 <= idx < V and idx not in seen:
                mat[i, idx] = 1.0
                seen.add(idx)
    return mat



class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        # Encoder: vocab_size → hidden_dim → embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Decoder: embedding_dim → hidden_dim → vocab_size  
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()  # Output probabilities
        )
    
    def forward(self, x):
        # Encode to bottleneck
        embedding = self.encoder(x)
        # Decode back to vocabulary space
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    


# Dataset class
class BoWDataset(Dataset):
    def __init__(self, bows): self.bows = bows
    def __len__(self): return self.bows.size(0)
    def __getitem__(self, i): x = self.bows[i]; return x, x  

# Count trainable parameters in model
def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# Count trainable parameters in encoder
def count_encoder_params(model):
    return sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

# Architecture string for printing
def arch_string(vocab_size, hidden_dim, embedding_dim):
    return f"{vocab_size} -> {hidden_dim} -> {embedding_dim} -> {hidden_dim} -> {vocab_size}"


def UTC_time():
    # always UTC with trailing Z
    return datetime.now(timezone.utc).isoformat()

# ---------- Training function ----------
def train_autoencoder(
    bows, vocab_size, out_dir,
    epochs=50, batch_size=32, hidden_dim=256, embedding_dim=128, lr=1e-3):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextAutoencoder(vocab_size, hidden_dim, embedding_dim).to(device)

    total_params = count_params(model)

    enc_params = count_encoder_params(model)
    print(f"Model: {arch_string(vocab_size, hidden_dim, embedding_dim)}")
    print(f"Encoder params: {enc_params:,} (limit 2,000,000)")
    if enc_params > 2_000_000:
        raise RuntimeError("Encoder params exceed 2,000,000")

    ds = BoWDataset(bows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    print("\nTraining autoencoder...")
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            recon, _ = model(xb)
            loss = crit(recon, yb)
            loss.backward()
            opt.step()
            running += loss.item()
        last_epoch_loss = running / len(dl)    
        print(f"Epoch {ep}/{epochs}, Loss: {running/len(dl):.4f}")
    print(f"Training complete in {time.time()-t0:.1f}s")

      # Save model
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": None,  # will be filled outside
        "model_config": {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim
        }
    }, os.path.join(out_dir, "model.pth"))
    return model  ,total_params, last_epoch_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=str)
    ap.add_argument("output_dir", type=str)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()


    print(f"Loading abstracts from {os.path.basename(args.input_json)}...")

    with open(args.input_json, "r") as f:
        data = json.load(f) 
    
    abstracts, arxiv_ids = [], []

    for i, obj in enumerate(data):
        if isinstance(obj, dict):
            k_abs = next((k for k in obj.keys() if k.lower()=="abstract"), None)
            k_id  = next((k for k in obj.keys() if k.lower() in ("arxiv_id","id")), None)
            abstracts.append(obj.get(k_abs, "") if k_abs else "")
            arxiv_ids.append(str(obj.get(k_id, i)) if k_id else str(i))
        else:
            abstracts.append(str(obj))
            arxiv_ids.append(str(i))
    print(f"Found {len(abstracts)} abstracts")

    docs = [clean_text(t) for t in abstracts]

    print(f"Building vocabulary from {len(docs)} words...")
    
    word2idx, idx2word, total_words = build_vocab(docs, max_vocab=5000)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size} words")

  
    seqs = [encode_sequence(d, word2idx, 150) for d in docs]
    bows = seqs_to_bow(seqs, vocab_size)

    # Train autoencoder
    start_time = UTC_time()
    model, total_params, last_epoch_loss = train_autoencoder(
        bows=bows, vocab_size=vocab_size, out_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden_dim=256, embedding_dim=128, lr=1e-3
    )
    end_time = UTC_time()

    # model.pth add vocab_to_idx
    model_pth_path = os.path.join(args.output_dir, "model.pth")
    ckpt = torch.load(model_pth_path, map_location="cpu")
    ckpt["vocab_to_idx"] = word2idx
    torch.save(ckpt, model_pth_path)

    # — vocabulary.json（vocab_to_idx、idx_to_vocab、vocab_size、total_words） —
    voc_path = os.path.join(args.output_dir, "vocabulary.json")
    idx_to_vocab = {str(i): w for w, i in word2idx.items()}
    with open(voc_path, "w") as f:
        json.dump({
            "vocab_to_idx": word2idx,
            "idx_to_vocab": idx_to_vocab,
            "vocab_size": vocab_size,
            "total_words": total_words
        }, f, indent=2)

    # —— embeddings.json（arxiv_id、embedding、reconstruction_loss） ——
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    crit_none = nn.BCELoss(reduction="none") # per-element loss
    embs_out = []
    with torch.no_grad():
        bs = max(256, args.batch_size)
        for i in range(0, bows.size(0), bs):
            xb = bows[i:i+bs].to(device)
            recon, z = model(xb)  # recon:[B,V], z:[B,E]
            # per-sample reconstruction loss
            loss_elem = crit_none(recon, xb)
            loss_per = loss_elem.mean(dim=1).cpu().tolist()
            z_list = z.cpu().tolist()
            for j, emb in enumerate(z_list):
                embs_out.append({
                    "arxiv_id": arxiv_ids[i+j],
                    "embedding": [float(v) for v in emb],
                    "reconstruction_loss": float(loss_per[j])
                })
    with open(os.path.join(args.output_dir, "embeddings.json"), "w") as f:
        json.dump(embs_out, f, indent=2)

    # —— training_log.json（start_time、end_time、epochs、final_loss、total_parameters、papers_processed、embedding_dimension） ——
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump({
            "start_time": start_time,
            "end_time": end_time,
            "epochs": args.epochs,
            "final_loss": float(last_epoch_loss) ,
            "total_parameters": int(total_params),
            "papers_processed": len(abstracts),
            "embedding_dimension":128
        }, f, indent=2)

if __name__ == "__main__":
    main()