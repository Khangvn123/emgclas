"""
Layer 07 — 17 Features + Population Coding + SNN LIF (ATan)  (3-class)
======================================================================
Pipeline:
  1. Load features_17.csv (đã trích xuất sẵn 17 đặc trưng TD/FD mỗi segment)
  2. Subject-level holdout split 80 / 10 / 10 (train / val / test)
  3. Chuẩn hóa p1/p99 từ train only -> [0, 1]
  4. Population coding:
        Mỗi feature được mô phỏng bằng N_POP neuron Gaussian tuning.
        Response r_i(x) = exp(-((x - mu_i) / sigma)^2)
        -> (n_feat * N_POP) response vector per segment
  5. Mã hóa spike theo TTFS (Time-to-First-Spike) -> (T, B, n_in)
        Neuron response cao -> spike sớm (t_fire = (1-r)*(T-1))
        Neuron response < TTFS_THR -> không spike
  6. SNN: TemporalLIF1 (238->256) + TemporalLIF2 (256->3, readout qua sum/T)
        Surrogate gradient: ATan  (snntorch-style)
  7. Huấn luyện giám sát với FocalLoss, dừng sớm theo macro-F1 val

Input features (17): MAV, RMS, VAR, WL, SD, ZC, SSC, WAMP, IEMG, SSI,
                     MV, LOG, MFL, DAMV, MNF, MDF, PKF
"""

import os
import re
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH      = 'features_17.csv'

FEATURE_COLS  = ['MAV', 'RMS', 'VAR', 'WL', 'SD', 'ZC', 'SSC', 'WAMP',
                 'IEMG', 'SSI', 'MV', 'LOG', 'MFL', 'DAMV',
                 'MNF', 'MDF', 'PKF']
N_FEAT        = len(FEATURE_COLS)           # 17

LABEL_MAP     = {'Healthy': 0, 'Myopathy': 1, 'Neuropathy': 2}
CLASS_NAMES   = ['Healthy', 'Myopathy', 'Neuropathy']
N_CLASSES     = 3

# Population coding
N_POP         = 14                        # số neuron / feature
SIGMA_POP     = 1.0 / (N_POP - 1) * 1.5   # độ rộng Gaussian (trong miền [0,1])
N_INPUT       = N_FEAT * N_POP               # 17*14 = 238

# TTFS spike encoding
T_STEPS       = 40                      # T=40: ~4 timestep/neuron -> đủ phân giải
TTFS_THR      = 0.05                    # hạ ngưỡng: giữ nhiều spike hơn

# Model
FC_DIM        = 64
HIDDEN        = 256
BETA_INIT     = 0.9   # decay nhanh hon: phan biet timing TTFS ro hon
THRESHOLD     = 0.4    # LIF de fire hon: tranh dead neuron voi input thoa
T_OUT         = 30

# Train
EPOCHS        = 300
LR            = 5e-4
BATCH         = 32
PATIENCE      = 80

SPLIT_SEED    = 42    # co dinh: quyet dinh data split, KHONG doi khi thi nghiem
TRAIN_SEED    = 42    # co the doi: chi anh huong weight init, khong anh huong split

TRAIN_RATIO   = 0.70
VALID_RATIO   = 0.10
TEST_RATIO    = 0.20

torch.manual_seed(TRAIN_SEED)
np.random.seed(SPLIT_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def _get_subject_id(filename: str) -> str:
    """Lấy subject id từ tên file: vd 'EMG _201 _01_ RB_Hea.asc' -> '201_01'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'(\d+)\s*_\s*(\d+)', base)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return base


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df['class'].isin(LABEL_MAP.keys())].reset_index(drop=True)

    X     = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y     = df['class'].map(LABEL_MAP).to_numpy(dtype=np.int64)
    subj  = df['filename'].apply(_get_subject_id).to_numpy()
    # Khóa nhận biết 1 subject (mỗi subject 1 lớp) - giữ nguyên class prefix
    # tránh trùng ID giữa các lớp
    keys  = np.array([f"{c}_{s}" for c, s in zip(df['class'], subj)])
    return X, y, keys


# ─────────────────────────────────────────────────────────────────────────────
# CHUẨN HÓA (p1/p99 robust, chỉ từ train)
# ─────────────────────────────────────────────────────────────────────────────
def fit_normalizer(X: np.ndarray):
    p1    = np.percentile(X,  1, axis=0).astype(np.float32)
    p99   = np.percentile(X, 99, axis=0).astype(np.float32)
    denom = np.where(p99 - p1 > 0, p99 - p1, 1.0).astype(np.float32)
    return p1, denom


def apply_normalizer(X: np.ndarray, p1, denom) -> np.ndarray:
    return np.clip((X - p1) / denom, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# POPULATION CODING (Gaussian receptive fields)
# ─────────────────────────────────────────────────────────────────────────────
# Mỗi feature normalized [0,1] -> N_POP neuron với mu_i trải đều [0,1]
#   r_i(x) = exp(-((x - mu_i)^2) / (2 * sigma^2))
# Kết quả: (n_sample, N_FEAT * N_POP) cường độ kích thích trong [0, 1]
MU_POP = np.linspace(0.0, 1.0, N_POP, dtype=np.float32)   # (N_POP,)


def population_encode(X_norm: np.ndarray) -> np.ndarray:
    """
    X_norm: (N, N_FEAT) da normalize ve [0, 1]
    return: (N, N_FEAT * N_POP) activation trong [0, 1]
    """
    x_exp  = X_norm[:, :, None]                         # (N, F, 1)
    mu     = MU_POP[None, None, :]                      # (1, 1, P)
    resp   = np.exp(-((x_exp - mu) ** 2) / (2 * SIGMA_POP ** 2))
    return resp.reshape(X_norm.shape[0], -1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SPIKE ENCODING — Time-to-First-Spike (TTFS)
#   Neuron population response r in [0,1]:
#     t_fire = round((1 - r) * (T - 1))   ->  r cao => spike sớm
#     r < TTFS_THR                          ->  không spike (xa trung tâm)
# ─────────────────────────────────────────────────────────────────────────────
def ttfs_spikes(rate: torch.Tensor, T: int,
                thr: float = TTFS_THR) -> torch.Tensor:
    """
    rate : (B, n_in) trong [0, 1]  — population Gaussian response
    return: (T, B, n_in) spike {0,1}, moi neuron toi da 1 spike
    """
    # t_fire in [0, T-1]: response = 1 -> t=0, response = 0 -> t=T-1
    t_fire = ((1.0 - rate) * (T - 1)).long().clamp(0, T - 1)   # (B, n_in)
    spikes = torch.zeros(T, *rate.shape, device=rate.device)
    # scatter mỗi neuron vào đúng timestep của nó
    spikes.scatter_(0, t_fire.unsqueeze(0), 1.0)
    # xóa spike của những neuron có response quá thấp (xa trung tâm Gaussian)
    spikes = spikes * (rate > thr).float().unsqueeze(0)
    return spikes


# ─────────────────────────────────────────────────────────────────────────────
# ATan SURROGATE GRADIENT  (snntorch-style)
#   forward : Heaviside
#   backward: 1 / (1 + (pi * (v - thr))^2)
# ─────────────────────────────────────────────────────────────────────────────
class AtanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v - thr)
        return (v >= thr).float()

    @staticmethod
    def backward(ctx, grad_out):
        (dv,) = ctx.saved_tensors
        grad = grad_out / (1.0 + (math.pi * dv) ** 2)
        return grad, None


def spike(v, thr=THRESHOLD):
    return AtanSpike.apply(v, thr)


# ─────────────────────────────────────────────────────────────────────────────
# LIF LAYERS
# ─────────────────────────────────────────────────────────────────────────────
class TemporalLIF(nn.Module):
    """
    (T, B, n_in) spike train -> (T, B, n_out) spike train.
    LIF nhan spike thau, tu tich luy mem, tra ve spike train day du.
    """
    def __init__(self, n_in: int, n_out: int,
                 beta: float = BETA_INIT, thr: float = THRESHOLD):
        super().__init__()
        self.fc  = nn.Linear(n_in, n_out, bias=False)  # bias=False: cur=0 khi input=0
        self.thr = thr
        raw = float(np.log(beta / (1.0 - beta + 1e-8)))
        self.raw_beta = nn.Parameter(torch.full((n_out,), raw))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T, B, _ = x_seq.shape
        beta = torch.sigmoid(self.raw_beta)
        mem  = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        spk  = torch.zeros_like(mem)
        spk_seq = []
        for t in range(T):
            cur = self.fc(x_seq[t])
            mem = beta * mem + cur - self.thr * spk
            spk = spike(mem, self.thr)
            spk_seq.append(spk)
        return torch.stack(spk_seq, dim=0)  # (T, B, n_out)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class PopSNN(nn.Module):
    """
    Input : (B, N_INPUT) population response [0, 1]
    Pipeline (SNN thuan):
      TTFS encode   : (B, N_INPUT) -> (T, B, N_INPUT) spike train
      TemporalLIF 1 : (T, B, N_INPUT) -> (T, B, HIDDEN) spike train
      TemporalLIF 2 : (T, B, HIDDEN)  -> (T, B, N_CLASSES) spike train
      Readout       : sum over T / T  -> (B, N_CLASSES) rate -> log_softmax
    """
    def __init__(self, n_in: int = N_INPUT, n_hidden: int = HIDDEN,
                 n_out: int = N_CLASSES, T: int = T_STEPS):
        super().__init__()
        self.T    = T
        self.lif1 = TemporalLIF(n_in,    n_hidden)
        self.lif2 = TemporalLIF(n_hidden, n_out)

    def forward(self, rate: torch.Tensor) -> torch.Tensor:
        spikes = ttfs_spikes(rate, self.T)      # (T, B, N_INPUT)
        spk1   = self.lif1(spikes)              # (T, B, HIDDEN)  spike train
        spk2   = self.lif2(spk1)               # (T, B, N_CLASSES) spike train
        out    = spk2.sum(dim=0) / self.T       # (B, N_CLASSES) firing rate
        return torch.log_softmax(out, dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # (N_CLASSES,) — class weight de can bang

    def forward(self, log_prob, target):
        prob = log_prob.exp()
        p_t  = prob.gather(1, target.view(-1, 1)).squeeze(1)
        fl   = -((1 - p_t) ** self.gamma) * p_t.log()
        if self.weight is not None:
            fl = fl * self.weight[target]
        return fl.mean()


criterion = None  # khoi tao trong main sau khi biet phan phoi class


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, optimizer, X, y, training=True):
    model.train(training)
    idx = np.random.permutation(len(X)) if training else np.arange(len(X))
    losses, lp_all = [], []
    for i in range(0, len(X), BATCH):
        batch = idx[i: i + BATCH]
        xb = torch.tensor(X[batch], device=device)
        yb = torch.tensor(y[batch], device=device)
        lp = model(xb)
        loss = criterion(lp, yb)
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(loss.item())
        lp_all.append(lp.detach().cpu())
    return float(np.mean(losses)), torch.cat(lp_all, dim=0)


N_AVG = 1    # TTFS là deterministic -> 1 lần forward là đủ


def get_preds(model, X):
    """TTFS deterministic: 1 forward pass cho kết quả ổn định."""
    model.eval()
    sum_prob = None
    with torch.no_grad():
        for _ in range(N_AVG):
            lp_list = []
            for i in range(0, len(X), BATCH):
                xb = torch.tensor(X[i: i + BATCH], device=device)
                lp_list.append(model(xb).exp().cpu())
            prob = torch.cat(lp_list, dim=0)   # (N, 3)
            sum_prob = prob if sum_prob is None else sum_prob + prob
    return (sum_prob / N_AVG).argmax(dim=1).numpy()


def train_model(X_tr, y_tr, X_vl, y_vl, seed=TRAIN_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PopSNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1, best_state, wait = 0.0, None, 0

    for ep in range(1, EPOCHS + 1):
        tr_loss, _       = run_epoch(model, opt, X_tr, y_tr, training=True)
        vl_loss, vl_lp   = run_epoch(model, opt, X_vl, y_vl, training=False)
        sched.step()

        vl_pred = vl_lp.argmax(dim=1).numpy()
        vl_f1   = f1_score(y_vl, vl_pred, average='macro', zero_division=0)

        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d} | tr_loss={tr_loss:.4f} "
                  f"vl_loss={vl_loss:.4f} vl_F1={vl_f1*100:.1f}%")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop ep {ep} | Best Val F1={best_f1*100:.1f}%")
                break

    model.load_state_dict(best_state)
    return model, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  LAYER 07 — 17 Features + Population Coding + SNN LIF (ATan)")
    print("  Holdout Split (subject-level): "
          f"Train {TRAIN_RATIO*100:.0f}% / Val {VALID_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")
    print("  Healthy | Myopathy | Neuropathy")
    print("=" * 70)
    print(f"  Device   : {device}")
    print(f"  Features : {N_FEAT} (MAV..PKF)  | N_POP={N_POP} | N_INPUT={N_INPUT}")
    print(f"  Encoding : Gaussian receptive field + TTFS T={T_STEPS} thr={TTFS_THR}")
    print(f"  Model    : TemporalLIF({N_INPUT}->{HIDDEN}) + TemporalLIF({HIDDEN}->{N_CLASSES})"
          f" [SNN thuan, spike train xuyen suot]")
    print(f"  Surrogate: ATan   | Beta0={BETA_INIT}  | Thr={THRESHOLD}")

    _tmp = PopSNN()
    print(f"  Params   : {count_params(_tmp):,}")
    del _tmp

    # ── Load ───────────────────────────────────────────────────────────────
    print(f"\n  Đang đọc {CSV_PATH} ...")
    X_all, y_all, keys_all = load_dataset(CSV_PATH)
    print(f"  Tổng số segment: {len(X_all)}")

    # ── Subject-level holdout split ────────────────────────────────────────
    uniq_keys   = np.unique(keys_all)
    uniq_labels = np.array([y_all[keys_all == k][0] for k in uniq_keys])

    print(f"\n  Phân bố subject:")
    for cls_id, cname in enumerate(CLASS_NAMES):
        n_c = int((uniq_labels == cls_id).sum())
        print(f"    {cname:12s}: {n_c} subject")

    k_tr, k_tmp, _, lb_tmp = train_test_split(
        uniq_keys, uniq_labels,
        test_size=(VALID_RATIO + TEST_RATIO),
        stratify=uniq_labels,
        random_state=SPLIT_SEED)

    vt_ratio = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    k_vl, k_te = train_test_split(
        k_tmp, test_size=(1.0 - vt_ratio),
        stratify=lb_tmp,
        random_state=SPLIT_SEED)

    def mask_of(k_list):
        return np.isin(keys_all, k_list)

    tr_m, vl_m, te_m = mask_of(k_tr), mask_of(k_vl), mask_of(k_te)

    X_tr_raw, y_tr = X_all[tr_m], y_all[tr_m]
    X_vl_raw, y_vl = X_all[vl_m], y_all[vl_m]
    X_te_raw, y_te = X_all[te_m], y_all[te_m]

    # ── Normalize từ train only ────────────────────────────────────────────
    print("\n  Chuẩn hóa p1/p99 từ train only...")
    p1, denom = fit_normalizer(X_tr_raw)
    X_tr_n = apply_normalizer(X_tr_raw, p1, denom)
    X_vl_n = apply_normalizer(X_vl_raw, p1, denom)
    X_te_n = apply_normalizer(X_te_raw, p1, denom)

    # ── Xuất CSV đặc trưng đã chuẩn hóa ──────────────────────────────────
    def export_normalized_csv(X_n, y, split_name, path):
        df_out = pd.DataFrame(X_n, columns=FEATURE_COLS)
        df_out.insert(0, 'split', split_name)
        df_out.insert(1, 'class', [CLASS_NAMES[c] for c in y])
        df_out.to_csv(path, index=False)

    export_normalized_csv(X_tr_n, y_tr, 'train', 'features_normalized_train.csv')
    export_normalized_csv(X_vl_n, y_vl, 'val',   'features_normalized_val.csv')
    export_normalized_csv(X_te_n, y_te, 'test',  'features_normalized_test.csv')

    # Gộp cả 3 vào 1 file
    df_all_norm = pd.concat([
        pd.read_csv('features_normalized_train.csv'),
        pd.read_csv('features_normalized_val.csv'),
        pd.read_csv('features_normalized_test.csv'),
    ], ignore_index=True)
    df_all_norm.to_csv('features_normalized_all.csv', index=False)
    print(f"  Đã xuất CSV chuẩn hóa:")
    print(f"    features_normalized_train.csv ({len(X_tr_n)} segment)")
    print(f"    features_normalized_val.csv   ({len(X_vl_n)} segment)")
    print(f"    features_normalized_test.csv  ({len(X_te_n)} segment)")
    print(f"    features_normalized_all.csv   ({len(df_all_norm)} segment, gộp)")

    # ── Population coding ──────────────────────────────────────────────────
    print(f"  Population coding (Gaussian RF, N_POP={N_POP})...")
    X_tr = population_encode(X_tr_n)
    X_vl = population_encode(X_vl_n)
    X_te = population_encode(X_te_n)

    for name, X_s, y_s in [('Train', X_tr, y_tr),
                             ('Valid', X_vl, y_vl),
                             ('Test ', X_te, y_te)]:
        cnts = " | ".join(f"{CLASS_NAMES[c]}={(y_s==c).sum()}"
                          for c in range(N_CLASSES))
        print(f"  {name}: {len(X_s):4d} segment  ({cnts})")

    # ── Class-weighted FocalLoss ───────────────────────────────────────────
    cnts_tr = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    cls_w   = torch.tensor(cnts_tr.sum() / (N_CLASSES * cnts_tr), device=device)
    print(f"  Class weights: " +
          " | ".join(f"{CLASS_NAMES[i]}={cls_w[i]:.2f}" for i in range(N_CLASSES)))
    criterion = FocalLoss(gamma=2.0, weight=cls_w)

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n  Đang train PopSNN (LR={LR}, BATCH={BATCH}, TTFS T={T_STEPS})...")
    model, best_val_f1 = train_model(X_tr, y_tr, X_vl, y_vl)

    # ── Test ───────────────────────────────────────────────────────────────
    pred_te = get_preds(model, X_te)
    f1_te   = f1_score(y_te, pred_te, average='macro', zero_division=0)
    cm      = confusion_matrix(y_te, pred_te, labels=list(range(N_CLASSES)))
    acc     = (pred_te == y_te).mean()

    print(f"\n{'='*70}")
    print(f"  KẾT QUẢ LAYER 07 — TEST SET ({len(X_te)} segment)")
    print(f"{'='*70}")

    col_w = 12
    header = f"{'':18s}" + "".join(f"Pred {c:<{col_w-5}}" for c in CLASS_NAMES)
    print(f"\n  Confusion Matrix (hàng=True, cột=Pred):")
    print(f"  {header}")
    sep = "  " + "─" * (18 + col_w * N_CLASSES)
    print(sep)
    for i, cname in enumerate(CLASS_NAMES):
        row = f"  True {cname:<13s}"
        for j in range(N_CLASSES):
            row += f"{cm[i,j]:<{col_w}d}"
        print(row)
    print(sep)

    print()
    print(classification_report(y_te, pred_te,
                                target_names=CLASS_NAMES, digits=3))
    print(f"  Accuracy      = {acc*100:.1f}%")
    print(f"  F1 Macro Test = {f1_te*100:.1f}%")
    print(f"  Best Val F1   = {best_val_f1*100:.1f}%")
    print(f"\n{'='*70}")
    print(f"    17 Features + PopCoding + SNN(ATan) : F1 Macro = {f1_te*100:.1f}%")
    print(f"{'='*70}")

    # ── Lưu ────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), 'layer07_pop_snn_holdout.pth')
    np.savez('layer07_normalizer_holdout.npz',
             p1=p1, denom=denom, mu_pop=MU_POP, sigma_pop=SIGMA_POP)
    print(f"\n  Model     : layer07_pop_snn_holdout.pth")
    print(f"  Normalizer: layer07_normalizer_holdout.npz")
