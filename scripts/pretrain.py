"""D-MPNN 인코더 자기지도 사전학습. 스펙 05 §2 P3~P6, §3.

python scripts/pretrain.py --method mask
python scripts/pretrain.py --method contrast

- 데이터: data/processed/zinc250k_pretrain.csv (scripts/curate_pretrain.py). 5%는 valid 손실 기록용
- HP 고정 (downstream으로 튜닝하지 않는다): Adam 1e-3, 배치 256, 100에폭, d_h 300, depth 3, dropout 0
- 장치: MPS (결정 T2, 2026-10-27). MPS는 결정론이 아니어서 다시 돌리면 가중치가 조금 달라진다.
  그래서 최종 가중치를 SHA-256과 함께 커밋하고, 미세조정 그리드는 CPU 결정론으로 이 파일을 쓴다
- 마지막 에폭 가중치를 쓴다 (valid 손실이나 downstream 성능으로 고르지 않음)
- 에폭마다 체크포인트(models/pretrained/ckpt/)를 남겨, 중단 후 같은 명령으로 이어서 돈다
→ models/pretrained/D_<method>.pt (인코더 state_dict) + D_<method>.json (손실 기록, 해시, 설정)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labelbudget.featurize.graph import batch_graphs, mol_graph
from labelbudget.models.dmpnn import DMPNNEncoder, readout
from labelbudget.models.gnn_space import FIXED_BACKBONE
from labelbudget.pretrain.tasks import (AUG_RATIO, MASK_RATE, TEMPERATURE, MaskHead, ProjectionHead, atom_classes,
                                        drop_atoms, graphcl_loss, mask_atoms)

ROOT = Path(__file__).resolve().parents[1]


def mask_step(enc, head, graphs, rng, device):
    masked, pos, targets, off = [], [], [], 0
    for g in graphs:
        mg, idx = mask_atoms(g, rng)
        masked.append(mg)
        pos.append(off + idx)  # 배치 안 원자 번호
        targets.append(atom_classes(g)[idx])
        off += len(g.V)
    bg = batch_graphs(masked).to(device)
    logits = head(enc(bg)[torch.from_numpy(np.concatenate(pos)).to(device)])
    target = torch.from_numpy(np.concatenate(targets)).to(device)
    loss = torch.nn.functional.cross_entropy(logits, target)
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc


def contrast_step(enc, proj, graphs, rng, device):
    v1 = [mask_atoms(g, rng, rate=AUG_RATIO, count="graphcl", fill="mean")[0] for g in graphs]
    v2 = [drop_atoms(g, rng) for g in graphs]
    zs = []
    for view in (v1, v2):
        bg = batch_graphs(view).to(device)
        zs.append(proj(readout(enc(bg), bg.batch, bg.n_mols, "mean")))
    loss = graphcl_loss(*zs)
    with torch.no_grad():
        z1, z2 = (torch.nn.functional.normalize(z, dim=1) for z in zs)
        acc = ((z1 @ z2.T).argmax(1) == torch.arange(len(graphs), device=device)).float().mean()
    return loss, acc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["mask", "contrast"], required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--valid-frac", type=float, default=0.05)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "zinc250k_pretrain.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "models" / "pretrained")
    p.add_argument("--limit", type=int, help="앞 N개 분자만 (스모크 테스트용)")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    smiles = pd.read_csv(args.data).smiles.tolist()[:args.limit]
    t0 = time.time()
    graphs = [mol_graph(s) for s in smiles]
    order = np.random.default_rng(args.seed).permutation(len(graphs))
    n_va = max(2, int(len(graphs) * args.valid_frac))  # 대조 손실은 배치에 2개 이상 필요
    va, tr = [graphs[i] for i in order[:n_va]], [graphs[i] for i in order[n_va:]]
    print(f"그래프 {len(graphs):,}개 ({time.time() - t0:.0f}s), train {len(tr):,} / valid {len(va):,}", flush=True)

    enc = DMPNNEncoder(d_h=FIXED_BACKBONE["d_h"], depth=FIXED_BACKBONE["depth"], dropout=0.0).to(device)
    head = (MaskHead if args.method == "mask" else ProjectionHead)(FIXED_BACKBONE["d_h"]).to(device)
    step = mask_step if args.method == "mask" else contrast_step
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"D_{args.method}_last.pt"
    rng = np.random.default_rng(args.seed + 1)
    history, start = [], 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        enc.load_state_dict(st["enc"]), head.load_state_dict(st["head"]), opt.load_state_dict(st["opt"])
        rng.bit_generator.state, history, start = st["rng"], st["history"], st["epoch"] + 1
        torch.set_rng_state(st["torch_rng"])
        print(f"체크포인트에서 이어서: 에폭 {start}", flush=True)

    bs = args.batch_size
    for epoch in range(start, args.epochs):
        t0 = time.time()
        enc.train(), head.train()
        perm = rng.permutation(len(tr))
        tot = n = 0.0
        for i in range(0, len(perm), bs):
            batch = [tr[j] for j in perm[i:i + bs]]
            if len(batch) < 2:
                continue
            loss, _ = step(enc, head, batch, rng, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(batch)
            n += len(batch)
        enc.eval(), head.eval()
        vrng = np.random.default_rng(args.seed + 2)  # valid 증강은 에폭마다 같게 → 손실끼리 비교 가능
        vl = va_acc = vn = 0.0
        with torch.no_grad():
            for i in range(0, len(va), bs):
                batch = va[i:i + bs]
                if len(batch) < 2:
                    continue
                loss, acc = step(enc, head, batch, vrng, device)
                vl += loss.item() * len(batch)
                va_acc += acc.item() * len(batch)
                vn += len(batch)
        rec = {"epoch": epoch, "train_loss": tot / n, "valid_loss": vl / vn, "valid_acc": va_acc / vn,
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "history": history,
                    "epoch": epoch}, ckpt.with_suffix(".tmp"))
        ckpt.with_suffix(".tmp").replace(ckpt)

    out = args.out_dir / f"D_{args.method}.pt"
    torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, out)
    meta = {"method": args.method, "weights_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "data": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
            "data_sha256": data_sha, "n_train": len(tr), "n_valid": len(va), "device": args.device,
            "torch": torch.__version__, "backbone": FIXED_BACKBONE, "epochs": args.epochs, "batch_size": bs,
            "lr": args.lr, "seed": args.seed, "mask_rate": MASK_RATE, "aug_ratio": AUG_RATIO,
            "temperature": TEMPERATURE, "limit": args.limit, "history": history}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"→ {out} sha256 {meta['weights_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
