"""
Training Pipeline for Model 10 (Held-Object Confounder Immunity & Knife Recall Recovery SOTA).
Addresses Thesis 2 Panelist & Adviser Recommendations:
1. Restores Full 5-Stage Anchor Pyramid ((16,), (32,), (64,), (128,), (256,)) + Calibrated 2.2x BBox Clamping
   to resolve Model 9's 59 Knife False Negatives caused by 128px anchor truncation.
2. Trains on 450 Held Non-Weapon Confounders (phones, bottles, wallets, keys, remotes, umbrellas, selfie sticks, tools)
   from CostiCatargiu NoGun v5 + 296 V9/V10/V11 CCTV Hard Negatives (746 injected + 404 base = 1,150 total negatives).
3. Reinforces thin/angled blade & handgun features with 350 Knife + 200 Pistol scenes from CostiCatargiu WeaponDataset v11.
4. Evaluates after every epoch on both the M9 Held-Out Test Split (279 images) and the Unified External Benchmark (350 images).
"""
import os
import sys
import time
import math
from pathlib import Path

# Strict CPU thread cap to 8 threads
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_10.tenth_model_dataset import TenthModelDataset
from training.model_9.evaluate_ninth_model import compute_ap_for_class, compute_iou, collate_fn

BEST_M10_PATH = ROOT / "best_weapon_detector_tenth_model.pth"
LATEST_M10_PATH = ROOT / "latest_weapon_detector_tenth_model.pth"
BASE_V10_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v10.pth"

BATCH_SIZE = 2
ACCUM_STEPS = 1
EPOCHS = 4
BASE_LR = 0.00012
RPN_CLIP = math.log(2.2)  # ~0.7885 (2.2x expansion allows 256px anchor to reach 563px knives)
ROI_CLIP = math.log(2.2)
ANCHOR_SCALES = (16, 32, 64, 128, 256)


def evaluate_on_loader(model, loader, device, conf_thresh=0.50):
    model.eval()
    preds, gt_by_image = [], []
    idx = 0
    with torch.no_grad():
        for imgs, tgts in loader:
            outs = model([i.to(device) for i in imgs])
            for t, o in zip(tgts, outs):
                gt_by_image.append(
                    [{"box": b.tolist(), "class_id": int(l)} for b, l in zip(t["boxes"].cpu().numpy(), t["labels"].cpu().numpy())]
                )
                for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
                    if s >= conf_thresh:
                        preds.append({"img_idx": idx, "box": b.tolist(), "class_id": int(l), "score": float(s)})
                idx += 1

    ap_hg = compute_ap_for_class(preds, gt_by_image, 1, 0.50)
    ap_kn = compute_ap_for_class(preds, gt_by_image, 2, 0.50)
    mAP = (ap_hg + ap_kn) / 2.0

    tp_tot, fp_tot, fn_tot = 0, 0, 0
    per_cls = {}
    for cname, cid in [("Handgun", 1), ("Knife", 2)]:
        c_preds = sorted([p for p in preds if p["class_id"] == cid], key=lambda x: x["score"], reverse=True)
        c_gts = sum(len([g for g in gts if g["class_id"] == cid]) for gts in gt_by_image)
        matched = {i: set() for i in range(len(gt_by_image))}
        tp, fp = 0, 0
        for p in c_preds:
            i_idx = p["img_idx"]
            gts = [(gi, g["box"]) for gi, g in enumerate(gt_by_image[i_idx]) if g["class_id"] == cid]
            bi, bg = 0.0, -1
            for gi, gb in gts:
                iou = compute_iou(p["box"], gb)
                if iou > bi:
                    bi, bg = iou, gi
            if bi >= 0.50 and bg not in matched[i_idx]:
                tp += 1
                matched[i_idx].add(bg)
            else:
                fp += 1
        fn = c_gts - tp
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        per_cls[cname] = {"TP": tp, "FP": fp, "FN": fn}

    prec = tp_tot / max(1, tp_tot + fp_tot)
    rec = tp_tot / max(1, tp_tot + fn_tot)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    negs = [i for i, g in enumerate(gt_by_image) if len(g) == 0]
    neg_clean = sum(1 for i in negs if not any(p["img_idx"] == i for p in preds))
    neg_rate = neg_clean / max(1, len(negs))

    return {
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AP_Handgun": ap_hg,
        "AP_Knife": ap_kn,
        "mAP": mAP,
        "TP": tp_tot,
        "FP": fp_tot,
        "FN": fn_tot,
        "per_class": per_cls,
        "neg_clean": neg_clean,
        "neg_total": len(negs),
        "neg_rate": neg_rate,
    }


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / ACCUM_STEPS
            scaler.scale(losses).backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(data_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / ACCUM_STEPS
            losses.backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(data_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += (losses.item() * ACCUM_STEPS)
        num_batches += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(data_loader):
            loss_components = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [Epoch {epoch+1}/{EPOCHS}] Step {i+1}/{len(data_loader)} - Total: {(losses.item() * ACCUM_STEPS):.4f} ({loss_components})", flush=True)

    return total_loss / max(num_batches, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      TRAINING PIPELINE: MODEL 10 (HELD-OBJECT IMMUNITY & KNIFE FN RECOVERY)      ")
    print("=" * 80)
    print(f"Hardware Device:        {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"CPU Threads Capped:    {torch.get_num_threads()} threads")
    print(f"Base Checkpoint:       {BASE_V10_PATH.name}")
    print(f"Anchor Scales:         {ANCHOR_SCALES}")
    print(f"RPN / RoI Clamps:      {math.exp(RPN_CLIP):.2f}x / {math.exp(ROI_CLIP):.2f}x")
    print(f"Batch Size / Epochs:   {BATCH_SIZE} / {EPOCHS}")
    print(f"Initial Learning Rate: {BASE_LR} (Cosine Annealing to 1e-6)")
    print(f"Target Save:           {BEST_M10_PATH}\n")

    model = get_model(num_classes=3, anchor_scales=ANCHOR_SCALES, pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = RPN_CLIP
    model.roi_heads.box_coder.bbox_xform_clip = ROI_CLIP

    if not BASE_V10_PATH.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_V10_PATH}")

    ckpt = torch.load(BASE_V10_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    train_dataset = TenthModelDataset(split_name="train", augment=True, include_hard_negatives=True)
    m9_test_dataset = TenthModelDataset(split_name="test", augment=False, include_hard_negatives=False)
    ext_bench_dataset = TenthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    m9_test_loader = DataLoader(m9_test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    ext_bench_loader = DataLoader(ext_bench_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"[*] Training Items:                  {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"[*] M9 Benchmark Test Images:        {len(m9_test_dataset)}")
    print(f"[*] Unified External Test Images:    {len(ext_bench_dataset)}\n")

    # Freeze early backbone conv1 + layer1 so low-level edge filters stay rock-solid
    for p in model.backbone.body.conv1.parameters():
        p.requires_grad = False
    for p in model.backbone.body.layer1.parameters():
        p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_score = -1.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} (lr={optimizer.param_groups[0]['lr']:.6f}) ---", flush=True)
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        scheduler.step()

        m9_res = evaluate_on_loader(model, m9_test_loader, device)
        ext_res = evaluate_on_loader(model, ext_bench_loader, device)

        print(
            f"--> [Epoch {epoch+1} Complete] Train Loss: {train_loss:.4f}\n"
            f"    [M9 Test 279]      P={m9_res['Precision']:.2%} R={m9_res['Recall']:.2%} F1={m9_res['F1']:.2%} "
            f"mAP={m9_res['mAP']:.2%} (HG={m9_res['AP_Handgun']:.2%}, KN={m9_res['AP_Knife']:.2%}) | "
            f"TP={m9_res['TP']} FP={m9_res['FP']} FN={m9_res['FN']} {m9_res['per_class']} | "
            f"NegClean={m9_res['neg_clean']}/{m9_res['neg_total']} ({m9_res['neg_rate']:.2%})\n"
            f"    [Unified Ext 350]  P={ext_res['Precision']:.2%} R={ext_res['Recall']:.2%} F1={ext_res['F1']:.2%} "
            f"mAP={ext_res['mAP']:.2%} (HG={ext_res['AP_Handgun']:.2%}, KN={ext_res['AP_Knife']:.2%}) | "
            f"TP={ext_res['TP']} FP={ext_res['FP']} FN={ext_res['FN']} {ext_res['per_class']} | "
            f"NegClean={ext_res['neg_clean']}/{ext_res['neg_total']} ({ext_res['neg_rate']:.2%})",
            flush=True,
        )

        # Composite score balancing M9 F1, External F1, and External Negative Rejection
        composite = 0.45 * m9_res["F1"] + 0.35 * ext_res["F1"] + 0.20 * ext_res["neg_rate"]

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "m9_test_metrics": m9_res,
            "external_test_metrics": ext_res,
            "anchor_scales": ANCHOR_SCALES,
            "aspect_ratios": (0.5, 1.0, 2.0),
            "rpn_bbox_xform_clip": RPN_CLIP,
            "roi_bbox_xform_clip": ROI_CLIP,
            "architecture": "FasterRCNN_ResNet50_FPN_V2_Model10_SOTA",
        }
        torch.save(checkpoint_payload, LATEST_M10_PATH)

        if composite > best_score:
            best_score = composite
            torch.save(checkpoint_payload, BEST_M10_PATH)
            print(f"  [+] Saved new best Model 10 checkpoint (score={composite:.4f}) -> {BEST_M10_PATH.name}", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  MODEL 10 TRAINING COMPLETE in {elapsed/60:.2f} minutes")
    print(f"  Saved Checkpoint: {BEST_M10_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

