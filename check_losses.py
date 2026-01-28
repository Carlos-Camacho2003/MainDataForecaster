"""Check model losses to diagnose poor predictions."""
import torch
import os

print("=" * 70)
print("MODEL LOSS ANALYSIS")
print("=" * 70)

for machine in ["DESF", "PICADORA"]:
    model_dir = f"models/{machine}"
    if not os.path.exists(model_dir):
        continue
    
    print(f"\n{machine}")
    print("-" * 50)
    
    losses = []
    for f in sorted(os.listdir(model_dir)):
        if f.endswith('.pth'):
            cp = torch.load(f'{model_dir}/{f}', map_location='cpu', weights_only=False)
            train_loss = cp["train_loss"]
            val_loss = cp["val_loss"]
            losses.append((f, train_loss, val_loss))
    
    # Sort by val_loss
    losses.sort(key=lambda x: x[2])
    
    print(f"{'Model':<45} {'Train':>8} {'Val':>8} {'Status'}")
    print("-" * 70)
    for name, train, val in losses:
        status = "OK" if val < 0.3 else ("WARN" if val < 0.5 else "POOR")
        print(f"{name:<45} {train:>8.4f} {val:>8.4f} {status}")
