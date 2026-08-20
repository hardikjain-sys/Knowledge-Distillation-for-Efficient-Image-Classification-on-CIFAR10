import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

train_transform = transforms.Compose([
    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

train_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=train_transform
)

val_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=val_transform
)

num_samples = len(train_full)

train_size = int(0.9 * num_samples)

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    num_samples,
    generator=generator
).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]


train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    val_full,
    val_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


class StudentCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 3 × 32 × 32
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16 × 16 × 16


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 × 8 × 8

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)

            # 64 × 4 × 4
        )


        # 64 × 4 × 4 → 64 × 1 × 1
        self.global_pool = nn.AdaptiveAvgPool2d(1)


        # 64 → 10
        self.classifier = nn.Linear(
            64,
            10
        )


    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x

model = StudentCNN().to(DEVICE)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

model_size_kb = (
    num_params * 4 / 1024
)

print("Parameters:", num_params)

print(
    f"FP32 model size: "
    f"{model_size_kb:.2f} KB"
)

print(
    f"Compression vs teacher: "
    f"{1422666 / num_params:.1f}x"
)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0


    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)


    train_accuracy = correct / total

    train_loss = (
        total_loss / len(train_loader)
    )


    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            predictions = logits.argmax(
                dim=1
            )

            val_correct += (
                predictions == y
            ).sum().item()

            val_total += y.size(0)


    val_accuracy = (
        val_correct / val_total
    )



    scheduler.step()

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "student_medium_baseline_best.pth"
        )

        marker = "  <-- BEST"

    else:

        marker = ""


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
        f"{marker}"
    )


print()
print("Training complete!")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Student saved as: "
    "student_medium_baseline_best.pth"
)