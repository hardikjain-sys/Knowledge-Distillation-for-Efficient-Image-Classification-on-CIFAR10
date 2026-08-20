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

print("Dataset size:", len(train_full))

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


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


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
x, y = train_dataset[0]

print("Image shape:", x.shape)
print("Label:", y)


class TeacherCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 3 x 32 x 32
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            # 64 x 32 x 32
            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            # 64 x 32 x 32
            nn.MaxPool2d(2),

            # 64 x 16 x 16


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            # 128 x 16 x 16
            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            # 128 x 16 x 16
            nn.MaxPool2d(2),

            # 128 x 8 x 8

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                10
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

model = TeacherCNN().to(DEVICE)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

model_size_mb = (
    num_params * 4
    / (1024 ** 2)
)

print("Parameters:", num_params)

print(
    f"FP32 model size: "
    f"{model_size_mb:.2f} MB"
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

    current_lr = optimizer.param_groups[0]["lr"]


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "teacher_cifar10_best.pth"
        )

        best_marker = "  <-- BEST"

    else:

        best_marker = ""

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6f}"
        f"{best_marker}"
    )

print()
print("Training complete!")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Best teacher saved as: "
    "teacher_cifar10_best.pth"
)