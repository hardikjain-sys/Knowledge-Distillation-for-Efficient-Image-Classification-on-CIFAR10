import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3

TEMPERATURE = 4.0
ALPHA = 0.5

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


# ============================================================
# CIFAR-10
# ============================================================

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


# ============================================================
# SAME train/validation split as before
# ============================================================

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


# ============================================================
# TEACHER
# ============================================================

class TeacherCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

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


# ============================================================
# STUDENT
# ============================================================

class StudentCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 3 × 32 × 32
            nn.Conv2d(
                3,
                8,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 8 × 16 × 16

            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16 × 8 × 8

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)

            # 32 × 4 × 4
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            32,
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


# ============================================================
# Create models
# ============================================================

teacher = TeacherCNN().to(DEVICE)
student = StudentCNN().to(DEVICE)


# ============================================================
# Load trained teacher
# ============================================================

teacher.load_state_dict(
    torch.load(
        "teacher_cifar10_best.pth",
        map_location=DEVICE
    )
)

print("Teacher loaded!")


# ============================================================
# Freeze teacher
# ============================================================

teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False


# ============================================================
# Model information
# ============================================================

student_params = sum(
    p.numel()
    for p in student.parameters()
)

student_size_kb = (
    student_params * 4 / 1024
)

teacher_params = sum(
    p.numel()
    for p in teacher.parameters()
)

teacher_size_mb = (
    teacher_params * 4 / (1024 ** 2)
)

print()
print("Teacher parameters:", teacher_params)
print(
    f"Teacher FP32 size: "
    f"{teacher_size_mb:.2f} MB"
)

print()

print("Student parameters:", student_params)
print(
    f"Student FP32 size: "
    f"{student_size_kb:.2f} KB"
)

print(
    f"Compression: "
    f"{teacher_params / student_params:.1f}x"
)


# ============================================================
# Losses
# ============================================================

classification_loss = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# Knowledge Distillation Loss
# ============================================================

def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    alpha
):

    # Normal classification loss
    ce_loss = classification_loss(
        student_logits,
        labels
    )


    # Soft teacher probabilities
    teacher_soft = torch.softmax(
        teacher_logits / temperature,
        dim=1
    )


    # Student log probabilities
    student_log_soft = torch.log_softmax(
        student_logits / temperature,
        dim=1
    )


    # KL divergence between teacher and student
    kd_loss = nn.functional.kl_div(
        student_log_soft,
        teacher_soft,
        reduction="batchmean"
    )


    # Temperature scaling
    kd_loss = kd_loss * (
        temperature ** 2
    )


    # Combined loss
    loss = (
        alpha * ce_loss
        +
        (1 - alpha) * kd_loss
    )


    return loss, ce_loss, kd_loss


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    student.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0

    correct = 0
    total = 0


    # ========================================================
    # TRAIN
    # ========================================================

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)


        optimizer.zero_grad()


        # ------------------------------------
        # Teacher prediction
        # ------------------------------------

        with torch.no_grad():

            teacher_logits = teacher(x)


        # ------------------------------------
        # Student prediction
        # ------------------------------------

        student_logits = student(x)


        # ------------------------------------
        # Distillation loss
        # ------------------------------------

        loss, ce_loss, kd_loss = distillation_loss(
            student_logits,
            teacher_logits,
            y,
            TEMPERATURE,
            ALPHA
        )


        # ------------------------------------
        # Backprop ONLY through student
        # ------------------------------------

        loss.backward()

        optimizer.step()


        # ------------------------------------
        # Statistics
        # ------------------------------------

        total_loss += loss.item()

        total_ce_loss += ce_loss.item()

        total_kd_loss += kd_loss.item()


        predictions = student_logits.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)


    train_accuracy = correct / total


    # ========================================================
    # VALIDATION
    # ========================================================

    student.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)


            student_logits = student(x)


            predictions = student_logits.argmax(
                dim=1
            )


            val_correct += (
                predictions == y
            ).sum().item()

            val_total += y.size(0)


    val_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler.step()


    # ========================================================
    # Save best student
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            student.state_dict(),
            "student_cifar10_distilled_best.pth"
        )

        best_marker = "  <-- BEST"

    else:

        best_marker = ""


    # ========================================================
    # Print
    # ========================================================

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: "
        f"{total_loss / len(train_loader):.4f} | "
        f"CE: "
        f"{total_ce_loss / len(train_loader):.4f} | "
        f"KD: "
        f"{total_kd_loss / len(train_loader):.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.4f} | "
        f"Val Acc: "
        f"{val_accuracy:.4f}"
        f"{best_marker}"
    )


# ============================================================
# Done
# ============================================================

print()
print("Distillation complete!")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Distilled student saved as: "
    "student_cifar10_distilled_best.pth"
)