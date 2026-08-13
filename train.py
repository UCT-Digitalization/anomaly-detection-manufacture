import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from torchvision.transforms import v2
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

DATA_ROOT = Path("data/step_01")
OUTPUT_ROOT = Path("outputs")

IMAGE_SIZE = (512, 512)
BACKBONE = "wide_resnet101_2"
FEATURE_LAYERS = ["layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.1

train_augmentations = v2.Compose([
    v2.RandomRotation(degrees=5),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

def main():

    # Check hardware
    print("=" * 60)
    print("PatchCore Training")
    print("=" * 60)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        accelerator = "gpu"
    else:
        print("GPU             : CPU mode")
        accelerator = "cpu"

    # Dataset
    datamodule = Folder(
        name="step_01",
        root=DATA_ROOT,
        # Only these images are used to build the normal model.
        normal_dir="train/good",
        # Normal images reserved for testing.
        normal_test_dir="test/good",
        # Abnormal images used only for evaluation.
        abnormal_dir="test/bad",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
        train_augmentations=train_augmentations,
    )

    # Image preprocessing
    pre_processor = Patchcore.configure_pre_processor(
        image_size=IMAGE_SIZE,
    )

    # PatchCore
    model = Patchcore(
        backbone=BACKBONE,
        layers=FEATURE_LAYERS,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        pre_processor=pre_processor,
    )

    # Training Engine
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=OUTPUT_ROOT,
    )

    # Build PatchCore memory bank
    print("\nBuilding PatchCore memory bank...\n")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nTraining finished.")

    # Evaluate immediately
    print("\nRunning test dataset...\n")

    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nTest results:")
    print(results)

if __name__ == "__main__":
    main()