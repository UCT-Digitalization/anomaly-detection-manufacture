import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything



# Configuration
seed_everything(42, workers=True)

DATA_ROOT = Path("data/step_01")

CHECKPOINT = Path("outputs/Patchcore/step_01/v0/weights/lightning/model.ckpt")

IMAGE_SIZE = (512, 512)
BACKBONE = "wide_resnet101_2"
FEATURE_LAYERS = ["layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.1


# Helper
def get_scalar(value):
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach().cpu()

    if hasattr(value, "numel") and value.numel() == 1:
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    return value

# Prediction
def predict_test_dataset(engine, model, datamodule):
    results = []

    predictions = engine.predict(
        model=model,
        ckpt_path=str(CHECKPOINT),
        datamodule=datamodule,
    )

    for batch in predictions:

        paths = getattr(batch, "image_path", None)
        scores = getattr(batch, "pred_score", None)
        labels = getattr(batch, "pred_label", None)

        if paths is None:
            continue

        if isinstance(paths, (str, Path)):
            paths = [paths]

        for i, path in enumerate(paths):

            path = Path(path)

            score = get_scalar(
                scores[i]
                if scores is not None
                else None
            )

            label = get_scalar(
                labels[i]
                if labels is not None
                else None
            )

            # Get ACTUAL label from folder
            # test/good/image.jpg -> GOOD
            # test/bad/image.jpg  -> BAD
            parent_folder = path.parent.name.lower()

            if parent_folder == "good":
                actual_label = "GOOD"

            elif parent_folder == "bad":
                actual_label = "BAD"

            else:
                actual_label = "UNKNOWN"

            # Model prediction
            if label is None:
                predicted = "UNKNOWN"

            elif int(label) == 1:
                predicted = "ANOMALY"

            else:
                predicted = "NORMAL"

            # Expected prediction
            if actual_label == "GOOD":
                expected_prediction = "NORMAL"

            elif actual_label == "BAD":
                expected_prediction = "ANOMALY"

            else:
                expected_prediction = "UNKNOWN"

            correct = predicted == expected_prediction

            results.append({
                "filename": path.name,
                "actual": actual_label,
                "score": score,
                "predicted": predicted,
                "correct": correct,
            })

    return results


# Print table
def print_table(results):

    print()

    print(
        f"{'Filename':<35}"
        f"{'Actual':<10}"
        f"{'Score':<12}"
        f"{'Predicted':<12}"
        f"{'Correct':<10}"
    )

    print("-" * 79)

    for result in results:

        score = (
            f"{result['score']:.4f}"
            if result["score"] is not None
            else "N/A"
        )

        correct = (
            "YES"
            if result["correct"]
            else "NO"
        )

        print(
            f"{result['filename']:<35}"
            f"{result['actual']:<10}"
            f"{score:<12}"
            f"{result['predicted']:<12}"
            f"{correct:<10}"
        )

# Summary
def print_summary(results):

    total = len(results)

    correct = sum(r["correct"] for r in results)

    incorrect = total - correct

    good_results = [r for r in results if r["actual"] == "GOOD"]

    bad_results = [r for r in results if r["actual"] == "BAD"]

    false_rejects = sum(1 for r in good_results if r["predicted"] == "ANOMALY")

    false_accepts = sum(1 for r in bad_results if r["predicted"] == "NORMAL")

    true_accepts = sum(1 for r in good_results if r["predicted"] == "NORMAL")

    true_rejects = sum(1 for r in bad_results if r["predicted"] == "ANOMALY")

    accuracy = (correct / total * 100 if total > 0 else 0)

    false_reject_rate = (false_rejects/len(good_results) * 100 if good_results else 0)

    false_accept_rate = (false_accepts / len(bad_results) * 100 if bad_results else 0)

    good_accept_rate = (true_accepts/ len(good_results)* 100 if good_results else 0)

    bad_detection_rate = (true_rejects / len(bad_results)* 100 if bad_results else 0)

    print()
    print("Summary")
    print("-" * 45)

    print(f"Total images         : {total}")
    print(f"GOOD images          : {len(good_results)}")
    print(f"BAD images           : {len(bad_results)}")

    print()

    print(f"Correct predictions  : {correct}")
    print(f"Wrong predictions    : {incorrect}")
    print(f"Accuracy             : {accuracy:.2f}%")

    print()

    print(
        f"False Accepts        : "
        f"{false_accepts}/{len(bad_results)}")

    print(
        f"False Accept Rate    : "
        f"{false_accept_rate:.2f}%")

    print()

    print(
        f"False Rejects        : "
        f"{false_rejects}/{len(good_results)}")

    print(
        f"False Reject Rate    : "
        f"{false_reject_rate:.2f}%")

    print()

    print(
        f"BAD Detection Rate   : "
        f"{bad_detection_rate:.2f}%")

    print(
        f"GOOD Acceptance Rate : "
        f"{good_accept_rate:.2f}%")

# Main
def main():

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_ROOT.resolve()}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT.resolve()}")

    # Accelerator
    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "cpu")

    print(f"\nUsing accelerator: {accelerator.upper()}")

    # EXACT SAME preprocessor as notebook
    pre_processor = Patchcore.configure_pre_processor(
        image_size=IMAGE_SIZE)

    # EXACT SAME PatchCore configuration
    model = Patchcore(
        backbone=BACKBONE,
        layers=FEATURE_LAYERS,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        pre_processor=pre_processor,)

    # Test datamodule for all test images for good and bad
    datamodule_test = Folder(
        name="step_01",
        root=DATA_ROOT,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=2,
        val_split_mode=ValSplitMode.NONE,)

    # PatchCore Engine
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,)

    # Prediction
    print("\nRunning prediction...\n")

    results = predict_test_dataset(
        engine,
        model,
        datamodule_test,)

    print_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()