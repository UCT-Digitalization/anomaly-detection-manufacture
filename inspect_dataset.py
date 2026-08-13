from pathlib import Path
from PIL import Image


DATA_ROOT = Path("data/step_01")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_folder(name: str, folder: Path) -> None:
    images = get_images(folder)

    print(f"\n{name}")
    print("-" * 50)
    print(f"Folder : {folder}")
    print(f"Images : {len(images)}")

    if not images:
        return

    # Inspect first image.
    first_image = images[0]

    try:
        with Image.open(first_image) as image:
            print(f"Example: {first_image.name}")
            print(f"Size   : {image.size}")
            print(f"Mode   : {image.mode}")

    except Exception as exc:
        print(f"ERROR reading {first_image}: {exc}")


def main():
    inspect_folder(
        "TRAIN GOOD",
        DATA_ROOT / "train" / "good",
    )

    inspect_folder(
        "TEST GOOD",
        DATA_ROOT / "test" / "good",
    )

    inspect_folder(
        "TEST BAD",
        DATA_ROOT / "test" / "bad",
    )


if __name__ == "__main__":
    main()