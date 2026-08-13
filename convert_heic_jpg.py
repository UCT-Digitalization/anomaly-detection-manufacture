from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


# Register HEIC/HEIF support with Pillow.
# Thumbnails and depth images are not needed for ordinary model-training images.
register_heif_opener(
    thumbnails=False,
    depth_images=False,
)


SUPPORTED_EXTENSIONS = {".heic", ".heif"}


def convert_image(
    input_path: Path,
    output_path: Path,
    output_format: str,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> bool:
    """
    Convert one HEIC/HEIF image to PNG or JPEG.

    Returns True if converted, or False if skipped.
    """

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Output exists: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        # Apply the camera orientation stored in EXIF metadata.
        image = ImageOps.exif_transpose(image)

        # Force a consistent 3-channel representation for model training.
        image = image.convert("RGB")

        if output_format == "png":
            image.save(
                output_path,
                format="PNG",
                optimize=False,
                compress_level=3,
            )

        elif output_format in {"jpg", "jpeg"}:
            image.save(
                output_path,
                format="JPEG",
                quality=jpeg_quality,
                subsampling=0,
                optimize=True,
            )

        else:
            raise ValueError(
                f"Unsupported output format: {output_format}"
            )

    print(f"[OK] {input_path} -> {output_path}")
    return True


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    output_format: str,
    jpeg_quality: int,
    recursive: bool,
    overwrite: bool,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    pattern = "**/*" if recursive else "*"

    input_files = sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not input_files:
        print(f"No HEIC or HEIF images found in: {input_dir}")
        return

    converted = 0
    skipped = 0
    failed = 0

    output_suffix = ".jpg" if output_format in {"jpg", "jpeg"} else ".png"

    for input_path in input_files:
        relative_path = input_path.relative_to(input_dir)
        output_path = (
            output_dir / relative_path
        ).with_suffix(output_suffix)

        try:
            was_converted = convert_image(
                input_path=input_path,
                output_path=output_path,
                output_format=output_format,
                jpeg_quality=jpeg_quality,
                overwrite=overwrite,
            )

            if was_converted:
                converted += 1
            else:
                skipped += 1

        except Exception as error:
            failed += 1
            print(f"[ERROR] {input_path}: {error}")

    print()
    print("Conversion summary")
    print("------------------")
    print(f"Found:     {len(input_files)}")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Output:    {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HEIC/HEIF images to PNG or JPEG."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing HEIC/HEIF images.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for converted images.",
    )

    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Output format. Default: png",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100. Default: 95",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search inside subdirectories.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.format.lower(),
        jpeg_quality=args.quality,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()