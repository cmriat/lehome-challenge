import argparse
import json
import shutil
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required. Run this script with the project environment, for example: pixi run python scripts/add_garment_type_prob.py ..."
    ) from exc

FEATURE_NAME = "observation.garment_type_prob"
FEATURE_DEF = {
    "dtype": "float32",
    "shape": [4],
    "names": [
        "top-long-sleeve",
        "top-short-sleeve",
        "long-pant",
        "short-pant",
    ],
}
CLASS_TO_PROB = {
    "top_long": [1.0, 0.0, 0.0, 0.0],
    "top_short": [0.0, 1.0, 0.0, 0.0],
    "pant_long": [0.0, 0.0, 1.0, 0.0],
    "pant_short": [0.0, 0.0, 0.0, 1.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add garment type probability feature to a LeRobot dataset."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--episode_class_path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_episode_prob_map(episode_class_path: Path) -> dict[int, list[float]]:
    with episode_class_path.open("r", encoding="utf-8") as f:
        classes = json.load(f)

    if not isinstance(classes, dict):
        raise ValueError(f"Invalid episode class mapping in {episode_class_path}")

    episode_prob_map: dict[int, list[float]] = {}
    for episode_key, class_name in classes.items():
        try:
            episode_index = int(episode_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid episode key: {episode_key}") from exc

        if class_name not in CLASS_TO_PROB:
            raise ValueError(f"Unknown class '{class_name}' for episode {episode_index}")

        episode_prob_map[episode_index] = CLASS_TO_PROB[class_name]

    return episode_prob_map


def copy_non_data_dirs(dataset_root: Path, output_root: Path, overwrite: bool) -> None:
    for subdir in ["meta", "videos", "images"]:
        src = dataset_root / subdir
        dst = output_root / subdir
        if not src.exists():
            continue
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} already exists, use --overwrite to replace")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def update_info_json(info_path: Path, overwrite: bool) -> None:
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.setdefault("features", {})
    if FEATURE_NAME in features and not overwrite:
        raise ValueError(f"{FEATURE_NAME} already exists in {info_path}")

    features[FEATURE_NAME] = FEATURE_DEF

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_type_prob_column(table: pa.Table, episode_prob_map: dict[int, list[float]]) -> pa.Array:
    if "episode_index" not in table.column_names:
        raise KeyError("Missing episode_index column")

    values = []
    missing_episode_indices: set[int] = set()
    for episode_index in table["episode_index"].to_pylist():
        normalized_episode_index = int(episode_index)
        prob = episode_prob_map.get(normalized_episode_index)
        if prob is None:
            missing_episode_indices.add(normalized_episode_index)
            continue
        values.append(prob)

    if missing_episode_indices:
        missing_text = ", ".join(map(str, sorted(missing_episode_indices)[:20]))
        raise ValueError(f"Missing class mapping for episode_index: {missing_text}")

    return pa.array(values, type=pa.list_(pa.float32(), 4))


def add_type_prob_to_parquet(
    src_path: Path,
    dst_path: Path,
    episode_prob_map: dict[int, list[float]],
    overwrite: bool,
) -> None:
    table = pq.read_table(src_path)
    type_prob_col = build_type_prob_column(table, episode_prob_map)

    if FEATURE_NAME in table.column_names:
        if not overwrite:
            raise ValueError(f"{FEATURE_NAME} already exists in {src_path}")
        column_index = table.column_names.index(FEATURE_NAME)
        table = table.remove_column(column_index)
        table = table.add_column(column_index, FEATURE_NAME, type_prob_col)
    else:
        table = table.append_column(FEATURE_NAME, type_prob_col)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    episode_class_path = (
        args.episode_class_path.resolve()
        if args.episode_class_path
        else (dataset_root / "meta" / "episode_class.json")
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not episode_class_path.exists():
        raise FileNotFoundError(f"episode_class.json not found: {episode_class_path}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output root already exists and is not empty: {output_root}")

    episode_prob_map = load_episode_prob_map(episode_class_path)
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    copy_non_data_dirs(dataset_root, output_root, overwrite=args.overwrite)

    for index, src_path in enumerate(parquet_files, start=1):
        rel_path = src_path.relative_to(dataset_root)
        dst_path = output_root / rel_path
        print(f"[{index}/{len(parquet_files)}] {rel_path}")
        add_type_prob_to_parquet(src_path, dst_path, episode_prob_map, args.overwrite)

    update_info_json(output_root / "meta" / "info.json", overwrite=args.overwrite)
    print(f"Done: added {FEATURE_NAME} to {output_root}")


if __name__ == "__main__":
    main()
