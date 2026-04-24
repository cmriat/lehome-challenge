#!/usr/bin/env python3

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(f"Failed to import pyarrow.parquet: {exc}")

try:
    import av
except ImportError as exc:
    raise SystemExit(f"Failed to import PyAV: {exc}")


@dataclass(frozen=True)
class CameraEpisodeMeta:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class VideoInfo:
    relative_path: str
    duration_s: float | None
    avg_fps: float | None
    frames: int | None
    last_pts_time: float | None
    tail_pts_times: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check LeRobot dataset video/parquet timestamp consistency before training."
    )
    parser.add_argument("--dataset_root", required=True, help="Dataset root path")
    parser.add_argument(
        "--cameras",
        default="top_rgb,left_rgb,right_rgb",
        help="Comma-separated cameras, e.g. top_rgb,left_rgb,right_rgb",
    )
    parser.add_argument(
        "--mode",
        choices=("sample", "full", "episode"),
        default="sample",
        help="Check mode",
    )
    parser.add_argument("--episode", type=int, help="Episode index for --mode episode")
    parser.add_argument(
        "--sample_episodes", type=int, default=100, help="Episode count in sample mode"
    )
    parser.add_argument(
        "--sample_tail", type=int, default=8, help="Tail frame count to inspect per video"
    )
    parser.add_argument(
        "--max_errors", type=int, default=20, help="Stop after this many errors"
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--verbose", action="store_true", help="Print detailed findings")
    parser.add_argument("--report_json", help="Optional JSON report path")
    return parser.parse_args()


def normalize_cameras(raw: str) -> list[str]:
    cameras = [item.strip() for item in raw.split(",") if item.strip()]
    if not cameras:
        raise ValueError("No cameras provided")
    return cameras


def fail(message: str, exit_code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def read_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        fail(f"Missing info.json: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_video_feature(camera: str) -> str:
    return f"observation.images.{camera}"


def require_paths(dataset_root: Path, cameras: list[str]) -> None:
    required = [dataset_root / "data", dataset_root / "meta/episodes", dataset_root / "videos"]
    for path in required:
        if not path.exists():
            fail(f"Missing required path: {path}")
    for camera in cameras:
        path = dataset_root / "videos" / get_video_feature(camera)
        if not path.exists():
            fail(f"Missing camera videos path: {path}")


def list_parquet_files(root: Path) -> list[Path]:
    files = sorted(root.glob("chunk-*/file-*.parquet"))
    if not files:
        fail(f"No parquet files found under: {root}")
    return files


def build_episode_stats(data_files: list[Path]) -> dict[int, dict[str, float | int]]:
    stats: dict[int, dict[str, float | int]] = {}
    for file_path in data_files:
        table = pq.read_table(file_path, columns=["episode_index", "timestamp", "frame_index"])
        episodes = table.column("episode_index").to_pylist()
        timestamps = table.column("timestamp").to_pylist()
        frame_indices = table.column("frame_index").to_pylist()
        for episode_index, timestamp, frame_index in zip(episodes, timestamps, frame_indices, strict=True):
            entry = stats.setdefault(
                int(episode_index),
                {
                    "min_timestamp": float(timestamp),
                    "max_timestamp": float(timestamp),
                    "min_frame_index": int(frame_index),
                    "max_frame_index": int(frame_index),
                    "count": 0,
                },
            )
            entry["min_timestamp"] = min(float(entry["min_timestamp"]), float(timestamp))
            entry["max_timestamp"] = max(float(entry["max_timestamp"]), float(timestamp))
            entry["min_frame_index"] = min(int(entry["min_frame_index"]), int(frame_index))
            entry["max_frame_index"] = max(int(entry["max_frame_index"]), int(frame_index))
            entry["count"] = int(entry["count"]) + 1
    return stats


def build_episode_meta(
    episode_files: list[Path], cameras: list[str]
) -> dict[int, dict[str, CameraEpisodeMeta]]:
    columns = ["episode_index"]
    for camera in cameras:
        feature = get_video_feature(camera)
        columns.extend(
            [
                f"videos/{feature}/chunk_index",
                f"videos/{feature}/file_index",
                f"videos/{feature}/from_timestamp",
                f"videos/{feature}/to_timestamp",
            ]
        )

    mapping: dict[int, dict[str, CameraEpisodeMeta]] = {}
    for file_path in episode_files:
        table = pq.read_table(file_path, columns=columns)
        rows = table.to_pylist()
        for row in rows:
            episode_index = int(row["episode_index"])
            camera_meta: dict[str, CameraEpisodeMeta] = {}
            for camera in cameras:
                feature = get_video_feature(camera)
                camera_meta[camera] = CameraEpisodeMeta(
                    chunk_index=int(row[f"videos/{feature}/chunk_index"]),
                    file_index=int(row[f"videos/{feature}/file_index"]),
                    from_timestamp=float(row[f"videos/{feature}/from_timestamp"]),
                    to_timestamp=float(row[f"videos/{feature}/to_timestamp"]),
                )
            mapping[episode_index] = camera_meta
    if not mapping:
        fail("No episode metadata loaded")
    return mapping


def inspect_video(video_path: Path, tail_count: int) -> VideoInfo:
    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open video {video_path}: {exc}") from exc

    with container:
        stream = next((s for s in container.streams.video), None)
        if stream is None:
            raise RuntimeError(f"No video stream found: {video_path}")

        avg_fps = None
        if stream.average_rate is not None:
            avg_fps = float(stream.average_rate)
        elif stream.base_rate is not None:
            avg_fps = float(stream.base_rate)

        duration_s = None
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration / av.time_base)

        frames = int(stream.frames) if stream.frames else None
        tail_pts: list[float] = []
        last_pts_time = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            pts_time = float(frame.pts * frame.time_base)
            tail_pts.append(pts_time)
            if len(tail_pts) > tail_count:
                tail_pts.pop(0)
            last_pts_time = pts_time

        return VideoInfo(
            relative_path=str(video_path),
            duration_s=duration_s,
            avg_fps=avg_fps,
            frames=frames,
            last_pts_time=last_pts_time,
            tail_pts_times=tuple(tail_pts),
        )


def collect_video_infos(
    dataset_root: Path,
    episode_meta: dict[int, dict[str, CameraEpisodeMeta]],
    cameras: list[str],
    tail_count: int,
) -> dict[str, VideoInfo]:
    needed: dict[str, Path] = {}
    for camera_map in episode_meta.values():
        for camera in cameras:
            meta = camera_map[camera]
            rel = Path("videos") / get_video_feature(camera) / f"chunk-{meta.chunk_index:03d}" / f"file-{meta.file_index:03d}.mp4"
            needed[str(rel)] = dataset_root / rel

    infos: dict[str, VideoInfo] = {}
    for rel, abs_path in sorted(needed.items()):
        if not abs_path.exists():
            raise RuntimeError(f"Missing video file: {abs_path}")
        infos[rel] = inspect_video(abs_path, tail_count=tail_count)
    return infos


def choose_episodes(
    episode_stats: dict[int, dict[str, float | int]],
    episode_meta: dict[int, dict[str, CameraEpisodeMeta]],
    mode: str,
    episode: int | None,
    sample_episodes: int,
) -> list[int]:
    all_episodes = sorted(set(episode_stats) & set(episode_meta))
    if not all_episodes:
        fail("No overlapping episode stats/meta entries found")
    if mode == "episode":
        if episode is None:
            fail("--episode is required when --mode episode")
        if episode not in episode_stats or episode not in episode_meta:
            fail(f"Episode not found: {episode}")
        return [episode]
    if mode == "full":
        return all_episodes

    chosen = set()
    rng = random.Random(0)
    chosen.update(all_episodes[: min(len(all_episodes), max(1, sample_episodes))])
    if len(chosen) < min(sample_episodes, len(all_episodes)):
        chosen.update(rng.sample(all_episodes, k=min(sample_episodes, len(all_episodes))))
    chosen.add(all_episodes[-1])
    return sorted(chosen)


def add_issue(issues: list[dict[str, Any]], level: str, **payload: Any) -> None:
    item = {"level": level, **payload}
    issues.append(item)


def check_episode_camera(
    episode_index: int,
    camera: str,
    stats: dict[str, float | int],
    meta: CameraEpisodeMeta,
    video: VideoInfo,
    declared_fps: float | None,
    strict: bool,
    issues: list[dict[str, Any]],
) -> None:
    feature = get_video_feature(camera)
    expected_start = meta.from_timestamp + float(stats["min_timestamp"])
    expected_max = meta.from_timestamp + float(stats["max_timestamp"])
    expected_last = expected_max
    rel = str(Path("videos") / feature / f"chunk-{meta.chunk_index:03d}" / f"file-{meta.file_index:03d}.mp4")

    if expected_start < meta.from_timestamp - 1e-6:
        add_issue(
            issues,
            "error",
            reason="episode_start_before_video_window",
            episode_index=episode_index,
            camera=camera,
            video=rel,
            expected_start=expected_start,
            from_timestamp=meta.from_timestamp,
        )

    if expected_max > meta.to_timestamp + 1e-6:
        add_issue(
            issues,
            "error",
            reason="episode_timestamp_exceeds_meta_window",
            episode_index=episode_index,
            camera=camera,
            video=rel,
            expected_max=expected_max,
            to_timestamp=meta.to_timestamp,
        )

    if video.avg_fps is not None and declared_fps is not None:
        fps_delta = abs(video.avg_fps - declared_fps)
        if fps_delta > 1e-2:
            add_issue(
                issues,
                "error" if strict else "warning",
                reason="fps_mismatch",
                episode_index=episode_index,
                camera=camera,
                video=rel,
                video_fps=video.avg_fps,
                declared_fps=declared_fps,
                delta=fps_delta,
            )

    coverage_limit = video.last_pts_time
    if coverage_limit is None and video.duration_s is not None:
        coverage_limit = video.duration_s
    if coverage_limit is None:
        add_issue(
            issues,
            "error",
            reason="video_has_no_decodable_timestamps",
            episode_index=episode_index,
            camera=camera,
            video=rel,
        )
        return

    tolerance = (1.0 / declared_fps) * 0.5 if declared_fps else 0.02
    if expected_last - coverage_limit > tolerance:
        add_issue(
            issues,
            "error",
            reason="video_tail_does_not_cover_expected_timestamp",
            episode_index=episode_index,
            camera=camera,
            video=rel,
            expected_last_timestamp=expected_last,
            actual_last_timestamp=coverage_limit,
            delta=expected_last - coverage_limit,
            meta_to_timestamp=meta.to_timestamp,
        )

    if video.tail_pts_times:
        tail_last = video.tail_pts_times[-1]
        if expected_max - tail_last > tolerance:
            add_issue(
                issues,
                "error",
                reason="tail_frame_gap_detected",
                episode_index=episode_index,
                camera=camera,
                video=rel,
                expected_max_timestamp=expected_max,
                actual_tail_last=tail_last,
                delta=expected_max - tail_last,
            )


def summarize(issues: list[dict[str, Any]]) -> dict[str, int]:
    errors = sum(1 for item in issues if item["level"] == "error")
    warnings = sum(1 for item in issues if item["level"] == "warning")
    return {"errors": errors, "warnings": warnings}


def print_summary(
    dataset_root: Path,
    cameras: list[str],
    selected_episodes: list[int],
    video_infos: dict[str, VideoInfo],
    issues: list[dict[str, Any]],
    verbose: bool,
) -> None:
    summary = summarize(issues)
    print(f"dataset_root: {dataset_root}")
    print(f"cameras: {','.join(cameras)}")
    print(f"checked_episodes: {len(selected_episodes)}")
    print(f"checked_videos: {len(video_infos)}")
    print(f"errors: {summary['errors']}")
    print(f"warnings: {summary['warnings']}")

    if verbose or issues:
        for item in issues:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))


def write_report(
    report_path: Path,
    dataset_root: Path,
    cameras: list[str],
    selected_episodes: list[int],
    issues: list[dict[str, Any]],
) -> None:
    report = {
        "dataset_root": str(dataset_root),
        "cameras": cameras,
        "checked_episodes": selected_episodes,
        "summary": summarize(issues),
        "issues": issues,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    cameras = normalize_cameras(args.cameras)
    require_paths(dataset_root, cameras)

    info = read_info(dataset_root)
    declared_fps = float(info.get("fps")) if info.get("fps") is not None else None

    data_files = list_parquet_files(dataset_root / "data")
    episode_files = list_parquet_files(dataset_root / "meta/episodes")

    episode_stats = build_episode_stats(data_files)
    episode_meta = build_episode_meta(episode_files, cameras)
    selected_episodes = choose_episodes(
        episode_stats=episode_stats,
        episode_meta=episode_meta,
        mode=args.mode,
        episode=args.episode,
        sample_episodes=args.sample_episodes,
    )

    selected_episode_meta = {episode: episode_meta[episode] for episode in selected_episodes}
    video_infos = collect_video_infos(
        dataset_root=dataset_root,
        episode_meta=selected_episode_meta,
        cameras=cameras,
        tail_count=max(1, args.sample_tail),
    )

    issues: list[dict[str, Any]] = []
    for episode_index in selected_episodes:
        stats = episode_stats[episode_index]
        for camera in cameras:
            meta = episode_meta[episode_index][camera]
            rel = str(
                Path("videos")
                / get_video_feature(camera)
                / f"chunk-{meta.chunk_index:03d}"
                / f"file-{meta.file_index:03d}.mp4"
            )
            video = video_infos[rel]
            check_episode_camera(
                episode_index=episode_index,
                camera=camera,
                stats=stats,
                meta=meta,
                video=video,
                declared_fps=declared_fps,
                strict=args.strict,
                issues=issues,
            )
            if summarize(issues)["errors"] >= args.max_errors:
                break
        if summarize(issues)["errors"] >= args.max_errors:
            break

    print_summary(
        dataset_root=dataset_root,
        cameras=cameras,
        selected_episodes=selected_episodes,
        video_infos=video_infos,
        issues=issues,
        verbose=args.verbose,
    )

    if args.report_json:
        write_report(Path(args.report_json).resolve(), dataset_root, cameras, selected_episodes, issues)

    summary = summarize(issues)
    if summary["errors"] > 0:
        return 1
    if args.strict and summary["warnings"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
