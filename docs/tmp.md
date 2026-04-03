pixi run dataset-sim replay --dataset_root Datasets/record/001 --num_replays 1 --disable_depth --device cpu --enable_cameras 

pixi run dataset-sim replay \
      --dataset_root Datasets/example/four_types_merged \
      --start_episode 0 \
      --end_episode 4 \
      --num_replays 1 \
      --disable_depth \
      --device cpu \
      --enable_cameras \

[INFO]: Completed setting up the environment...
2026-03-19 16:43:21 - replay_standalone - INFO - Initializing environment...
2026-03-19 16:43:21 - replay_standalone - INFO - Environment initialized
2026-03-19 16:43:21 - replay_standalone - INFO - Replaying episodes 0 to 3 (total 4)
2026-03-19 16:43:21 - replay_standalone - INFO - 
2026-03-19 16:43:21 - replay_standalone - INFO - ============================================================
2026-03-19 16:43:21 - replay_standalone - INFO - Episode 1/4 (index=0)
2026-03-19 16:43:21 - replay_standalone - INFO - ============================================================
2026-03-19 16:43:21 - replay_standalone - INFO -   Frames: 364
2026-03-19 16:43:21 - lehome.assets.object.Garment - INFO - [GarmentObject] Reset complete - pos: [-0.014497632368341631, -0.0018161215060495499, 0.67], ori: [18.74873421698507, 19.765769869113967, 0.0]
Warp DeprecationWarning: The symbol `warp.types.warp_type_to_np_dtype` will soon be removed from the public API. It can still be accessed from `warp._src.types.warp_type_to_np_dtype` but might be changed or removed without notice.
2026-03-19 16:43:21 - lehome.assets.object.Garment - INFO - [GarmentObject] set_all_pose Reset complete - pos: [-0.04258476197719574, -0.016093116253614426, 0.6700000166893005], ori: [-2.393900156021118, -16.63031768798828, 0.0]
  Replaying:   4%|███████▊                                                                                                                                                                                                    | 14/364 [00:00<00:17, 19.46frame/s]2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.19 <= 9.450000000000001 -> ✗
2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 37.72 <= 12.15 -> ✗
2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.72 <= 9.0 -> ✗
2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.62 >= 13.05 -> ✓
2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.07 >= 8.55 -> ✓
2026-03-19 16:43:23 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  10%|████████████████████▋                                                                                                                                                                                       | 37/364 [00:02<00:16, 19.45frame/s]2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.22 <= 9.450000000000001 -> ✗
2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 37.82 <= 12.15 -> ✗
2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.90 <= 9.0 -> ✗
2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.69 >= 13.05 -> ✓
2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.13 >= 8.55 -> ✓
2026-03-19 16:43:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  18%|███████████████████████████████████▊                                                                                                                                                                        | 64/364 [00:03<00:14, 20.07frame/s]2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.23 <= 9.450000000000001 -> ✗
2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 36.75 <= 12.15 -> ✗
2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.95 <= 9.0 -> ✗
2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.71 >= 13.05 -> ✓
2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.17 >= 8.55 -> ✓
2026-03-19 16:43:26 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  24%|█████████████████████████████████████████████████▎                                                                                                                                                          | 88/364 [00:04<00:13, 20.16frame/s]2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.23 <= 9.450000000000001 -> ✗
2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 27.23 <= 12.15 -> ✗
2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.24 <= 9.0 -> ✗
2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.78 >= 13.05 -> ✓
2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.23 >= 8.55 -> ✓
2026-03-19 16:43:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  31%|███████████████████████████████████████████████████████████████                                                                                                                                            | 113/364 [00:06<00:12, 20.05frame/s]2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.24 <= 9.450000000000001 -> ✗
2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 18.69 <= 12.15 -> ✗
2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.39 <= 9.0 -> ✗
2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.99 >= 13.05 -> ✓
2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.29 >= 8.55 -> ✓
2026-03-19 16:43:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  38%|█████████████████████████████████████████████████████████████████████████████▌                                                                                                                             | 139/364 [00:07<00:12, 18.43frame/s]2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.24 <= 9.450000000000001 -> ✗
2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 18.74 <= 12.15 -> ✗
2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.45 <= 9.0 -> ✗
2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 19.01 >= 13.05 -> ✓
2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.31 >= 8.55 -> ✓
2026-03-19 16:43:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  45%|███████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                               | 164/364 [00:08<00:11, 17.86frame/s]2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.29 <= 9.450000000000001 -> ✗
2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 18.08 <= 12.15 -> ✗
2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.47 <= 9.0 -> ✗
2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 19.02 >= 13.05 -> ✓
2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.29 >= 8.55 -> ✓
2026-03-19 16:43:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  52%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                 | 189/364 [00:10<00:09, 18.05frame/s]2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.07 <= 9.450000000000001 -> ✗
2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 16.85 <= 12.15 -> ✗
2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.48 <= 9.0 -> ✗
2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.48 >= 13.05 -> ✓
2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.30 >= 8.55 -> ✓
2026-03-19 16:43:32 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  59%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                   | 214/364 [00:11<00:07, 19.05frame/s]2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.42 <= 9.450000000000001 -> ✗
2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.11 <= 12.15 -> ✓
2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.51 <= 9.0 -> ✗
2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.73 >= 13.05 -> ✓
2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.36 >= 8.55 -> ✓
2026-03-19 16:43:34 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  65%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                      | 238/364 [00:12<00:06, 19.37frame/s]2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.59 <= 9.450000000000001 -> ✗
2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.08 <= 12.15 -> ✓
2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.52 <= 9.0 -> ✗
2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.78 >= 13.05 -> ✓
2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.35 >= 8.55 -> ✓
2026-03-19 16:43:35 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  72%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                         | 262/364 [00:13<00:05, 20.18frame/s]2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.94 <= 9.450000000000001 -> ✗
2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.03 <= 12.15 -> ✓
2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.63 <= 9.0 -> ✗
2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.81 >= 13.05 -> ✓
2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.34 >= 8.55 -> ✓
2026-03-19 16:43:36 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  79%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                          | 288/364 [00:15<00:04, 18.04frame/s]2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 24.70 <= 9.450000000000001 -> ✗
2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.06 <= 12.15 -> ✓
2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 26.40 <= 9.0 -> ✗
2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.81 >= 13.05 -> ✓
2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 16.11 >= 8.55 -> ✓
2026-03-19 16:43:37 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  86%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                            | 314/364 [00:16<00:02, 20.32frame/s]2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 7.86 <= 9.450000000000001 -> ✓
2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.44 <= 12.15 -> ✓
2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 19.45 <= 9.0 -> ✗
2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.95 >= 13.05 -> ✓
2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 19.12 >= 8.55 -> ✓
2026-03-19 16:43:39 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying:  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉               | 337/364 [00:17<00:01, 19.69frame/s]2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 3.13 <= 9.450000000000001 -> ✓
2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.30 <= 12.15 -> ✓
2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 19.99 <= 9.0 -> ✗
2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.90 >= 13.05 -> ✓
2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 20.81 >= 8.55 -> ✓
2026-03-19 16:43:40 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
  Replaying: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 364/364 [00:19<00:00, 18.99frame/s]
2026-03-19 16:43:41 - replay_standalone - INFO -   Result: Failed


[INFO]: Completed setting up the environment...
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - Initializing observations...
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - Observations initialized successfully
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - Replaying episodes 0 to 3 (displayed as 1 to 4)
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - 
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - ============================================================
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - Episode 1/4
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - ============================================================
2026-03-19 16:52:09 - scripts.utils.dataset_replay - INFO - Episode length: 364 frames
2026-03-19 16:52:09 - lehome.assets.object.Garment - INFO - [GarmentObject] Reset complete - pos: [-0.06525139389171754, 0.005687144242335945, 0.67], ori: [10.17143241684871, -11.447679909741298, 0.0]
Warp DeprecationWarning: The symbol `warp.types.warp_type_to_np_dtype` will soon be removed from the public API. It can still be accessed from `warp._src.types.warp_type_to_np_dtype` but might be changed or removed without notice.
2026-03-19 16:52:09 - lehome.assets.object.Garment - INFO - [GarmentObject] set_all_pose Reset complete - pos: [-0.04258476197719574, -0.016093116253614426, 0.6700000166893005], ori: [-2.393900156021118, -16.63031768798828, 0.0]
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.85 <= 9.450000000000001 -> ✗
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 37.51 <= 12.15 -> ✗
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.42 <= 9.0 -> ✗
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.78 >= 13.05 -> ✓
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.88 >= 8.55 -> ✓
2026-03-19 16:52:12 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.89 <= 9.450000000000001 -> ✗
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 37.62 <= 12.15 -> ✗
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.50 <= 9.0 -> ✗
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.85 >= 13.05 -> ✓
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.74 >= 8.55 -> ✓
2026-03-19 16:52:13 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.92 <= 9.450000000000001 -> ✗
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 36.88 <= 12.15 -> ✗
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.60 <= 9.0 -> ✗
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.91 >= 13.05 -> ✓
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.68 >= 8.55 -> ✓
2026-03-19 16:52:15 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.91 <= 9.450000000000001 -> ✗
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 27.70 <= 12.15 -> ✗
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.67 <= 9.0 -> ✗
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.79 >= 13.05 -> ✓
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.68 >= 8.55 -> ✓
2026-03-19 16:52:16 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.92 <= 9.450000000000001 -> ✗
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 17.44 <= 12.15 -> ✗
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 28.95 <= 9.0 -> ✗
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 19.08 >= 13.05 -> ✓
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.69 >= 8.55 -> ✓
2026-03-19 16:52:18 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.92 <= 9.450000000000001 -> ✗
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 17.55 <= 12.15 -> ✗
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.00 <= 9.0 -> ✗
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 19.12 >= 13.05 -> ✓
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.68 >= 8.55 -> ✓
2026-03-19 16:52:19 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.96 <= 9.450000000000001 -> ✗
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 17.20 <= 12.15 -> ✗
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.08 <= 9.0 -> ✗
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 19.13 >= 13.05 -> ✓
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.70 >= 8.55 -> ✓
2026-03-19 16:52:21 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.80 <= 9.450000000000001 -> ✗
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 16.51 <= 12.15 -> ✗
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.13 <= 9.0 -> ✗
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 18.16 >= 13.05 -> ✓
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.71 >= 8.55 -> ✓
2026-03-19 16:52:22 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.78 <= 9.450000000000001 -> ✗
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.77 <= 12.15 -> ✓
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.18 <= 9.0 -> ✗
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 13.90 >= 13.05 -> ✓
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.59 >= 8.55 -> ✓
2026-03-19 16:52:24 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.88 <= 9.450000000000001 -> ✗
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.72 <= 12.15 -> ✓
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.19 <= 9.0 -> ✗
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 14.35 >= 13.05 -> ✓
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.57 >= 8.55 -> ✓
2026-03-19 16:52:25 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 28.97 <= 9.450000000000001 -> ✗
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.56 <= 12.15 -> ✓
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 29.25 <= 9.0 -> ✗
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 14.74 >= 13.05 -> ✓
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 17.51 >= 8.55 -> ✓
2026-03-19 16:52:27 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 24.83 <= 9.450000000000001 -> ✗
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.49 <= 12.15 -> ✓
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 26.03 <= 9.0 -> ✗
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 15.27 >= 13.05 -> ✓
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 16.55 >= 8.55 -> ✓
2026-03-19 16:52:28 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Failed ✗
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 7.82 <= 9.450000000000001 -> ✓
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.35 <= 12.15 -> ✓
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 7.28 <= 9.0 -> ✓
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 16.08 >= 13.05 -> ✓
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 18.48 >= 8.55 -> ✓
2026-03-19 16:52:30 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Success ✓
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Garment type: top-short-sleeve, Thresholds: [9.450000000000001, 12.15, 9.0, 13.05, 8.55]
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[4]) = 6.91 <= 9.450000000000001 -> ✓
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[2], p[3]) = 4.27 <= 12.15 -> ✓
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[1], p[5]) = 4.75 <= 9.0 -> ✓
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[0], p[1]) = 17.07 >= 13.05 -> ✓
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO -   dist(p[4], p[5]) = 19.79 >= 8.55 -> ✓
2026-03-19 16:52:31 - lehome.tasks.bedroom.garment_bi_v2 - INFO - [Success Check] Final result: Success ✓
2026-03-19 16:52:33 - scripts.utils.dataset_replay - INFO -   [Replay 1/1] Success

当前 Replay 能做的                                                                                                                
                                                                                                                                    
可以作为数据增强手段，但能力有限：                                                                                                
                                                                                                                                
1. 多次回放 — --num_replays N 对同一轨迹重复回放 N 次，保存时会产生多份"相同"数据                                                 
2. 筛选成功轨迹 — --save_successful_only 只保留回放成功的，过滤掉失败的                                                           
3. 重新采集观测 — 每次回放时重新 env._get_observations()，意味着相机视角等观测可能有微小物理差异                                  
                                                                                                                                
当前 Replay 不能做的                                                                                                              
                                                                                                                                
没有动作扰动（action perturbation）功能。看 replay_episode 第429行：                                                              
                                                                                                                                
action = episode_data[idx]["action"].to(device).unsqueeze(0)                                                                      
                                                                                                                                
直接原样复用录制的动作，没有任何噪声注入。这意味着：                                                                              
- 无法生成 corner case 数据                                                                                                       
- 多次回放产生的是高度相似的数据，多样性几乎为零                                                                                  
- 物理仿真的随机性带来的差异非常小                                                                                                
                                                                                                                                
如果要真正实现"踩数据"                                                                                                            
                                                                                                                                
需要在动作上注入噪声。核心改动在 replay_episode 函数中，大约需要加这些：                                                          
                                                                                                                                
┌──────────────────────┬───────────────────────────────────────────┬──────────┐                                                   
│       增强方式       │                   说明                    │ 实现难度 │                                                   
├──────────────────────┼───────────────────────────────────────────┼──────────┤                                                   
│ Action 高斯噪声      │ action += N(0, σ) 加关节空间噪声          │ 低       │                                                   
├──────────────────────┼───────────────────────────────────────────┼──────────┤                                                   
│ 随机初始位姿扰动     │ initial_pose 加微小偏移，改变衣物起始位置 │ 低       │                                                   
├──────────────────────┼───────────────────────────────────────────┼──────────┤                                                   
│ 随机衣物姿态         │ reset 时随机旋转/平移衣物                 │ 中       │                                                   
├──────────────────────┼───────────────────────────────────────────┼──────────┤                                                   
│ 随机丢弃帧           │ 随机跳过若干帧，模拟时序抖动              │ 低       │                                                   
├──────────────────────┼───────────────────────────────────────────┼──────────┤                                                   
│ Domain Randomization │ 摄像头噪声、光照变化等                    │ 高       │                                                   
└──────────────────────┴───────────────────────────────────────────┴──────────┘                                                   
                                                                                                                                
最实用且最简单的是前两个：action 噪声 + 初始位姿扰动。要不要我帮你加上 action noise 的功能？加一个 --action_noise_std             
参数即可，改动量很小。

2026-03-19 17:03:36 - replay_standalone - INFO -   Result: Success
2026-03-19 17:03:36 - replay_standalone - INFO - 
2026-03-19 17:03:36 - replay_standalone - INFO - ============================================================
2026-03-19 17:03:36 - replay_standalone - INFO - Replay Summary
2026-03-19 17:03:36 - replay_standalone - INFO - ============================================================
2026-03-19 17:03:36 - replay_standalone - INFO -   Total episodes: 4
2026-03-19 17:03:36 - replay_standalone - INFO -   Successes:      2
2026-03-19 17:03:36 - replay_standalone - INFO -   Success rate:   50.0%
2026-03-19 17:03:36 - replay_standalone - INFO - ============================================================
[145.187s] Simulation App Shutting Down

pixi run dataset-sim replay --dataset_root Datasets/example/four_types_merged --start_episode 0 --end_episode 2 --enable_cameras 2>&1 | tee /tmp/original.log

python -m scripts.replay_lehome_dataset --dataset_root Datasets/example/four_types_merged --start_episode 0 --end_episode 2 --enable_cameras --save_video

sk-kimi-dVVCt5mCQbz7ISKRs5zbNds0oeDwOUgmBk1a5MPJspyTpDcz8QxVj4yBx85UuveH

maozan authored and lck666666 committed 2 minutes ago


python -m scripts.eval --policy_type lerobot --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step80k --dataset_root Datasets/example/four_types_merged --save_datasets --eval_dataset_path Datasets/eval_pi05 --num_episodes 50 --num_envs 1 --max_steps 600 --task_description "fold the garment on the table"