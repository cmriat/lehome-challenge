import torch
import numpy as np
from typing import Dict, Any, Optional, Set, Union
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor.core import TransitionKey

from lehome.utils.logger import get_logger
from scripts.pi05_implicit_policy import register_pi05_implicit_policy
from scripts.utils.eval_utils import preprocess_observation
from scripts.utils.garment_latent_utils import (
    ImplicitConditioner,
    load_implicit_conditioning_config,
)
from .base_policy import BasePolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


@PolicyRegistry.register("lerobot")
class LeRobotPolicy(BasePolicy):
    """
    Adapter class for official LeRobot policies (ACT, Diffusion, SmolVLA, etc.).
    
    This class handles:
    1. Loading policy weights and configurations.
    2. Filtering observation keys to match policy requirements.
    3. Preprocessing raw numpy observations into tensors (including image normalization).
    4. Running inference.
    5. Postprocessing actions (un-normalization).
    """

    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str,
        device: str = "cuda",
        use_delta_actions: bool = False,
        delta_stats_path: Optional[str] = None,
    ):
        """
        Initialize the LeRobot policy.

        Args:
            policy_path: Path to the pretrained model checkpoint.
            dataset_root: Path to the dataset root (used for metadata).
            task_description: Text description of the task (for VLA models).
            device: Device to run the model on ('cpu' or 'cuda').
            use_delta_actions: If True, model outputs delta actions that need
                to be converted back to absolute joint positions.
            delta_stats_path: Path to delta_action_stats.json. If None and
                use_delta_actions is True, will look in dataset_root/meta/.
        """
        super().__init__()
        self.device = torch.device(device)
        self.task_description = task_description
        self.use_delta_actions = use_delta_actions
        register_pi05_implicit_policy()
        
        logger.info(f"Loading LeRobot policy from: {policy_path}")
        
        # 1. Load Metadata
        meta = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)
        
        # 2. Load Policy Config
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides={})
        policy_cfg.pretrained_path = policy_path
        self.implicit_conditioning = load_implicit_conditioning_config(policy_cfg)
        
        # 3. Filter Metadata (Logic from original create_il_policy)
        # Identify features required by the policy
        self.input_features: Optional[Set[str]] = None
        if hasattr(policy_cfg, "input_features"):
            self.input_features = set(policy_cfg.input_features.keys())
            train_only_targets = self._get_train_only_observation_targets(policy_cfg)
            self.input_features -= train_only_targets
            self._filter_metadata(meta, self.input_features)

        # 4. Create Policy
        self.policy = make_policy(policy_cfg, ds_meta=meta)
        self.policy.eval()
        self.policy.to(self.device)
        
        # 5. Create Processors
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )

        # 5.1 Delta action support: replace action stats in postprocessor
        if self.use_delta_actions:
            self._setup_delta_actions(dataset_root, delta_stats_path)

        # 6. Infer Action Dimension (Logic from original run_evaluation_loop)
        self.action_dim = self._infer_action_dim(meta, task_description)
        logger.info(
            f"LeRobotPolicy initialized. Action dim: {self.action_dim}, "
            f"delta_actions: {self.use_delta_actions}"
        )

    def reset(self):
        """Reset the internal state of the policy."""
        self.policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate action from observation.

        Args:
            observation: Dictionary of numpy arrays (raw environment output).

        Returns:
            action: Numpy array of action values (un-normalized, absolute joint positions).
        """
        # Save current state BEFORE filtering for debug and optional delta→absolute conversion
        current_state = observation.get("observation.state")

        # 1. Filter observations (keep only what the policy needs)
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)

        # 2. Preprocess (Numpy -> Tensor Batch, Normalize, etc.)
        batch_obs = self._process_observation(observation)

        # 3. Inference
        with torch.inference_mode():
            batch_action = self.policy.select_action(batch_obs)

        # 4. Postprocess (Un-normalize)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        # 5. Convert to Numpy (Remove batch dimension)
        action_np = batch_action.squeeze(0).cpu().numpy()

        # 6. Delta-to-absolute conversion
        if self.use_delta_actions and current_state is not None:
            action_np = action_np + current_state

        return action_np

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    def _get_train_only_observation_targets(self, policy_cfg: PreTrainedConfig) -> Set[str]:
        """Return observation keys used only as training supervision targets."""
        targets: Set[str] = set()
        type_target_key = getattr(policy_cfg, "type_target_key", None)
        if isinstance(type_target_key, str) and type_target_key.startswith("observation."):
            targets.add(type_target_key)
        return targets

    def _setup_delta_actions(self, dataset_root: str, delta_stats_path: Optional[str]) -> None:
        """Load delta stats and replace action stats in the postprocessor."""
        from pathlib import Path
        from scripts.utils.delta_action_stats import load_delta_action_stats
        from scripts.utils.delta_action_processor import replace_action_stats_in_processor

        # Find delta stats file
        if delta_stats_path and Path(delta_stats_path).exists():
            stats_path = Path(delta_stats_path)
        else:
            stats_path = Path(dataset_root) / "meta" / "delta_action_stats.json"

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Delta action stats not found at {stats_path}. "
                f"Run delta-stats computation first."
            )

        delta_stats = load_delta_action_stats(stats_path)
        logger.info(f"Loaded delta action stats from {stats_path}")

        # Replace action stats in postprocessor so unnormalization uses delta range
        if self.postprocessor:
            replace_action_stats_in_processor(self.postprocessor, delta_stats)
            logger.info("Replaced action stats in postprocessor with delta stats")

    def _filter_metadata(self, meta: LeRobotDatasetMetadata, expected_keys: Set[str]):
        """Remove extra features from metadata that are not required by the policy."""
        dataset_features = set(meta.features.keys())
        
        # Find features in dataset but not needed by policy
        extra_features = dataset_features - expected_keys
        
        # Filter out system features (these are OK to have extra)
        system_features = {
            "timestamp", "frame_index", "episode_index", 
            "index", "task_index", "next.done"
        }
        extra_features = extra_features - system_features

        # Remove extra observation features from metadata to prevent validation errors
        for feature in extra_features:
            if feature.startswith("observation."):
                del meta.features[feature]

    def _infer_action_dim(self, meta: LeRobotDatasetMetadata, task_description: str) -> int:
        """Infer action dimension from metadata or task description."""
        action_dim = None

        # Try metadata 'action' shape
        if meta and hasattr(meta, "features") and "action" in meta.features:
            action_shape = meta.features["action"].get("shape", [])
            if action_shape and len(action_shape) > 0:
                action_dim = action_shape[0]

        # Try metadata 'observation.state' shape (fallback)
        if (
            action_dim is None
            and meta
            and hasattr(meta, "features")
            and "observation.state" in meta.features
        ):
            state_shape = meta.features["observation.state"].get("shape", [])
            if state_shape and len(state_shape) > 0:
                action_dim = state_shape[0]
                
        # Final fallback based on task name (heuristic)
        if action_dim is None:
            if "Bi" in task_description or "bi" in task_description.lower():
                action_dim = 12  # Dual-arm
            else:
                action_dim = 6   # Single-arm
        
        return action_dim

    def _filter_observations(self, obs_dict: Dict[str, Any], policy_input_features: Set[str]) -> Dict[str, Any]:
        """Filter observation dictionary to only include features expected by policy."""
        filtered = {}
        required_keys = set(policy_input_features)
        if self.implicit_conditioning.enabled:
            required_keys.add(self.implicit_conditioning.source_image_key)
            required_keys.add(self.implicit_conditioning.target_state_key)
        missing_required = [key for key in required_keys if key.startswith("observation.") and key not in obs_dict]
        if missing_required:
            raise KeyError(
                "Missing required observation keys for policy input: " + ", ".join(sorted(missing_required))
            )
        for key, value in obs_dict.items():
            if not key.startswith("observation."):
                filtered[key] = value
            elif key in required_keys:
                filtered[key] = value
        return filtered

    def _prepare_for_preprocessor(self, observation_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation dictionary for LeRobot preprocessor pipeline."""
        obs_for_preproc = {}
        for key, value in observation_dict.items():
            if not key.startswith("observation."):
                continue

            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                    # (H, W, C) -> (C, H, W), [0, 1] normalization
                    value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
                else:
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
            else:
                obs_for_preproc[key] = value

        # Create transition format with complementary_data for VLA models
        dummy_action = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }
        return transition

    def _process_observation(self, observation_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Process observation using the LeRobot preprocessor or manual fallback."""
        if self.preprocessor is not None:
            transition = self._prepare_for_preprocessor(observation_dict)
            transformed_transition = self.preprocessor._forward(transition)
            return self.preprocessor.to_output(transformed_transition)
        else:
            # Fallback to manual preprocessing (moved to utils in Step A)
            return preprocess_observation(
                observation_dict, self.device, self.task_description
            )