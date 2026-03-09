import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher
from datetime import datetime
from pathlib import Path

from .utils import common
from .utils.parser import setup_eval_parser
from .utils.common import launch_app_from_args


def main():
    """Main entry point for evaluation script."""
    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Set up log file name: {model_name}_{garment_type}_{YYYYMMDD}_{HH}.log
    from lehome.utils.logger import set_global_log_file_name, get_logger

    # Extract model name from policy_path (use second-to-last directory name)
    policy_path = Path(args.policy_path)
    model_name = policy_path.parent.name if policy_path.parent.name else policy_path.name
    garment_type = args.garment_type
    datetime_str = datetime.now().strftime("%Y%m%d_%H")
    log_file_name = f"{model_name}_{garment_type}_{datetime_str}.log"
    set_global_log_file_name(log_file_name)

    # Now create logger (will use the global log file name)
    logger = get_logger(__name__)

    simulation_app = launch_app_from_args(args)
    try:
        import lehome.tasks.bedroom
        from .utils.evaluation import eval

        if getattr(args, "headless", False):
            import os

            os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"
        eval(args, simulation_app)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
