import argparse

from .config import RuntimeConfig
from .pipeline import HockeyTrackingPipeline


def main():
    parser = argparse.ArgumentParser(description="Dynamic Ice Hockey Player Tracker")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = RuntimeConfig.from_yaml(args.config)
    pipeline = HockeyTrackingPipeline(cfg)

    for result in pipeline.run(args.video):
        print(f"frame={result.frame_index} tracks={len(result.tracks)}")


if __name__ == "__main__":
    main()
