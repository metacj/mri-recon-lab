import argparse
from pathlib import Path

import yaml


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(description="Test YAML experiment config loading.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("Config loaded successfully.")
    print("Experiment name:", config["experiment_name"])
    print("Dataset:", config["data"]["dataset_name"])
    print("Sampling:", config["sampling"]["type"])
    print("Acceleration:", config["sampling"]["acceleration"])
    print("Operator:", config["operator"]["type"])
    print("Model:", config["model"]["name"])
    print("Phases:", config["model"]["phases"])
    print("Epochs:", config["train"]["epochs"])
    print("Batch size:", config["train"]["batch_size"])


if __name__ == "__main__":
    main()
