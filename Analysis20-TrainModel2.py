#!/usr/bin/env python3

from pathlib import Path
import importlib.util


def load_base_module():
    base_path = Path(__file__).with_name("Analysis20-TrainModel1.py")
    spec = importlib.util.spec_from_file_location("analysis20_train_model_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base module from {base_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_base_module()
    module.configure_model(model_number=2, feature_end=8, output_dir=Path("Model2") / "Training")
    module.main()


if __name__ == "__main__":
    main()
