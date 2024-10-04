from bionemo.geneformer.run.config_models import ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig
from bionemo.llm.train import train
from bionemo.llm.config.config_models import MainConfig
import argparse
import json
from typing import Optional


def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Geneformer pretraining")
        parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
        parser.add_argument("--model-config-t", default=ExposedFineTuneSeqLenBioBertConfig, required=False, help="fully resolvable python import path to the ModelConfig object.")
        parser.add_argument("--data-config-t", default=GeneformerPretrainingDataConfig, required=False, help="fully resolvable python import path to the ModelConfig object.")
        parser.add_argument("--resume-if-exists", default=True, help="Resume training if a checkpoint exists that matches the current experiment configuration.")
        return parser.parse_args()

    def string_to_class(path: str):
        import importlib
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def load_config(config_path: str, model_config_t: Optional[str], data_config_t: Optional[str]) -> MainConfig:
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # model/data_config_t is used to select the parser dynamically.
        if model_config_t is None:
            # our parser doesnt like literals that are already imported.
            model_config_t = ExposedFineTuneSeqLenBioBertConfig
        elif isinstance(model_config_t, str):
            model_config_t = string_to_class(model_config_t)

        if data_config_t is None:
            data_config_t = GeneformerPretrainingDataConfig
        elif isinstance(data_config_t, str):
            data_config_t = string_to_class(data_config_t)
        
        return MainConfig[model_config_t, data_config_t](**config_dict)

    args = parse_args()
    config = load_config(args.config, args.model_config_t, args.data_config_t)
    # New
    train(
        bionemo_exposed_model_config=config.bionemo_model_config,
        data_config=config.data_config,
        parallel_config=config.parallel_config,
        training_config=config.training_config,
        optim_config=config.optim_config,
        experiment_config=config.experiment_config,
        wandb_config=config.wandb_config,
        resume_if_exists=args.resume_if_exists,
    )

if __name__ == "__main__":
    main()