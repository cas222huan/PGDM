import argparse
from lib.trainer import ResShiftTrainer, MocolskTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LST downscaling model")
    
    parser.add_argument('--config_base', type=str, default='configs/base.yaml')
    parser.add_argument('--config_var', type=str, default='configs/var.yaml')
    parser.add_argument('--var', type=str)
    parser.add_argument('--type', type=str, default="resshift", choices=["resshift", "mocolsk"])

    args = parser.parse_args()
    return args

def get_loaders_by_task(trainer):
    if trainer.task_type == "groklst":
        return trainer.trainloader_groklst, trainer.valloader_groklst
    if trainer.task_type == "landsat_cn20":
        return trainer.trainloader_landsat_cn20, trainer.valloader_landsat_cn20
    raise ValueError(f"Unsupported task_type: {trainer.task_type}")

TRAINERS = {
    "resshift": ResShiftTrainer,
    "mocolsk": MocolskTrainer,
}

if __name__ == "__main__":
    args = parse_args()
    print(f"Base config: {args.config_base}")
    print(f"Variable config: {args.config_var}")
    print(f"Variable: {args.var}")

    trainer_class = TRAINERS[args.type]
    trainer = trainer_class(config_base_path=args.config_base,
                           config_var_path=args.config_var,
                           var_name=args.var)
    trainloader, valloader = get_loaders_by_task(trainer)
    trainer.train(trainloader, valloader)