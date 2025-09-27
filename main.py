import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import get_logger, set_seed_everywhere
import runners


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    set_seed_everywhere(config.seed)

    # get workdir
    config.workdir = os.path.join('run', config.problem.manifold,
                                   config.problem.dataset, 
                                   config.training.algo,
                                   config.save_prefix+f"-{config.seed}-"+config.now)

    # get logger
    log_dir = os.path.join(config.workdir, 'logs')
    logging = get_logger(log_dir)
    logging.info(">" * 100)
    logging.info(">" * 100)
    logging.info(">" * 100)
    logging.info(">" * 100)

    # add device
    if torch.cuda.is_available():
        config.device = 'cuda'
        # torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    else:
        config.device = 'cpu'

    logging.info(f"Found {os.cpu_count()} total number of CPUs.")
    if config.device == torch.device('cuda'):
        logging.info(f"Found {torch.cuda.device_count()} CUDA devices.")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"{props.name} with Memory: {props.total_memory / (1024 ** 3):.2f}GB")
    logging.info(f"Using device: {config.device}")

    yaml_str = OmegaConf.to_yaml(config)
    with open(os.path.join(log_dir, 'config.yml'), 'w') as file:
        file.write(yaml_str)
    logging.info(f"Writing log file to {log_dir}")
    logging.info(">" * 80)
    logging.info(yaml_str)
    logging.info("<" * 80)
    '---------------------------run----------------------------------'
    runner = getattr(runners, config.problem.runner)(config)
    runner.run()


if __name__ == '__main__':
    main()
