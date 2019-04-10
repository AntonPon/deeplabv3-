import click
import json
from pathlib import Path
import torch

@click.command()
@click.option('--config-path', required=True, help='the path to config.json', type=str)
def main(config_path):
    path_to_config = Path(config_path)

    if not (path_to_config.exists()):
        raise ValueError('{} doesn\'t exist'.format(path_to_config))
    elif path_to_config.suffix.lower() != '.json' or not path_to_config.is_file():
        raise ValueError('{} is not .json config file'.format(path_to_config))

    model_configs = load_json(path_to_config)

    path_to_data = Path(model_configs['path_to_data'])
    train_model = model_configs['train_model']
    workers_num = model_configs['workers_num']
    batch_size = model_configs['batch_size']
    data_loaders = get_data_loaders(path_to_data, batch_size, workers_num, train_model)




def get_data_loaders(path_to_data, batch_size=1, workers_num=1, train_model=True):
    train_data_loaders = None
    val_data_loaders = None
    return {'train': train_data_loaders, 'val':val_data_loaders}



def load_json(path_to_json):
    with path_to_json.open() as json_file:
        configs = json.load(json_file)
    return configs


if __name__ == '__main__':
    main()
