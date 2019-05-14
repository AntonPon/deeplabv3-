import click
import json
from pathlib import Path
from src.model.deeplab_v3plus import DeepLabV3Plus
from torch import cuda, nn, optim
from tensorboardX import SummaryWriter
from src.train_val import train, val




def get_data_loaders(path_to_data, batch_size=1, workers_num=1, train_model=True):
    train_data_loaders = None
    val_data_loaders = None
    return {'train': train_data_loaders, 'val':val_data_loaders}


def load_json(path_to_json):
    with path_to_json.open() as json_file:
        configs = json.load(json_file)
    return configs


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

    model = DeepLabV3Plus(model_configs['output_classes'])

    device = 'cpu'
    device_count = 0
    if cuda.is_available() and model_configs['cuda_usage']:
        device = 'cuda'
        device_count = cuda.device_count()

    if device is not 'cpu' and device_count > 1:
        model = nn.DataParallel(model).cuda()
    elif device is not 'cpu':
        model = model.cuda()

    criterion = None
    metric = None
    optimizer = optim.SGD(model.parameters(), lr=model_configs['learning_rate'], momentum=0.9)

    info_paths = model_configs['info_paths']

    writer = SummaryWriter(log_dir=info_paths['log_dir'])
    total_epochs = model_configs['epochs']

    for epoch in range(total_epochs):
        model.train()
        train(model, data_loaders['train'], epoch, optimizer, criterion, metric, writer, device=device)
        model.val()
        val(model, criterion, metric, data_loaders['val'], epoch, writer, device=device)




if __name__ == '__main__':
    main()
