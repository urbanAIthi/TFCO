import importlib
import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from typing import Tuple, Dict, Any

from models import SpacialTemporalDecoder
from train import process_batch
from utils.criterion_utils import CombinedLoss
from utils.dataset_utils import collate_fn, get_dataset
from utils.eval_utils import analyze_results, get_mean_iou, get_num_halocinate, get_unhidden_vehicles
from utils.train_utils import chunked_indices_split, set_seed, transformations

def get_cfg_from_run(trained_model_dir: str, model_name: str) -> Tuple[str, Dict, Dict]:
    model_path = os.path.join(trained_model_dir, 'models', model_name)

    # use importlib to import the config file
    module_model_dir = trained_model_dir.replace('/', '.')
    cfg_module = importlib.import_module(f'{module_model_dir}.config')
    cfg = cfg_module.special_temporal_decoder_new
    network_configs = cfg_module.network_configs

    return model_path, cfg, network_configs

def test_epoch(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, cfg: Dict[str, Any], path: str) -> Tuple[float, float, Dict[str, float]]:
    criterion = CombinedLoss(cfg)
    model.eval()
    running_loss, running_iou, total_samples = 0.0, 0.0, 0
    running_individual_losses = {key: 0.0 for key in criterion.losses.keys()}
    data = {
        'name': [],
        'num_unhidden': [],
        'num_halocinate': [],
        'num_lost': [],
        'num_gt': [],
        'num_current': []
    }

    df = pd.DataFrame(data)
    with torch.no_grad():
        for counter, batch in enumerate(tqdm(val_loader)):
            loss, individual_losses, outputs, biv_target, input_sequence, target_sequence, input_vehicle_dicts, target_vehicle_dicts, names = process_batch(batch, model, criterion, device, cfg["pre_train"], test=True)
            running_loss += loss.item()
            running_iou += get_mean_iou(outputs, biv_target)
            total_samples += 1
            for key, value in individual_losses.items():
                running_individual_losses[key] += value

            for batch_index in range(len(outputs)):
                analyze_results(outputs, target_sequence, input_sequence, counter, batch_index, os.path.join(path, 'results', 'plots'), 'test', save_plot=True, overlay_image_path=None, name=names[batch_index])

            num_unhidden, num_lost = get_unhidden_vehicles(outputs, target_vehicle_dicts, input_vehicle_dicts, names)
            num_halocinate = get_num_halocinate(outputs, target_sequence, names)

            for name, unhidden, halocinate, lost, target_vehicle_dict, input_vehicle_dict in zip(names, num_unhidden, num_halocinate, num_lost, target_vehicle_dicts, input_vehicle_dicts):
                new_row = {
                    'name': name,
                    'num_unhidden': unhidden,
                    'num_halocinate': halocinate,
                    'num_lost': lost,
                    'num_gt': len(target_vehicle_dict),
                    'num_current': len(input_vehicle_dict)
                }
                df = df.append(new_row, ignore_index=True)

            df.to_csv(os.path.join(path, 'results', 'df.csv'), index=False)

    average_running_loss = running_loss / total_samples
    mean_iou = running_iou / total_samples
    for key in running_individual_losses:
        running_individual_losses[key] /= total_samples

    return average_running_loss, mean_iou, running_individual_losses


if __name__ == '__main__':
    trained_model_dir = 'trained_spacial_temporal_decoder/i3040_10pct_6-9h_75m_large_30-11_12-29-59/transformer'
    model_name = 'model_epoch_57.pth'

    # Prepare the run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Set the seed
    set_seed(42)

    # Get configuration from the run
    model_path, cfg, network_configs = get_cfg_from_run(trained_model_dir, model_name)

    # Create the dataset
    dataset = get_dataset(cfg, transformations)
    train_idx, val_idx = chunked_indices_split(
        len(dataset), cfg['split_chunk_size'], only_nonseen=False,
        dataset_path=f'data/{cfg["dataset_name"]}', sequence_len=cfg['sequence_len']
    )
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create DataLoaders
    data_loader_args = {
        'batch_size': cfg['batch_size'],
        'num_workers': 8,
        'collate_fn': collate_fn if not cfg['pre_train'] else None,
        'sampler': None
    }
    train_loader = DataLoader(dataset, **data_loader_args, sampler=train_sampler)
    val_loader = DataLoader(dataset, **data_loader_args, sampler=val_sampler)

    logging.info(f'Dataset length: {len(dataset)} and split into {len(train_idx)} train and {len(val_idx)} val samples with chunk size {cfg["split_chunk_size"]}')

    # Create and load the model
    logging.info('Creating and loading model')
    model = SpacialTemporalDecoder(cfg, network_configs).to(device)
    model.load_state_dict(torch.load(model_path), strict=True)

    # Test the model
    test_epoch(model, val_loader, device, cfg, trained_model_dir)