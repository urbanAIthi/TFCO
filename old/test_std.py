import torch
import os
from typing import List
import wandb
from torch.utils.data import DataLoader
from torch import nn
#from utils.train_utils import LRSchedulerTransformer, LargeSequentialDataset, merge_sequence, unmerge_sequence, analyze_results, apply_image_dropout
from utils.logger_utils import logger


@logger
def test(model: torch.nn.Module, test_IDs: List[str], device: str, filename: str, cfg: dict):
    # create test path
    os.mkdir(os.path.join(filename, 'test'))
    test_path = os.path.join(filename, 'test')

    # prepare the model
    model = model.to(device)
    model.eval()
    criterion = nn.BCELoss()

    test_loss = 0.0
    num_test_samples = 0
    for ID in test_IDs:
        test_set = LargeSequentialDataset(ID)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
        with torch.no_grad():
            for input_sequence, target_sequence, time in test_loader:
                input_sequence = input_sequence.squeeze(dim=2)
                target_sequence = target_sequence.squeeze(dim=2)
                biv_target = target_sequence[:, -1, ...]
                num_test_samples += input_sequence.size(0)
                input_sequence = input_sequence.permute(1, 0, 2, 3, 4)
                input_sequence, biv_target = input_sequence.to(device), biv_target.to(device)
                outputs = model(input_sequence)
                loss = criterion(outputs, biv_target)
                test_loss += loss.item()
                for i in range(input_sequence.size(0)):
                    analyze_results(outputs, target_sequence, input_sequence, time, i, test_path,
                                    'test', test_set.dataset, save_plot = True, save_data = True)
    average_test_loss = test_loss / num_test_samples
    logger.info(f"Test loss: {average_test_loss}")
    wandb.log({'test_loss': average_test_loss})

def test_dropout(model, test_IDs, device, path):
    pass

@logger
def test_prediction(model: torch.nn.Module, test_IDs: List[str], device: str, filename: str, cfg: dict, max_prediction_steps: int):
    # create test path
    os.mkdir(os.path.join(filename, 'test_prediction'))
    for pred in range(max_prediction_steps):
        logger.info(f'running test for prediction {pred + 1}')
        test_path = os.path.join(filename, 'test_prediction', f'prediction_{pred}')
        os.mkdir(test_path)

        # prepare the model
        model = model.to(device)
        model.eval()
        criterion = nn.BCELoss()

        test_loss = 0.0
        num_test_samples = 0
        for ID in test_IDs:
            test_set = LargeSequentialDataset(ID)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
            with torch.no_grad():
                for input_sequence, target_sequence, time in test_loader:
                    input_sequence = create_prediction_input(input_sequence, pred + 1)
                    biv_target = target_sequence[:, -1, ...]
                    num_test_samples += input_sequence.size(0)
                    input_sequence = input_sequence.permute(1, 0, 2, 3, 4)
                    input_sequence, biv_target = input_sequence.to(device), biv_target.to(device)
                    outputs = model(input_sequence)
                    loss = criterion(outputs, biv_target)
                    test_loss += loss.item()
                    for i in range(input_sequence.size(0)):
                        analyze_results(outputs, target_sequence, input_sequence, time, i, test_path,
                                        'test', test_set.dataset, save_plot = True, save_data = True)
        average_test_loss = test_loss / num_test_samples
        logger.info(f"Prediction_{pred + 1}__loss: {average_test_loss}")
        wandb.log({f"Prediction_{pred + 1}_loss": average_test_loss})


def create_prediction_input(sequence: torch.Tensor, prediction_steps: int) -> torch.Tensor:
    zero_tensor = torch.zeros_like(sequence[0][0]).to(sequence.device)
    for b, batch_item in enumerate(sequence):
        # set the last prediction_steps images to zero
        for i in range(1, prediction_steps + 1):
            sequence[b][-i] = zero_tensor
    return sequence


if __name__ == '__main__':
    from solving_occlusion.solving_occlusion.spacial_temporal_decoder import LargeSequentialDataset, analyze_results
    # define tst dataset
    test_path = ''
    test_set = [6]
    dataset_name = "i3040_newvector"

    # load configs and model from test path

