import torch.nn as nn 
import torch

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super(CombinedLoss, self).__init__()

        if 'types' not in config['criterion']:
            raise ValueError("Loss type not specified in the configuration")

        assert len(config['criterion']['types']) == len(config['criterion']['types_weight']), "The length of the types and types_weight must be equal"
        assert sum(config['criterion']['types_weight']) == 1, "The sum of the types_weight must be equal to 1"

        self.losses = nn.ModuleDict()
        self.weights = {}

        for loss_name, loss_weight in zip(config['criterion']['types'], config['criterion']['types_weight']):
            if loss_name == 'BCE':
                bce_weight = config['criterion'].get('BCE_weight', 1)
                pos_weight = torch.tensor([1.0 / bce_weight]).cuda()
                self.losses[loss_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif loss_name == 'IOU':
                self.losses[loss_name] = IoULoss()  # Assuming IoULoss is defined elsewhere
            else:
                raise ValueError(f"Loss type {loss_name} not supported")

            self.weights[loss_name] = loss_weight

    def forward(self, input, target):
        total_loss = 0.0
        individual_losses = {}
        for loss_name, loss_fn in self.losses.items():
            loss_weight = self.weights[loss_name]
            total_loss += loss_weight * loss_fn(input, target)
            individual_losses[loss_name] = loss_fn(input, target).item()
        return total_loss, individual_losses



def get_criterion(config):
    if 'types' not in config['criterion']:
        raise ValueError("Loss type not specified in the configuration")
    
    # check if types len is equal to the types_weight len
    assert len(config['criterion']['types']) == len(config['criterion']['types_weight']), "The length of the types and types_weight must be equal"
    # check if the sum of the types_weight is equal to 1
    assert sum(config['criterion']['types_weight']) == 1, "The sum of the types_weight must be equal to 1"


    
    loss_type = config['criterion']['type']

    # Handle BCELoss specifically
    if loss_type == 'BCELoss':
        weight = config['criterion'].get('weight', None)
        
        # If weight is provided and it's 'BCELoss', use BCEWithLogitsLoss with pos_weight
        if weight is not None:
            pos_weight = torch.tensor([1.0 / weight]).cuda()
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCELoss()
    
    if loss_type == 'IOU':
        return IoULoss()

    else:
        raise ValueError(f"Loss type {loss_type} not supported")


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets):
        outputs = self.sigmoid(outputs)

        # invert the gt maks and outputs
        targets = 1 - targets
        outputs = 1 - outputs

        # Intersection
        intersection = torch.sum(outputs * targets, dim=[1, 2, 3])
        
        # Union
        total = torch.sum(outputs + targets, dim=[1, 2, 3])
        union = total - intersection

        # IoU
        iou = intersection / (union + 1e-10)

        # clamp values to (0, 1)
        iou = torch.clamp(iou, 0.0, 1.0)

        # Loss
        loss = 1.0 - iou

        # Mean IoU Loss over the batch
        return loss.mean()