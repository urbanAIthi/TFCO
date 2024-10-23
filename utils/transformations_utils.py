from torchvision import transforms

def get_transformations(config):
    image_size = config['image_size']
    transformations = transforms.Compose([
                                            transforms.Resize((image_size,image_size)),
                                            transforms.ToTensor(),  # Convert the PIL Image to a PyTorch tensor
                                        ])
    
    rotation_augmentation = config['image_rotation']
    shift_augmentation = config['image_shift']

    if rotation_augmentation:
        pass

    return transformations



class RandomNormalRotation:
    def __init__(self, angle):
        self.angle = angle
        
    def __call__(self, img):
        return transforms.functional.rotate(img, self.angle)
    

