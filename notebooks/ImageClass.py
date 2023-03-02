import torch.utils.data as data
from PIL import Image


class CustomImageClass(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.root = data_path
        self.transform = transform
        
    def __getitem__(self, indx):
        image = Image.open(self.root[indx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.root)