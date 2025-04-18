import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE

class EchoDataset(Dataset):
    def __init__(self, split="train", augment=False):
        self.split = split
        self.augment = augment
        
        # Define paths based on split
        if split == "train":
            self.root_dir = TRAIN_DIR
        elif split == "val":
            self.root_dir = VAL_DIR
        elif split == "test":
            self.root_dir = TEST_DIR
        else:
            raise ValueError("Invalid split! Use 'train', 'val', or 'test'.")

        self.image_dir = os.path.join(self.root_dir, "images")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.image_files = sorted(os.listdir(self.image_dir))

        # Define transforms
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace(".png", "_mask.png")

        # Load image and mask
        image = cv2.imread(os.path.join(self.image_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.augment:
            image = self.train_transforms(image)
        else:
            image = self.val_transforms(image)

        # Convert mask to tensor
        mask = torch.from_numpy(mask / 255.0).unsqueeze(0).float()

        return image, mask

def get_dataloaders():
    train_dataset = EchoDataset(split="train", augment=True)
    val_dataset = EchoDataset(split="val", augment=False)
    test_dataset = EchoDataset(split="test", augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader