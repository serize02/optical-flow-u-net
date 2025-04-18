import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import UNet

def visualize_predictions(model, dataloader, device, num_samples=1, threshold=0.5):
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        preds = model(images)
        preds_bin = (preds > threshold).float()
    
    images_np = images.cpu().numpy().squeeze()
    masks_np = masks.cpu().numpy().squeeze()
    preds_bin_np = preds_bin.cpu().numpy().squeeze()
    
    plt.figure(figsize=(20, 5 * num_samples))
    
    for i in range(num_samples):
        dice = 2 * (masks_np[i] * preds_bin_np[i]).sum() / (masks_np[i].sum() + preds_bin_np[i].sum() + 1e-5)
        iou = (masks_np[i] * preds_bin_np[i]).sum() / (masks_np[i].sum() + preds_bin_np[i].sum() - (masks_np[i] * preds_bin_np[i]).sum() + 1e-5)
    
        plt.subplot(num_samples, 4, (i*4)+1)
        plt.imshow(images_np[i], cmap='gray')
        plt.title(f'Original Image\nShape: {images_np[i].shape}')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, (i*4)+2)
        plt.imshow(images_np[i], cmap='gray')
        plt.imshow(np.ma.masked_where(masks_np[i] == 0, masks_np[i]), 
                   cmap='Reds', 
                   alpha=0.5, 
                   vmin=0, 
                   vmax=1)
        plt.title('Ground Truth Mask (Red)')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, (i*4)+3)
        plt.imshow(images_np[i], cmap='gray')
        plt.imshow(np.ma.masked_where(preds_bin_np[i] == 0, preds_bin_np[i]), 
                   cmap='Blues', 
                   alpha=0.5, 
                   vmin=0, 
                   vmax=1)
        plt.title('Predicted Mask (Blue)')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, (i*4)+4)
        plt.imshow(images_np[i], cmap='gray')
        plt.imshow(np.ma.masked_where(masks_np[i] == 0, masks_np[i]), 
                   cmap='Reds', 
                   alpha=0.3, 
                   vmin=0, 
                   vmax=1)
        plt.imshow(np.ma.masked_where(preds_bin_np[i] == 0, preds_bin_np[i]), 
                   cmap='Blues', 
                   alpha=0.3, 
                   vmin=0, 
                   vmax=1)
        plt.title(f'Combined Masks\nDice: {dice:.2f}, IoU: {iou:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            intersection = (preds * masks).sum()
            dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-5)
            dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


if __name__ == '__main__':

    model = UNet()
    model.load_state_dict(torch.load('u-net/models/best_unet.pth', map_location=torch.device('cpu')))

    _, __, test_loader = get_dataloaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dice_test = evaluate_model(model, test_loader, device)
    print(f"Dice Score en test: {dice_test:.4f}")

    visualize_predictions(model, test_loader, device)
