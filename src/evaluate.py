import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import UNet

def visualize_predictions(model, dataloader, device, num_samples=2, threshold=0.5):
    
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        preds = model(images)
        preds_bin = (preds > threshold).float()
    
    images_np = images.cpu().numpy().squeeze()
    masks_np = masks.cpu().numpy().squeeze()
    preds_np = preds.cpu().numpy().squeeze()
    preds_bin_np = preds_bin.cpu().numpy().squeeze()
    
    plt.figure(figsize=(15, 6 * num_samples))
    for i in range(num_samples):
        
        dice = (2 * (masks_np[i] * preds_bin_np[i]).sum()) / (masks_np[i].sum() + preds_bin_np[i].sum() + 1e-5)
        iou = (masks_np[i] * preds_bin_np[i]).sum() / (masks_np[i] + preds_bin_np[i] - (masks_np[i] * preds_bin_np[i])).sum()
        
        plt.subplot(num_samples, 4, (i*4)+1)
        plt.imshow(images_np[i], cmap='gray')
        plt.title(f'Imagen Original\n{images_np[i].shape}')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, (i*4)+2)
        plt.imshow(masks_np[i], cmap='gray')
        plt.title(f'Máscara Real\nDice: {dice:.2f}, IoU: {iou:.2f}')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, (i*4)+3)
        plt.imshow(preds_bin_np[i], cmap='gray')
        plt.title('Predicción Binarizada')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, (i*4)+4)
        plt.imshow(images_np[i], cmap='gray')
        plt.imshow(preds_bin_np[i], alpha=0.3, cmap='jet')
        plt.title('Superposición Imagen-Predicción')
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
            
            # Calcular Dice
            intersection = (preds * masks).sum()
            dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-5)
            dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


if __name__ == '__main__':

    model = UNet()
    model.load_state_dict(torch.load('artifacts/models/best_unet.pth', map_location=torch.device('cpu')))

    _, __, test_loader = get_dataloaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dice_test = evaluate_model(model, test_loader, device)
    print(f"Dice Score en test: {dice_test:.4f}")

    visualize_predictions(model, test_loader, device)
