import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import UNet


def visualize_predictions(model, dataloader, device, threshold=0.5):
    model.eval()
    
    dataset = dataloader.dataset
    random_idx = np.random.randint(0, len(dataset))
    image, mask = dataset[random_idx]
    
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_tensor)
        pred_bin = (pred > threshold).float()
    
    image_np = image.numpy().squeeze().astype(np.float32)
    mask_np = mask.numpy().squeeze().astype(np.uint8)
    pred_np = pred_bin.cpu().numpy().squeeze().astype(np.uint8)
    
    dice = 2 * (mask_np * pred_np).sum() / (mask_np.sum() + pred_np.sum() + 1e-5)
    iou = (mask_np * pred_np).sum() / (mask_np.sum() + pred_np.sum() - (mask_np * pred_np).sum() + 1e-5)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title('Imagen Original\nShape: {}'.format(image_np.shape))
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(image_np, cmap='gray')
    plt.imshow(mask_np, alpha=0.5, cmap='Reds')
    plt.title('Máscara Real (Rojo)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(image_np, cmap='gray')
    plt.imshow(pred_np, alpha=0.5, cmap='Blues')
    plt.title('Predicción Modelo (Azul)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(image_np, cmap='gray')
    plt.imshow(mask_np, alpha=0.3, cmap='Reds')
    plt.imshow(pred_np, alpha=0.3, cmap='Blues')
    plt.title(f'Superposición\nDice: {dice:.2f}, IoU: {iou:.2f}')
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

    # dice_test = evaluate_model(model, test_loader, device)
    # print(f"Dice Score en test: {dice_test:.4f}")

    visualize_predictions(model, test_loader, device)

