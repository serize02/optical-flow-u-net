import torch
import numpy as np
from dataset import get_dataloaders
from model import UNet

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

