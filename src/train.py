import torch
from torch import optim
from config import MODEL_SAVE_DIR, LEARNING_RATE, NUM_EPOCHS
from model import UNet
from dataset import get_dataloaders
from utils import dice_loss, bce_dice_loss

def train():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader, _ = get_dataloaders()

    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += bce_dice_loss(outputs, masks).item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_unet.pth"))

if __name__ == "__main__":
    train()