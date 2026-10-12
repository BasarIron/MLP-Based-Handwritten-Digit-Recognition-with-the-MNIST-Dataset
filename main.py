import torch
from src.dataset import get_dataloaders
from src.model import MLP
from src.train import train_model, plot_metrics

def evaluate_test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan donanım: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    model = MLP().to(device)

    train_losses, val_losses, train_acc, val_acc = train_model(
        model, train_loader, val_loader, epochs=10, device=device
    )

    plot_metrics(train_losses, val_losses, train_acc, val_acc)

    evaluate_test(model, test_loader, device)
    
    torch.save(model.state_dict(), 'mlp_model.pth')