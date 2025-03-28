import torch
from torch.utils.data import DataLoader
from torchvision.datasets import EuroSAT
from torchvision.transforms import transforms
from model import get_model
from config import Config
import matplotlib.pyplot as plt
import numpy as np

def test():
    # Load config and model
    config = Config()
    model = get_model(config).to(config.device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Prepare test data
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EuroSAT(
        root=config.data_root, 
        transform=transform,
        download=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # Test loop
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Print results
    print(f'Overall Accuracy: {100 * correct / total:.2f}%')
    print('\nClass-wise Accuracy:')
    for i in range(10):
        print(f'Class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')

    # Visualize some predictions
    visualize_predictions(model, test_dataset, config.device)

def visualize_predictions(model, dataset, device, num_images=5):
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
               'Highway', 'Industrial', 'Pasture', 
               'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    indices = np.random.choice(len(dataset), num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
        
        # Convert image back for display
        image = image.squeeze().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'True: {classes[label]}\nPred: {classes[predicted.item()]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()
