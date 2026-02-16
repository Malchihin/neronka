import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):  # 5 классов для указанных знаков
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 3 канала для RGB
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 43)  # количество классов

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def neronka(img):
    model = SimpleCNN()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.RandomInvert(p=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Преобразование изображения в формат PIL
    image = Image.fromarray(img).convert('L')

    # Применение преобразований
    image = transform(image)
    image = image.unsqueeze(0)

    # Предсказание
    with torch.no_grad():
        output = model(image)
        probability, predicted = torch.max(output.data, 1)
        predicted_class = predicted.item()

    print(f"Предсказанный класс: {predicted_class}")
    print(f"Вероятность: {probability % 100}")

    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f'Предсказанный класс: {predicted_class}')
    plt.show()

    return image.squeeze().numpy()
