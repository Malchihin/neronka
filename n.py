import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(ImprovedCNN, self).__init__()
        # Match the layer names from the saved model
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Input: 32x32 -> after 3 pooling layers: 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def predict_image(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN(num_classes=16)
    
    # Load with map_location to handle CPU/GPU differences
    state_dict = torch.load('model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Convert frame to PIL Image
    image = Image.fromarray(img).convert('L')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        probability, predicted = torch.max(probabilities, 1)
        
        predicted = predicted.item()
        probability = probability.item()

    print(f"Предсказанный класс: {predicted}")
    print(f"Вероятность: {probability * 100:.2f}%")

    return predicted, probability

# Rest of your code remains the same...
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predicted_class, probability = predict_image(gray_frame)

    cv2.putText(frame, f'Class: {predicted_class}, Prob: {probability * 100:.2f}%', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()