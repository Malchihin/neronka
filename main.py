import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class RoadSignClassifier:
    """Полный пайплайн для классификации дорожных знаков"""
    
    def __init__(self, data_dir='my_road_signs_dataset'):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Названия классов
        self.class_names = ['прямо', 'направо', 'налево', 'кирпич']
        
        # Трансформации
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }
        
        print(f"Using device: {self.device}")
    
    def load_datasets(self):
        """Загрузка датасетов"""
        print("Loading datasets...")
        
        train_dataset = SimpleImageDataset(
            self.data_dir / 'train',
            transform=self.transform['train']
        )
        
        val_dataset = SimpleImageDataset(
            self.data_dir / 'val',
            transform=self.transform['val']
        )
        
        test_dataset = SimpleImageDataset(
            self.data_dir / 'test',
            transform=self.transform['val']
        )
        
        print(f"Train: {len(train_dataset)} images")
        print(f"Val: {len(val_dataset)} images")
        print(f"Test: {len(test_dataset)} images")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self, num_classes=4):
        """Создание модели"""
        # Используем предобученную модель (transfer learning)
        model = models.resnet18(pretrained=True)
        
        # Заменяем последний слой
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        model = model.to(self.device)
        return model
    
    def train(self, epochs=10, batch_size=32):
        """Обучение модели"""
        print("Starting training...")
        
        # Загрузка данных
        train_dataset, val_dataset, _ = self.load_datasets()
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        # Создание модели
        model = self.create_model()
        
        # Функция потерь и оптимизатор
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # История обучения
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
            
            # Валидация
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Статистика эпохи
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = 100 * correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Acc: {val_acc:.2f}%')
            
            scheduler.step()
        
        # Сохранение модели
        torch.save(model.state_dict(), 'road_sign_model.pth')
        print("Model saved to road_sign_model.pth")
        
        return model, history
    
    def test(self, model_path='road_sign_model.pth'):
        """Тестирование модели"""
        print("Testing model...")
        
        _, _, test_dataset = self.load_datasets()
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Загрузка модели
        model = self.create_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Тестирование
        correct = 0
        total = 0
        class_correct = [0] * 4
        class_total = [0] * 4
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # По классам
                for i in range(labels.size(0)):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
        
        # Результаты
        print(f"\nTest Accuracy: {100 * correct / total:.2f}%")
        print("\nAccuracy by class:")
        for i in range(4):
            if class_total[i] > 0:
                print(f"  {self.class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
    
    def predict_image(self, image_path, model_path='road_sign_model.pth'):
        """Предсказание для одного изображения"""
        # Загрузка модели
        model = self.create_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Загрузка и преобразование изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform['val'](image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
        
        # Результат
        predicted_class = self.class_names[predicted.item()]
        
        print(f"\nPrediction for {image_path}:")
        print(f"  Class: {predicted_class}")
        print("\nProbabilities:")
        for i, prob in enumerate(probabilities.cpu().numpy()):
            print(f"  {self.class_names[i]}: {prob*100:.2f}%")
        
        return predicted_class

class SimpleImageDataset(Dataset):
    """Простой Dataset для загрузки изображений"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Собираем все изображения
        self.samples = []
        self.class_to_idx = {}
        
        # Ищем папки с классами
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            
            # Собираем изображения в папке
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_file), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Если ошибка загрузки, создаем черное изображение
            image = Image.new('RGB', (128, 128), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 4. ПОЛНЫЙ ПАЙПЛАЙН - ОТ СОЗДАНИЯ ДАТАСЕТА ДО ОБУЧЕНИЯ

def complete_pipeline():
    """Полный пайплайн: создание датасета + обучение"""
    
    print("="*60)
    print("ПОЛНЫЙ ПАЙПЛАЙН СОЗДАНИЯ И ОБУЧЕНИЯ МОДЕЛИ")
    print("="*60)
    
    # Шаг 1: Создаем датасет
    print("\n1. СОЗДАНИЕ СИНТЕТИЧЕСКОГО ДАТАСЕТА...")
    creator = QuickSignDatasetCreator()
    dataset_path = creator.generate_dataset(
        images_per_class=200,  # 200 на класс = 800 всего
        output_dir='road_signs_dataset'
    )
    
    # Шаг 2: Обучаем модель
    print("\n2. ОБУЧЕНИЕ МОДЕЛИ...")
    classifier = RoadSignClassifier(data_dir=dataset_path)
    model, history = classifier.train(epochs=10, batch_size=32)
    
    # Шаг 3: Тестируем
    print("\n3. ТЕСТИРОВАНИЕ МОДЕЛИ...")
    classifier.test()
    
    # Шаг 4: Пример предсказания
    print("\n4. ПРИМЕР ПРЕДСКАЗАНИЯ...")
    
    # Создаем тестовое изображение
    test_img = creator.create_synthetic_sign('forward', size=128)
    cv2.imwrite('test_sign.png', test_img)
    
    # Предсказываем
    result = classifier.predict_image('test_sign.png')
    
    print("\n" + "="*60)
    print("ПАЙПЛАЙН ЗАВЕРШЕН!")
    print(f"Модель обучена и сохранена как 'road_sign_model.pth'")
    print(f"Датасет создан в папке '{dataset_path}'")
    print("="*60)

# 5. ЗАПУСК ВСЕГО ПАЙПЛАЙНА

if __name__ == "__main__":
    # Вариант 1: Быстро создать датасет
    # create_quick_dataset()
    
    # Вариант 2: Запустить полный пайплайн
    complete_pipeline()
    
    # Вариант 3: Если у вас уже есть фотографии:
    # 1. Создайте папку 'my_photos'
    # 2. Внутри создайте 4 папки: forward, right, left, no_entry
    # 3. Разложите фото по папкам
    # 4. Запустите:
    #    classifier = RoadSignClassifier(data_dir='my_photos')
    #    classifier.train()