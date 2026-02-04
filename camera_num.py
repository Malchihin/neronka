import cv2
from neronka_lib import neronka

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Преобразование кадра в оттенки серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обработка кадра с использованием библиотеки neronka
    processed_image = neronka(gray_frame)

    # Отображение результата
    cv2.imshow('Processed Image', processed_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
