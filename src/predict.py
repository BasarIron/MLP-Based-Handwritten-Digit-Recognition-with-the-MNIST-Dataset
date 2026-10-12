import torch
import cv2
import numpy as np
import os
import glob
from src.model import MLP

def predict_full_image(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 150
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]

    bounding_boxes.sort(key=lambda b: b[0])

    print(f"\nFotoğrafta {len(bounding_boxes)} adet rakam tespit edildi. İşleniyor...\n")

    predicted_number_str = ""

    for (x, y, w, h) in bounding_boxes:
        digit_roi = thresh[y:y+h, x:x+w]

        scale = 20.0 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(digit_roi, (new_w, new_h))

        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = img_resized

        canvas_normalized = canvas.astype(np.float32) / 255.0
        canvas_normalized = (canvas_normalized - 0.5) / 0.5
        tensor = torch.tensor(canvas_normalized).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            pred_digit = predicted.item()
            
        predicted_number_str += str(pred_digit)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(img, str(pred_digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    file_name = os.path.basename(image_path)
    save_path = os.path.join(os.path.dirname(image_path), f"analiz_{file_name}")
    cv2.imwrite(save_path, img)

    print(f"Modelin okuduğu tam numara: {predicted_number_str}")
    print(f"Görselleştirilmiş sonuç kaydedildi: {save_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    model_file = os.path.join(project_root, 'mlp_model.pth')
    custom_dir = os.path.join(project_root, 'custom_images')
    
    
    image_paths = [
        os.path.join(custom_dir, f) 
        for f in os.listdir(custom_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('analiz_')
    ]
    
    if image_paths:
        predict_full_image(image_paths[0], model_file)
    else:
        print("Uyarı: 'custom_images' klasöründe analiz edilecek resim bulunamadı!")