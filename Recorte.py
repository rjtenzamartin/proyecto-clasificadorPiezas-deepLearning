import json
import os
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm 

# ================= CONFIGURACIÓN =================

BASE_DIR = 'data'  
IMAGES_DIR = os.path.join(BASE_DIR, 'images') 
OUTPUT_DIR = 'dataset' 
SPLITS = {
    'train': 'mvtec_screws_train.json',
    'validation': 'mvtec_screws_val.json',
    'test': 'mvtec_screws_test.json'
}

def crop_oriented_box(image, annotation):
    """
    Recorta el objeto basándose en la caja orientada.
    Versión compatible con NumPy 1.24+
    """
    if 'bbox' not in annotation:
        return np.array([]) 

    # Extraer datos del formato COCO extendido [row, col, width, height, phi]
    bbox = annotation['bbox']
    
    r = bbox[0]   # row (y)
    c = bbox[1]   # col (x)
    w = bbox[2]   # width
    h = bbox[3]   # height
    phi = bbox[4] # angulo en radianes
    
    # OpenCV espera ((center_x, center_y), (width, height), angle_degrees)
    rect = ((c, r), (w, h), math.degrees(phi))
    
    # Obtener los 4 puntos de la caja rotada
    box = cv2.boxPoints(rect)
    box = np.int64(box) 
    
    # Calcular el bounding box recto (Axis Aligned)
    x_min = max(0, np.min(box[:, 0]))
    x_max = min(image.shape[1], np.max(box[:, 0]))
    y_min = max(0, np.min(box[:, 1]))
    y_max = min(image.shape[0], np.max(box[:, 1]))
    
    if y_min >= y_max or x_min >= x_max:
        return np.array([])

    # Recortar
    crop = image[y_min:y_max, x_min:x_max]
    return crop

def process_split(split_name, json_file):
    json_path = os.path.join(BASE_DIR, json_file)
    
    if not os.path.exists(json_path):
        print(f"No se encontró {json_path}. Saltando...")
        return

    print(f"Procesando {split_name} desde {json_file}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Crear mapa de categorías: ID -> Nombre
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Crear mapa de imágenes: ID -> Nombre de archivo
    images_info = {img['id']: img['file_name'] for img in data['images']}
    
    count = 0
    # Iterar sobre cada anotación (cada tornillo individual)
    for ann in tqdm(data['annotations']):
        image_id = ann['image_id']
        category_id = ann['category_id']
        category_name = categories[category_id]
        
        # Cargar imagen original
        img_filename = images_info[image_id]
        img_path = os.path.join(IMAGES_DIR, img_filename)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
            
        # Recortar la pieza
        crop = crop_oriented_box(image, ann)
        
        if crop.size == 0:
            continue

        save_dir = os.path.join(OUTPUT_DIR, split_name, category_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{ann['id']}.png")
        cv2.imwrite(save_path, crop)
        count += 1
        
    print(f"Guardados {count} recortes para {split_name}.")

# Ejecutar procesamiento
if __name__ == "__main__":
    for split, json_file in SPLITS.items():
        process_split(split, json_file)