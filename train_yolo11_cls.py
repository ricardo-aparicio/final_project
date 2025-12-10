#!/usr/bin/env python3
from pathlib import Path
from ultralytics import YOLO

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset_split"       


MODEL_WEIGHTS = "yolo11n-cls.pt"


def main():

    model = YOLO(MODEL_WEIGHTS)  

    
    results = model.train(
        data=str(DATA_DIR),      
        epochs=20,              
        imgsz=256,            
        batch=32,            
        lr0=1e-3,                #
        optimizer="adam",        
        patience=0,             #
        device=0,                #
        project="runs_yolo11_cls",
        name="drone_spectrograms_20ep",
        verbose=True,
    )

    print("Directorio de resultados:", results.save_dir)

   
    metrics_test = model.val(
        data=str(DATA_DIR),     
        split="test",            
        imgsz=256,
        batch=64,
    )
    
    print("Test top-1 acc:", metrics_test.top1)
    print("Test top-5 acc:", metrics_test.top5)


    sample_img_dir = ROOT / "autel_noisy01" 
    sample_img = next(sample_img_dir.glob("*.png"))

    preds = model.predict(source=str(sample_img), imgsz=256)
    probs = preds[0].probs  

    print("Imagen de prueba:", sample_img.name)
    print("Probabilidades (orden model.names):", probs.data.cpu().numpy())
    print("Clase predicha:", probs.top1, "->", model.names[int(probs.top1)])


if __name__ == "__main__":
    main()
