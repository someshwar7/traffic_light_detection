from ultralytics import YOLO

def train_yolov8():
    # Load a pretrained model
    model = YOLO('yolov8n.pt')  # Using nano version, you can change to s/m/l/x

    # Train the model with your specific paths
    results = model.train(
        data='C:/Users/somes/Desktop/folder/data.yaml',  # Your YAML file path
        epochs=30,                    # Number of training epochs
        imgsz=640,                     # Image size
        batch=16,                      # Batch size
        name='harshu_yolov8',          # Custom name for this run
        patience=50,                   # Early stopping patience
        device='cpu',                  # Changed to CPU since no GPU is available
        workers=4,                     # Number of data loading workers
        optimizer='Adam',              # Optimizer
        lr0=0.001,                     # Initial learning rate
        momentum=0.937,                # SGD momentum/Adam beta1
        weight_decay=0.0005,           # Optimizer weight decay
        project='runs/train',          # Save directory
        exist_ok=True,                 # Allow overwriting existing runs
        pretrained=True,               # Use pretrained weights
        verbose=True,                  # Print detailed training info
        save=True,                     # Explicitly enable saving
        save_period=10                 # Save weights every 10 epochs
    )

    # Validate the model
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50:95: {metrics.box.map}")

    # Get the path to the best weights
    weights_path = f"runs/train/som_yolov8/weights/best.pt"
    print(f"Best weights saved at: {weights_path}")

    # Export the model to ONNX format (optional)
    onnx_path = model.export(format='onnx')
    print(f"Model exported to ONNX at: {onnx_path}")

    # Optionally save an additional copy with a custom name
    model.save('C:/Users/somes/Desktop/folder/yolov8_final.pt')

if __name__ == '__main__':
    train_yolov8()