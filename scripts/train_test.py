from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train yolo segmentation model")
    parser.add_argument('--data', type=str, required=True, help="file path to yaml file of dataset to train on")
    parser.add_argument('--model', type=str, default="yolov8n-seg", help="model to begin training with")
    parser.add_argument('--batch', type=int, default=128, help="batch size")
    parser.add_argument('--epochs', type=int, default=800, help="number of epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="training images size")
    parser.add_argument('--device', type=list, default=[0], help="specify GPUs")

    args = parser.parse_args()

    model = YOLO(model=f'./models/{args.model}.pt')  # load a pretrained model (recommended for training)
    model.train(
        data=f'./configs/{args.data}.yaml', 
        batch=args.batch, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        device=[dev for dev in args.device]
    )
    model.val()
    model.export()

