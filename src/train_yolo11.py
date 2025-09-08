import multiprocessing
from ultralytics import YOLO

def main():
    # Project parameters
    project_name = "runs"
    run_name     = "yolo11_run"
    yaml_file    = "data/test_train/data.yaml"
    weights_path = "weights/yolo11l.pt"
    img_size     = 640  # you can also pull this from a config

    # Initialize YOLO model (will download weights if missing)
    model = YOLO(weights_path)

    # 1) Train
    model.train(
        project = project_name,
        name    = run_name,
        patience= 10,
        data    = yaml_file,
        epochs  = 2,
        batch   = 8,
        imgsz   = img_size,
        iou     = 0.3,
    )

    # 2) Validate on the test split
    model.val(
        data  = yaml_file,
        split = "test",
        imgsz = img_size,
    )

if __name__ == "__main__":
    # On Windows this is required for child process bootstrap
    multiprocessing.freeze_support()

    # Now kick off training + validation
    main()
