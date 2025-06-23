# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

# Standard library imports for file operations and system functions
import argparse           # For parsing command-line arguments
import csv                # For saving results in CSV format
import os                 # For operating system dependent functionality
import platform           # For getting information about the platform
import sys                # For system-specific parameters and functions
from pathlib import Path  # Object-oriented filesystem paths
import torch              # PyTorch import - the deep learning framework used by YOLOv5
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np

# Start Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("video")
picam2.start()

# Setting up path to ensure imports work correctly
FILE = Path(__file__).resolve()         # Get the absolute path of the current file
ROOT = FILE.parents[0]                  # YOLOv5 root directory (parent directory of this file)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))          # Add ROOT to Python path if it's not already there
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Get the relative path from current working directory

# Import plotting utilities for visualization
from ultralytics.utils.plotting import Annotator, colors, save_one_box

# Import the core YOLOv5 components
from models.common import DetectMultiBackend  # For loading different model formats
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # Data loading utilities
from utils.general import (  # General utility functions
    LOGGER,             # Logger for printing status messages
    Profile,            # For timing code execution
    check_file,         # Verifies file existence
    check_img_size,     # Makes sure image size is valid for the model
    check_imshow,       # Checks if OpenCV's imshow will work
    check_requirements,  # Verifies required packages are installed
    colorstr,           # Colorizes terminal text
    cv2,                # OpenCV for image processing
    increment_path,     # Creates unique numbered run folders
    non_max_suppression,  # Removes overlapping bounding boxes
    print_args,         # Prints function arguments
    scale_boxes,        # Scales bounding boxes
    strip_optimizer,    # Removes training info from models
    xyxy2xywh,          # Converts box format from [x1,y1,x2,y2] to [x,y,width,height]
)
from utils.torch_utils import select_device, smart_inference_mode  # PyTorch specific utilities


# Main function for running detection with the decorator for efficient inference
@smart_inference_mode()             # This decorator optimizes inference by disabling gradients
def run(
    weights=ROOT / "PillV5n.pt",     # Default model: YOLOv5 small
    source=ROOT / "data/images",        # Default source: images in data directory
    data=ROOT / "data/Pill.yaml",       # Default dataset config: COCO128
    imgsz=(640, 640),                   # Default image size: 640x640
    conf_thres=0.2,                    # Confidence threshold: only show detections above this confidence
    iou_thres=0.45,                     # IoU threshold: used for non-maximum suppression to remove overlapping boxes
    max_det=1000,                       # Maximum detections per image
    device="",                          # Default device: will auto-select GPU if available, otherwise CPU
    view_img=False,                     # Whether to display results on screen
    save_txt=False,                     # Whether to save results as text files
    save_format=0,                      # Format for saved coordinates (0=YOLO, 1=Pascal VOC)
    save_csv=False,                     # Whether to save results as CSV
    save_conf=False,                    # Whether to include confidence in text files
    save_crop=False,                    # Whether to save cropped detection images
    nosave=True,                        # Don't save images/videos - CHANGED DEFAULT TO TRUE
    classes=None,                       # Filter by class (e.g., only detect persons)
    agnostic_nms=False,                 # Class-agnostic NMS (don't consider classes during NMS)
    augment=False,                      # Augmented inference (test-time augmentation)
    visualize=False,                    # Visualize model features
    update=False,                       # Update all models (strip optimizer)
    project=ROOT / "runs/detect",       # Save results directory
    name="exp",                         # Experiment name
    exist_ok=False,                     # Whether to overwrite existing experiment
    line_thickness=3,                   # Bounding box thickness
    hide_labels=False,                  # Hide labels
    hide_conf=False,                    # Hide confidences
    half=False,                         # Use FP16 half-precision inference (faster but slightly less accurate)
    dnn=False,                          # Use OpenCV DNN for ONNX inference
    vid_stride=1,                       # Video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.
    """
    # Convert source to string for easier handling
    source = str(source)
    
    # Determine whether to save images based on inputs
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    
    # Check if source is a file, URL, webcam, or screenshot
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    
    # Download file if it's a URL
    if is_url and is_file:
        source = check_file(source)  # download

    # Set up directories for saving results ONLY if we need to save something
    # This is the key change to prevent creating directories when nosave is True
    save_dir = Path("/tmp")  # Default to tmp as fallback
    if not nosave and (save_txt or save_img or save_csv or save_crop):
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    # select_device will use GPU if available, otherwise CPU
    device = select_device(device)
    
    # DetectMultiBackend handles loading models from various formats (PyTorch, ONNX, TensorFlow, etc.)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    
    # Ensure image size is valid (must be divisible by the model's stride)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Set up dataloader based on the source type
    bs = 1  # batch_size (default)
    if webcam:
        # For webcam, use LoadStreams which can handle multiple concurrent streams
        frame = picam2.capture_array()
        view_img = check_imshow(warn=True)
        dataset = frame
        bs = len(dataset)  # batch size = number of streams
    elif screenshot:
        # For screen capture, use LoadScreenshots
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # For files, directories, etc., use LoadImages
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    
    # Initialize variables for video writing
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Warm up the model (run a dummy inference to initialize it)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    
    # Initialize counters and timers
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    
    # Define the CSV path only if we're saving CSVs
    csv_path = save_dir / "predictions.csv" if save_csv else None

    # Helper function to write detection results to CSV
    def write_to_csv(image_name, prediction, confidence):
        """Writes prediction data for an image to a CSV file, appending if the file exists."""
        if not save_csv:  # Skip if we're not saving CSVs
            return
            
        data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write header row if file is new
            writer.writerow(data)  # Write the detection data
    
    # Main loop over images/frames
    for path, im, im0s, vid_cap, s in dataset:
        # Pre-processing time measurement
        with dt[0]:
            # Load and preprocess the image
            im = torch.from_numpy(im).to(model.device)  # Move image to device (GPU/CPU)
            im = im.half() if model.fp16 else im.float()  # Convert to FP16 if using half precision
            im /= 255  # Normalize pixel values to 0-1
            if len(im.shape) == 3:
                im = im[None]  # Add batch dimension if needed
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)  # Split batch for OpenVINO if needed

        # Inference time measurement
        with dt[1]:
            # Set up visualization path if requested (and if we're saving)
            if visualize and not nosave:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True)
            else:
                visualize = False
            
            # Run inference with the model
            if model.xml and im.shape[0] > 1:
                # Special handling for OpenVINO with batches
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                # Standard inference
                pred = model(im, augment=augment, visualize=visualize)
                
        # NMS (Non-Maximum Suppression) time measurement
        with dt[2]:
            # Apply NMS to remove overlapping bounding boxes
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions for each image/frame
        for i, det in enumerate(pred):  # det contains detections for one image
            seen += 1  # Increment counter of processed images
            
            # Set up paths and names based on source type
            if webcam:  # For webcam, handle multiple streams
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "  # Add stream index to output string
            else:  # For single images/videos
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            # Convert path to Path object
            p = Path(p)  # to Path
            
            # Define output paths only if we're saving
            if not nosave:
                save_path = str(save_dir / p.name)  # path to save image/video
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # path for text results
            else:
                save_path = ""
                txt_path = ""
            
            # Add image dimensions to the status string
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            
            # Set up normalization factor for bounding box coordinates
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            # Make a copy of the image for crop saving if needed and we're saving
            imc = im0.copy() if (save_crop and not nosave) else im0
            
            # Initialize the annotator for drawing on images
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            # Process detections if any were found
            if len(det):
                # Rescale boxes from model size to original image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Count detections by class and add to status string
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # number of detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string (pluralize if >1)

                # Process each detection box
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"  # class name
                    confidence = float(conf)  # confidence score
                    confidence_str = f"{confidence:.2f}"  # formatted confidence string

                    # Save to CSV if requested
                    if save_csv and not nosave:
                        write_to_csv(p.name, label, confidence_str)

                    # Save to text file if requested
                    if save_txt and not nosave:  # Write to file
                        if save_format == 0:
                            # YOLO format: [class, x_center, y_center, width, height] (normalized)
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            # Pascal VOC format: [class, x1, y1, x2, y2] (normalized)
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        
                        # Create the output line with class, coordinates, and optionally confidence
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # Add bounding box to image for display
                    if view_img or (save_img and not nosave):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    # Save cropped detection if requested
                    if save_crop and not nosave:
                        crop_dir = save_dir / "crops" / names[c]
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        save_one_box(xyxy, imc, file=crop_dir / f"{p.stem}.jpg", BGR=True)

            # Get the final annotated image
            im0 = annotator.result()
            
            # Display the result if requested
            if view_img:
                # Handle window creation differently on Linux
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond delay

            # Save results (image with detections or video) if not nosave
            if save_img and not nosave:
                if dataset.mode == "image":
                    # For images, just write the file
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # For videos, initialize and write to video writer
                    if vid_path[i] != save_path:  # new video file
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        
                        # Get video properties (FPS, width, height)
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        
                        # Ensure output is saved as MP4
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    
                    # Write the frame to the video file
                    vid_writer[i].write(im0)

        # Log inference time and detection results
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print overall performance statistics
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    
    # Only log saved results if we actually saved something
    if (save_txt or save_img) and not nosave:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    # Update all models if requested (strips optimizer from models to reduce size)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


# Function to parse command-line arguments
def parse_opt():
    parser = argparse.ArgumentParser()
    # Add all the command-line arguments with descriptions
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "PillV5n.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/Pill.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.2, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-format", type=int, default=0, help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", default=True, help="do not save images/videos")  # CHANGED DEFAULT TO TRUE
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="/tmp/yolov5_detect", help="save results to project/name")  # CHANGED DEFAULT PATH
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    
    # Parse arguments and convert single image size to [h, w]
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    # Print all arguments for debugging
    print_args(vars(opt))
    return opt


# Main entry point function
def main(opt):
    # Check if all requirements are met
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    
    # Run the detection with the parsed arguments
    run(**vars(opt))


# Script entry point
if __name__ == "__main__":
    opt = parse_opt()                       # Parse command-line arguments
    main(opt)                               # Execute the main function
