import cv2, os, sys, json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import json

FPS = 25
WIDTH = 1920
HEIGHT = 1080

# Convert video into series of pngs
def save_frames(video_input_path):
    if "~" in video_input_path:
        video_input_path = os.path.expanduser(video_input_path)
    if not os.path.isfile(video_input_path):
        _fail(video_input_path + " not found")

    capture = cv2.VideoCapture(video_input_path)
    frames = []
    while True:
        ret, frame = capture.read()
        if ret is False or frame is None:
            break
        frames.append(frame)
    return frames
        

def convert_frame_data(jsonfilepath, imagedir, labeldir):
    with open(jsonfilepath, "r") as jsonfile:
        j = json.loads(jsonfile.read())
        assert len(j["exist"]) == len(j["gt_rect"]), "Invalid frame data inputs"
        idx = 0
        # get file name
        filename = jsonfilepath.split("/")[-2]
        for exists, bbox, frame in zip(j["exist"], j["gt_rect"], ):
            bbox = [int(x) for x in bbox]
            if exists == '1':
                ctr_x = (bbox[0] + bbox[2] / 2) / WIDTH
                ctr_y = (bbox[1] + bbox[3] / 2) / HEIGHT
                bbox_width = bbox[2] / WIDTH
                bbox_height = bbox[3] / HEIGHT
                # Write frame data to file
                with open(f"{labeldir}/{filename}_f{idx}.txt", "w") as datafile:
                    datafile.write(f"0 {ctr_x} {ctr_y} {bbox_width} {bbox_height}")
                cv2.imwrite(f"{imagedir}/{idx}.jpg", frame)
            idx += 1

def test_bbox(image_file, bbox):
    image = Image.open(image_file)
    plotted_image = ImageDraw.Draw(image)
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[0] + bbox[2]
    y1 = bbox[1] + bbox[3]
    plotted_image.rectangle(((x0, y0), (x1, y1)))
    print(f"Bbox = {bbox}")
    plt.imshow(np.array(image))
    plt.show()

def _fail(msg):
    print(msg)
    exit()

if __name__ == '__main__':
    # video = sys.argv[1]
    # path = sys.argv[2]
    # test_bbox("./tmp/0.jpg", [650,390,69,42])
    # test_bbox("./tmp/1.jpg", [631,393,70,43])
    with open("/Users/sam/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Anti-UAV-RGBT/val/20190925_101846_1_4/visible.json", "r") as jfile:
        j = json.loads(jfile.read())
        idx = 0
        for box in j["gt_rect"]:
            test_bbox(f"./tmp/{idx}.jpg", box)
            idx += 1
    #save_frames(video, path)
