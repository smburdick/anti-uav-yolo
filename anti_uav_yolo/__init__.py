import sys
from anti_uav_yolo.convert_files import save_frames

__version__ = '0.1.0'

if __name__ == '__main__':
    video = sys.argv[1]
    path = sys.argv[2]

    save_frames(video, path)
