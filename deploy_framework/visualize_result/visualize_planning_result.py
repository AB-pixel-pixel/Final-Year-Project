import os
import time
import cv2

# 指定文件夹路径
folder_path = 'path/to/your/folder'

def get_latest_frame(folder_path):
    files = os.listdir(folder_path)
    max_frame_num = -1
    latest_frame = None
    
    for file in files:
        if file.startswith("frame_") and file.endswith(".png"):
            # 提取数字
            frame_num = int(file.split('_')[1].split('.')[0])
            if frame_num > max_frame_num:
                max_frame_num = frame_num
                latest_frame = file
                
    return latest_frame

while True:
    latest_frame = get_latest_frame(folder_path)
    
    if latest_frame:
        # 读取并展示最新的图片
        image_path = os.path.join(folder_path, latest_frame)
        image = cv2.imread(image_path)
        
        if image is not None:
            cv2.imshow('Latest Frame', image)
            cv2.waitKey(500)  # 显示500毫秒
        else:
            print("Error loading image:", latest_frame)
    
    time.sleep(0.5)  # 每0.5秒检查一次