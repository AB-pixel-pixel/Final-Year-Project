import cv2
import os
import re
import numpy as np
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import k_ex_result_folder

def create_video_from_images(image_folder,_index = 0,save_folder = "",scene_name = "-1", version = "-1"):

    print("读取俯视图： ",image_folder)
    # 获取所有png图片路径
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # 过滤掉名称包含_depth, _seg, _map的图片
    filtered_image_paths = [p for p in image_paths if "line" in p]

    # 提取序号并排序
    image_paths_with_index = []
    for path in filtered_image_paths:
        match = re.search(r'_(\d+)\.jpg$', path)  # Match the pattern "_number.png"
        if match:
            index = int(match.group(1))
            image_paths_with_index.append((index, path))

    image_paths_with_index.sort(key=lambda x: x[0])  # 按序号排序

    sorted_image_paths = [path for _, path in image_paths_with_index]

    if not sorted_image_paths:
        print("没有找到符合条件的图片")
        return

    # 读取第一张图片获取尺寸
    first_image = cv2.imread(sorted_image_paths[0])
    height, width, layers = first_image.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    # video_path = os.path.join(image_folder, "output.mp4")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if save_folder:
        video_path = os.path.join(save_folder, f"scene_{scene_name}_version_{version}_{timestamp}_20fps.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height)) # 假设帧率为30fps
        print(f"saved in {video_path}")
    else:
        video_path = os.path.join(os.path.dirname(image_folder), f"{_index}_output_20fps.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height)) # 假设帧率为30fps
        print(f"saved in {video_path}")


    for image_path in sorted_image_paths:
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"读取图片失败: {image_path}")
                continue # 跳过无法读取的图片
            video_writer.write(img)
        except Exception as e:
            print(f"处理图片{image_path}时出错: {e}")

    video_writer.release()
    print(f"视频已保存到: {video_path}")

if __name__ == '__main__':
    for scene  in range(24):
        image_folder = f"{k_ex_result_folder}/{scene}/top_down_image"
        print("image folder:",image_folder)

        if os.path.exists(image_folder):
            create_video_from_images(image_folder,save_folder="/media/airs/BIN/top_down_result",scene_name = scene)

