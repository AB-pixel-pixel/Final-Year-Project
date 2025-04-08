import cv2
import os
import re
import numpy as np
import time
def create_video_from_images(image_folder,_index = 0):
    # 获取所有png图片路径
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    # 过滤掉名称包含_depth, _seg, _map的图片
    filtered_image_paths = [p for p in image_paths if not any(s in p for s in ['_depth', '_seg', '_map'])]

    # 提取序号并排序
    image_paths_with_index = []
    for path in filtered_image_paths:
        match = re.search(r"/(\d+)_", path)  # 使用正则表达式提取序号
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
    video_path = os.path.join(os.path.dirname(image_folder), f"{_index}_output_20fps.mp4")

    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height)) # 假设帧率为30fps


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

ex_index = "6_v1"
from config import k_ex_result_folder
prefix_path = k_ex_result_folder
image_folders = [f"{prefix_path}/{ex_index}/Images/0"
                 ,f"{prefix_path}/{ex_index}/Images/1"]
for index, image_folder in  enumerate(image_folders):
    create_video_from_images(image_folder,index)

