from PIL import Image

def get_image_size(image_path):
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 获取图片大小
            width, height = img.size
            print(f"图片大小: {width} x {height}")
    except Exception as e:
        print(f"无法打开图片: {e}")

if __name__ == "__main__":
    # 你可以在这里输入图片的路径
    image_path = input("请输入图片路径: ")
    get_image_size(image_path)