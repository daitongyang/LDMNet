import cv2
import numpy as np
def color_map(dataname):
    # “”“
    # 这里可以预设多个对应关系，疯狂if... elif...即可
    # 这里以LoveDa数据集为例：
    # 该方法返回的是对应关系，类型为字典dict
    # 注意图像通道 （B G R）RGB
    # ”“”
    # 0: [41, 167, 224],  # 障碍物
    # 1: [247, 195, 37],  # 水
    # 2: [90, 75, 164],  # 天
    if dataname == "LoveDa":
        color_map_ = {
            0:  [37, 195, 247], # 障碍物
            1: [224, 167, 41],  # 水
            2:  [164, 75, 90],  # 天
            3: [0,0,0],  # road
            4: [90, 75, 164],  # water
            5: [0,0,0],  # barren
            6: [0,0,0],  # forest
            7: [0,0,0],  # agriculture
            }
        return color_map_
    else:
        print("DATA name error!")
def gt2color(path):
    # read image （np.array）
    img = cv2.imread(path)
    # show image
    cv2.imshow("GT", img)
    # 生成一个尺寸相同的图像，注意这里label图像是3通道的，如果你label是单通道，稍微修改下
    img_out = np.zeros(img.shape, np.uint8)
    # 获取图像宽*高
    img_x, img_y = img.shape[0], img.shape[1]
    # 得到该数据色彩映射关系
    label_color = color_map(dataname=dataname)
    # 每行每列像素值依次替换，这里写的简单，你也可以用for ... enumerate...获取索引和值
    for x in range(img_x):
        for y in range(img_y):
            label = img[x][y][0]    # get img label
            img_out[x][y] = label_color[label]
    cv2.imwrite(r"E:\daity\mmsegmentation-at-af\mmsegmentation\demo\img\outImg\groundtruth_usv\X19_03_0000002000new.png", img_out)
    cv2.imshow("RGB", img_out)

    return img_out
    # 另存图片


if __name__ == "__main__":
    # 定义数据集，图像路径
    dataname = r"LoveDa"
    img_path = r"E:\daity\dataset\USVInland_both\annotations\train\X19_03_0000002000new.png"
    # img_fold = r"...\datasets" # 批处理根目录
    get_color_img = gt2color(img_path)
    cv2.waitKey()