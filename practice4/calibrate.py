import math
import cv2
import numpy as np
import os


def rotationMatrixToEulerAngles(R):
    """ 将旋转矩阵转换为欧拉角（ZYX顺序）"""
    # 检查是否为有效旋转矩阵
    assert (np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6))
    assert (np.allclose(np.linalg.det(R), 1.0, atol=1e-6))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])  # 绕X轴
        y = math.atan2(-R[2, 0], sy)  # 绕Y轴
        z = math.atan2(R[1, 0], R[0, 0])  # 绕Z轴
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([z, y, x])  # 返回 ZYX 欧拉角（单位：度）


# 配置
chessboard_size = (9, 6)  # 角点数量
square_size = 17.0  # 每个格子的边长（mm）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

# 角点的世界坐标，x轴从左向右，y轴自上而下，z轴垂直指向相机
world = []
for i in range(chessboard_size[1]):
    for j in range(chessboard_size[0]):
        world.append([j * square_size, i * square_size, 0])
world = np.array(world, dtype=np.float32)

world_points = []  # 世界坐标
img_points = []  # 像素坐标

# 遍历图像，识别角点
image_names = os.listdir(image_dir)
for name in image_names:
    path = os.path.join(image_dir, name)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转成灰度图
    # corners(9*6, 1, 2)，表示找到的角点的像素坐标
    exist_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if exist_corners:
        world_points.append(world)
        # 对刚才的像素坐标进一步精确定位
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        # 显示识别出的角点
        cv2.drawChessboardCorners(img, chessboard_size, corners2, exist_corners)
        cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Corners", 800, 600)
        screen_img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
        cv2.imshow("Corners", screen_img)
        cv2.waitKey(100)
cv2.destroyAllWindows()

# 标定
exist_corners, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=world_points,
    imagePoints=img_points,
    imageSize=gray.shape[::-1],
    cameraMatrix=None,  # 初始相机内参矩阵,none表示自动估计
    distCoeffs=None  # 畸变系数
)

# 输出标定结果并转换为欧拉角
print("内参矩阵 K\n", camera_matrix)
print("畸变系数  \n", dist_coeffs.ravel())
for i in range(len(rvecs)):
    print("\n图像 {}:".format(i + 1))
    print("旋转向量  :", rvecs[i].ravel())
    print("平移向量 t:", tvecs[i].ravel())
    R, _ = cv2.Rodrigues(rvecs[i])  # 旋转向量 → 旋转矩阵
    euler_angles = rotationMatrixToEulerAngles(R)  # 旋转矩阵 → 欧拉角 (ZYX顺序)
    print("欧拉角(ZYX): Z={:.2f}°, Y={:.2f}°, X={:.2f}°".format(*euler_angles))

# 计算平均重投影误差
total_error = 0
for i in range(len(world_points)):
    imgpoints2, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("平均重投影误差\n", total_error / len(world_points))

# 保存结果
np.savez(f"{BASE_DIR}/camera_params.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\n参数已保存为 camera_params.npz")
