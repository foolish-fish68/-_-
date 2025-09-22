import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
# 导入os模块（与操作系统交互的标准库），用于文件路径操作、目录管理、环境变量读取等（比如判断文件是否存在、拼接路径）
from pdf2image import convert_from_path
# pdf2image 是专门用于将 PDF 文件转换为图像的库，convert_from_path 是其核心函数（通过 PDF 文件的路径，生成图像对象）
from PIL import Image
# 从 Pillow 库（Python 图像处理库，是经典库 PIL 的分支）中导入 Image 模块。Pillow 用于图像的打开、编辑、保存等操作（比如调整图像大小、格式转换、像素级处理）
import tempfile
# 导入 tempfile 模块，用于创建临时文件 / 临时目录（比如在程序运行时临时存储 PDF 转换后的图像，程序结束后自动清理，避免占用永久存储空间）


def resize_with_aspect_ratio(image, max_width=1200, max_height=1000):
    """保持宽高比缩放图像，方便预览"""
    height, width = image.shape[:2]
    ratio_w = width / max_width
    ratio_h = height / max_height
    ratio = max(ratio_w, ratio_h)

    if ratio > 1:
        new_width = int(width / ratio)
        new_height = int(height / ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # 调用 cv2.resize 缩放图像，interpolation=cv2.INTER_AREA 表示用「区域插值法」（适合图像缩小，能保证画质）
    else:
        return image.copy()
        # 说明图像尺寸在最大限制内，无需缩放，直接返回原图像的副本（image.copy() 避免直接修改原图像）


def select_pdf_files():
    """选择PDF文件（支持多选）"""
    root = tk.Tk()
    root.withdraw()
    # 隐藏刚创建的根窗口（因为只需要 “文件选择对话框”，不需要显示主窗口，避免界面冗余）
    file_paths = filedialog.askopenfilenames(
        title="选择PDF文件",
        filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
    )

    # 过滤无效路径
    valid_paths = [path for path in file_paths if os.path.isfile(path)]
    return valid_paths if valid_paths else None
    # 函数返回值：
    # 如果 valid_paths 非空（选到了有效 PDF 文件），返回有效路径列表；
    # 如果 valid_paths 为空（没选到有效文件），返回 None（方便调用者判断是否选到了有效文件）


def detect_markers(image):
    """检测并筛选定位点（顶2圆，底2方）"""
    img_copy = image.copy()
    # 复制输入图像到 img_copy。目的是保留原始图像不被后续操作修改，方便后续可能的对比或其他处理

    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 调用 OpenCV 的 cvtColor 函数，将BGR 彩色图像转换为灰度图像
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 对灰度图像进行高斯模糊，减少噪声
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    # 对模糊后的图像进行反二值化，将图像转为 “黑白分明” 的二值图
    kernel = np.ones((5, 5), np.uint8)
    # 用 NumPy 创建一个 5×5 的全 1 数组，数据类型为 np.uint8（无符号 8 位整数）
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    # 对二值化图像进行膨胀操作（扩大前景区域）
    thresh = cv2.erode(thresh, kernel, iterations=1)
    # 对膨胀后的图像进行腐蚀操作（缩小前景区域）
    # 这段代码是图像预处理流程：通过 “灰度转换→高斯模糊→反二值化→膨胀→腐蚀”，减少图像噪声、增强目标（定位标记）的轮廓特征，为后续 “检测并筛选定位点（顶 2 圆、底 2 方）” 的核心操作做准备
    
    # 找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours：OpenCV 查找轮廓的核心函数。
    # 第 1 参数 thresh.copy()：输入二值化图像（用 copy() 避免修改原始图像）。
    # 第 2 参数 cv2.RETR_EXTERNAL：轮廓检索模式，仅检索最外层轮廓（忽略内部嵌套的小轮廓）。
    # 第 3 参数 cv2.CHAIN_APPROX_SIMPLE：轮廓近似方式，仅保留轮廓的关键端点（压缩冗余点，减少内存占用）。
    # 返回值：contours 是找到的所有轮廓的列表；_ 是轮廓的 “层级信息”（此处用 _ 忽略，因为不需要层级关系）
    circles = []
    squares = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > 10000:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # 圆形判断
        roundness = 4 * np.pi * area / (perimeter ** 2)
        if roundness > 0.85:
            # 唯一圆形判断标准
            M = cv2.moments(contour)
            # 调用 cv2.moments 计算轮廓的矩（描述轮廓的形状特征，如面积、中心坐标等）
            if M["m00"] == 0:
                continue
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            # 计算轮廓的中心坐标 (cx, cy)：
            # 一阶矩 M["m10"]、M["m01"] 描述轮廓在 x、y 方向的分布。
            # 中心 x 坐标 = m10 / m00，中心 y 坐标 = m01 / m00，最后转成整数
            circles.append((cX, cY))
            # 将圆形的中心坐标 (cx, cy) 添加到 circles 列表中
            # 注释掉预览相关代码
            # cv2.circle(img_copy, (cX, cY), 10, (0, 255, 0), -1)
            # cv2.putText(img_copy, f"Circle ({cX},{cY})", (cX + 10, cY),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            continue

        # 方形判断
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # 提取4个顶点的坐标
            points = [tuple(approx[i][0]) for i in range(4)]
            if len(points) != 4:
                # 1.4顶点判断，不符合跳过
                continue

            # 过滤过小的轮廓
            area = cv2.contourArea(contour)
            if area < 100:
                # 2.面积过小跳过
                continue

            # 计算四条边长度
            edges = []
            for i in range(4):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % 4]
                length = np.hypot(x2 - x1, y2 - y1)
                edges.append(length)

            # 计算两条对角线长度
            diag1 = np.hypot(points[0][0] - points[2][0], points[0][1] - points[2][1])
            diag2 = np.hypot(points[1][0] - points[3][0], points[1][1] - points[3][1])

            # 更严格的正方形筛选条件
            max_edge = max(edges)
            min_edge = min(edges)
            edge_ratio = min_edge / max_edge

            diag_ratio = min(diag1, diag2) / max(diag1, diag2)

            if edge_ratio < 0.9 or diag_ratio < 0.9:
                # 3.边长比过小，对角线比过小，证明不是方形，跳过；
                continue

            # 原有代码：计算角度余弦值
            angles = []
            for i in range(4):
                p1, p2, p3 = approx[i][0], approx[(i + 1) % 4][0], approx[(i + 2) % 4][0]
                v1, v2 = p2 - p1, p3 - p2
                dot = np.dot(v1, v2)
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    continue
                angles.append(abs(dot / (norm1 * norm2)))
            if len(angles) >= 3 and all(a < 0.3 for a in angles[:3]):
                # 4.连续3个顶点的角度余弦值大于等于0.3，证明不是方形，跳过
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                squares.append((cX, cY))
                # 注释掉预览相关代码
                # cv2.circle(img_copy, (cX, cY), 10, (0, 0, 255), -1)
                # cv2.putText(img_copy, f"Square ({cX},{cY})", (cX + 10, cY),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 按y坐标筛选
    if len(circles) > 2:
        circles = sorted(circles, key=lambda p: p[1])[:2]  # 顶部2个圆
    if len(squares) > 2:
        squares = sorted(squares, key=lambda p: p[1], reverse=True)[:2]  # 底部2个方

    # 验证数量
    if len(circles) != 2 or len(squares) != 2:
        # 注释掉预览相关代码
        # print(f"筛选后：圆{len(circles)}个，方{len(squares)}个（需各2个）")
        # cv2.imshow("检测调试", resize_with_aspect_ratio(img_copy))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        raise ValueError("标记数量不足")
        # 后期检查是否需要增加异常捕获，以免遇到错误直接退出，不处理后续文件

    # 排序角点：左上→右上→右下→左下
    circles_sorted = sorted(circles, key=lambda p: p[0])
    top_left, top_right = circles_sorted
    squares_sorted = sorted(squares, key=lambda p: p[0])
    bottom_left, bottom_right = squares_sorted
    corner_order = [top_left, top_right, bottom_right, bottom_left]

    # 注释掉预览相关代码
    # 标记目标点
    # for i, (x, y) in enumerate(corner_order):
    #     cv2.circle(img_copy, (x, y), 15, (255, 0, 0), 2)
    #     cv2.putText(img_copy, f"P{i + 1}", (x - 10, y - 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 显示筛选结果
    # cv2.imshow("定位点识别结果", resize_with_aspect_ratio(img_copy))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return corner_order


def correct_to_a4_300dpi(image, src_points):
    """
    将图像校正为A4 300dpi标准尺寸(2481×3508像素)
    源点与标准坐标严格映射，确保所有内容完整显示
    """
    # A4 300dpi标准尺寸：2481×3508像素
    A4_WIDTH = 2481
    A4_HEIGHT = 3508

    # 标准定位点坐标
    standard_dst_points = np.array([
        [145, 125],  # 左上标准点
        [2335, 125],  # 右上标准点
        [2335, 3380],  # 右下标准点
        [145, 3380]  # 左下标准点
    ], dtype=np.float32)

    # 确保标准点在A4范围内
    for (x, y) in standard_dst_points:
        if not (0 <= x <= A4_WIDTH and 0 <= y <= A4_HEIGHT):
            raise ValueError(f"标准坐标({x},{y})超出A4 300dpi范围(0-2481, 0-3508)")

    # 透视变换：将识别到的源点映射到A4标准点
    src = np.array(src_points, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, standard_dst_points)

    # 添加边界填充参数，将超出区域填充为白色
    corrected_img = cv2.warpPerspective(
        image,
        M,
        (A4_WIDTH, A4_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # 在校正后的图像上标记标准定位点（静默处理，不显示）
    for (x, y) in standard_dst_points:
        cv2.circle(corrected_img, (int(x), int(y)), 10, (0, 255, 255), -1)

    return corrected_img


def process_pdf(pdf_path):
    """处理单个PDF文件：转换为图像→处理每一页→合并为新PDF"""
    try:
        # 将PDF转换为图像
        pages = convert_from_path(pdf_path, dpi=300)
        processed_pages = []

        # 处理每一页
        for page in pages:
            # 转换为OpenCV格式
            open_cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

            # 检测定位点
            detected_corners = detect_markers(open_cv_image)

            # 校正为A4 300dpi
            corrected_img = correct_to_a4_300dpi(open_cv_image, detected_corners)

            # 转换回PIL格式
            pil_image = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
            processed_pages.append(pil_image)

        # 保存处理后的PDF（覆盖原文件）
        if processed_pages:
            # 先保存到临时文件，再替换原文件，避免处理失败时损坏原文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name

            processed_pages[0].save(temp_path, save_all=True, append_images=processed_pages[1:])

            # 删除原文件并替换
            os.remove(pdf_path)
            os.rename(temp_path, pdf_path)

            return True, f"成功处理PDF: {pdf_path}，共{len(processed_pages)}页"
        else:
            return False, f"PDF文件{pdf_path}没有页面内容"

    except Exception as e:
        return False, f"处理PDF {pdf_path} 时出错: {str(e)}"


def main():
    pdf_paths = select_pdf_files()
    if not pdf_paths:
        print("未选择PDF文件，程序退出")
        return

    print(f"已选择{len(pdf_paths)}个PDF文件，开始处理...")

    # 逐个处理PDF文件
    for pdf_path in pdf_paths:
        success, message = process_pdf(pdf_path)
        print(message)

    print("所有文件处理完成！")


if __name__ == "__main__":

    main()
