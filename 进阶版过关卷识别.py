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
import pytesseract
import shutil
import re
import PyPDF2
from datetime import datetime
import time
import glob

# 配置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 全局统一的识别区域坐标（后续修改只需改这里！）
STUDENT_ID_REGION = (480, 110, 660, 220)  # 学号区域：(x1, y1, x2, y2)
PASS_ID_REGION = (600, 3330, 885, 3385)   # 过关编号区域：(x1, y1, x2, y2)

# 新增：预览功能开关
PREVIEW_ENABLED = False  # True启用预览，False禁用预览

# 成绩区域坐标 (x1, y1, x2, y2)
GRADE_REGIONS = {
    'S': (1250, 530, 1310, 590),
    'A': (1250, 628, 1310, 688),
    'B': (1250, 726, 1310, 786),
    'C': (1250, 824, 1310, 884),
    'D': (1250, 922, 1310, 982),
    'E': (1250, 1020, 1310, 1080)
}
# 红色像素检测阈值（可调整）
RED_PIXEL_THRESHOLD = 0.01  # 10%以上红色像素即认为选中

# 涂卡式学号识别配置（2B铅笔填涂，基于黑色像素比例判断）
CARD_WIDTH = 50          # 涂卡区域宽度（x方向增量，固定）
CARD_HEIGHT = 35         # 涂卡区域高度（y方向增量，固定）
CARD_BLACK_THRESHOLD = 0.4  # 黑色像素占比阈值（>80%判定为填涂）
CARD_BINARY_THRESHOLD = 127 # 二值化阈值（低于此值判定为黑色）
# 涂卡区域字典：key=区域编号（10-49），value=左上角坐标(x,y)
STUDENT_ID_CARD_REGIONS = {
    10: (230, 855), 11: (325, 855), 12: (420, 855), 13: (515, 855), 14: (605, 855),
    15: (700, 855), 16: (795, 855), 17: (890, 855), 18: (985, 855), 19: (1075, 855),
    20: (230, 920), 21: (325, 920), 22: (420, 920), 23: (515, 920), 24: (605, 920),
    25: (700, 920), 26: (795, 920), 27: (890, 920), 28: (985, 920), 29: (1075, 920),
    30: (230, 985), 31: (325, 985), 32: (420, 985), 33: (515, 985), 34: (605, 985),
    35: (700, 985), 36: (795, 985), 37: (890, 985), 38: (985, 985), 39: (1075, 985),
    40: (230, 1050), 41: (325, 1050), 42: (420, 1050), 43: (515, 1050), 44: (605, 1050),
    45: (700, 1050), 46: (795, 1050), 47: (890, 1050), 48: (985, 1050), 49: (1075, 1050)
}

def preview_roi(image, x1, y1, x2, y2, region_name):
    """显示指定区域的截图预览"""
    if not PREVIEW_ENABLED:
        return

    # 提取ROI区域
    roi = image[y1:y2, x1:x2].copy()

    # 绘制边框以便识别
    cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 255, 0), 2)

    # 显示预览窗口
    window_name = f"预览: {region_name} (按任意键继续)"
    cv2.imshow(window_name, resize_with_aspect_ratio(roi))
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyWindow(window_name)

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


def get_today_date_str():
    """获取当天日期字符串，格式为MMDD"""
    return datetime.now().strftime("%m%d")


def create_directory_structure(base_path):
    """创建所需的目录结构"""
    today = get_today_date_str()
    split_dir = os.path.join(base_path, today, "拆分文件")
    keep_dir = os.path.join(base_path, today, "留存")
    # 新增两个目录
    pre_identified_dir = os.path.join(base_path, today, "预识别")
    pre_non_file_dir = os.path.join(base_path, today, "预非文件")

    # 创建目录（如果不存在）
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(pre_identified_dir, exist_ok=True)  # 新增
    os.makedirs(pre_non_file_dir, exist_ok=True)    # 新增

    return split_dir, keep_dir, pre_identified_dir, pre_non_file_dir  # 修改返回值

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


def split_pdf_by_two_pages(pdf_path, output_dir):
    """将PDF按每两页拆分，并返回拆分后的文件路径列表"""
    split_files = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            # 计算拆分后的文件数量
            num_files = (total_pages + 1) // 2  # 向上取整

            for i in range(num_files):
                writer = PyPDF2.PdfWriter()
                start_page = i * 2
                end_page = min(start_page + 2, total_pages)

                # 添加页面到新PDF
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])

                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_filename = f"{base_name}_split_{i + 1}.pdf"
                output_path = os.path.join(output_dir, output_filename)

                # 保存拆分后的PDF
                with open(output_path, 'wb') as out_file:
                    writer.write(out_file)

                split_files.append(output_path)

        return True, split_files
    except Exception as e:
        return False, f"拆分PDF出错: {str(e)}"

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

def extract_text_from_region(image, x1, y1, x2, y2):
    """从图像的指定区域提取文本"""
    # 提取ROI区域（感兴趣区域）
    roi = image[y1:y2, x1:x2]

    # 预处理提高识别率：转为灰度图并二值化
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)

    # 识别文本
    custom_config = r'--oem 3 --psm 6'  # OCR引擎模式和页面分割模式
    text = pytesseract.image_to_string(thresh_roi, config=custom_config)

    # 去除空白字符和特殊符号
    return re.sub(r'\W+', '', text)

def extract_text_from_region_with_preview(image, x1, y1, x2, y2, region_name):
    """带预览功能的区域文本提取"""
    # 显示预览
    preview_roi(image, x1, y1, x2, y2, region_name)

    # 调用原有函数提取文本
    return extract_text_from_region(image, x1, y1, x2, y2)


def detect_red_pixels(image, x1, y1, x2, y2):
    """检测指定区域内红色像素的比例"""
    # 提取ROI区域
    roi = image[y1:y2, x1:x2]

    # 转换到HSV颜色空间，更适合颜色检测
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（考虑红色在HSV中的两个范围）
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 检测红色像素
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 计算红色像素比例
    total_pixels = red_mask.size
    red_pixels = cv2.countNonZero(red_mask)
    ratio = red_pixels / total_pixels if total_pixels > 0 else 0

    # 预览功能
    if PREVIEW_ENABLED:
        preview_roi(red_mask, 0, 0, red_mask.shape[1], red_mask.shape[0], f"红色检测区域 (比例: {ratio:.2f})")

    return ratio


def recognize_grade(image):
    """识别成绩等级"""
    detected_grades = []

    # 检测每个成绩区域
    for grade, (x1, y1, x2, y2) in GRADE_REGIONS.items():
        ratio = detect_red_pixels(image, x1, y1, x2, y2)
        print(f"  {grade}区域红色像素比例: {ratio:.2f}")

        if ratio >= RED_PIXEL_THRESHOLD:
            detected_grades.append(grade)

    # 根据检测结果确定最终成绩
    if len(detected_grades) == 1 and detected_grades[0] != 'E':
        return detected_grades[0]
    else:
        return 'E'  # 多个区域被检测到或未检测到任何区域，均返回E

def recognize_student_id(image):
    """识别学号（4位数字）"""
    # 复用全局坐标，不再重复输入
    x1, y1, x2, y2 = STUDENT_ID_REGION
    text = extract_text_from_region(image, x1, y1, x2, y2)
    # 原有验证逻辑不变
    if re.match(r'^\d{4}$', text):
        return text
    return None


def recognize_pass_id(image):
    """识别过关编号（8位数字）"""
    # 复用全局坐标，不再重复输入
    x1, y1, x2, y2 = PASS_ID_REGION
    text = extract_text_from_region(image, x1, y1, x2, y2)
    # 原有验证逻辑不变
    if re.match(r'^\d{8}$', text):
        return text
    return None


def detect_black_pixels(image, x1, y1, x2, y2, region_name):
    """检测指定区域内黑色像素比例（用于涂卡识别）"""
    # 提取涂卡区域ROI（感兴趣区域）
    roi = image[y1:y2, x1:x2].copy()

    # 预处理：灰度转换→二值化（突出黑色涂卡部分）
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 二值化：低于阈值的像素设为255（黑色前景），高于设为0（白色背景）
    _, thresh_roi = cv2.threshold(gray_roi, CARD_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # 计算黑色像素占比（二值化后非零像素即为黑色）
    total_pixels = thresh_roi.size
    black_pixels = cv2.countNonZero(thresh_roi)
    ratio = black_pixels / total_pixels if total_pixels > 0 else 0

    # 预览功能：显示二值化后的涂卡区域（方便调试）
    if PREVIEW_ENABLED:
        preview_roi(thresh_roi, 0, 0, thresh_roi.shape[1], thresh_roi.shape[0],
                    f"{region_name}（黑占比: {ratio:.2f}）")

    return ratio


def recognize_student_id_by_card(image):
    """通过涂卡方式识别4位学号（返回有效学号或None）"""
    # 定义学号每一位对应的涂卡区域范围（十位=位序：1=第1位，2=第2位...）
    digit_region_ranges = [
        (10, 19),  # 第1位（区域10-19）→ 数字0-9
        (20, 29),  # 第2位（区域20-29）→ 数字0-9
        (30, 39),  # 第3位（区域30-39）→ 数字0-9
        (40, 49)  # 第4位（区域40-49）→ 数字0-9
    ]
    student_id_digits = []  # 存储每一位识别到的有效数字

    # 逐位识别涂卡结果
    for digit_order, (start_num, end_num) in enumerate(digit_region_ranges, 1):
        print(f"  涂卡识别：正在处理学号第{digit_order}位（区域{start_num}-{end_num}）")
        detected_digits = []  # 该位中超过黑色阈值的数字

        # 遍历当前位的所有涂卡区域
        for region_num in range(start_num, end_num + 1):
            # 获取区域左上角坐标，计算右下角坐标
            if region_num not in STUDENT_ID_CARD_REGIONS:
                print(f"    警告：涂卡区域{region_num}未定义，跳过")
                continue
            x1, y1 = STUDENT_ID_CARD_REGIONS[region_num]
            x2 = x1 + CARD_WIDTH  # 自动推算右下角x（x+宽度）
            y2 = y1 + CARD_HEIGHT  # 自动推算右下角y（y+高度）

            # 计算当前区域黑色像素比例
            digit = region_num % 10  # 区域编号个位=对应数字（10→0，11→1...）
            region_name = f"学号第{digit_order}位-{region_num}（数字{digit}）"
            black_ratio = detect_black_pixels(image, x1, y1, x2, y2, region_name)

            # 判定是否为有效填涂（超过阈值）
            if black_ratio >= CARD_BLACK_THRESHOLD:
                detected_digits.append((digit, black_ratio))
                print(f"    区域{region_num}（数字{digit}）黑占比{black_ratio:.2f}，符合条件")

        # 校验当前位结果：仅允许1个有效数字
        if len(detected_digits) == 1:
            valid_digit = detected_digits[0][0]
            student_id_digits.append(str(valid_digit))
            print(f"  涂卡识别：学号第{digit_order}位确认→{valid_digit}")
        elif len(detected_digits) > 1:
            print(f"  涂卡识别失败：学号第{digit_order}位检测到{len(detected_digits)}个填涂区域")
            return None  # 同一位多结果→无效
        else:
            print(f"  涂卡识别失败：学号第{digit_order}位未检测到填涂")
            return None  # 同一位无结果→无效

    # 组合4位数字为完整学号
    if len(student_id_digits) == 4:
        card_student_id = ''.join(student_id_digits)
        print(f"  涂卡识别成功：学号={card_student_id}")
        return card_student_id
    else:
        print("  涂卡识别失败：学号位数不足")
        return None


def get_final_student_id(ocr_id, card_id):
    """对比OCR学号和涂卡学号，返回最终有效学号（None表示双无效）"""
    print(f"\n  学号对比：OCR识别结果={ocr_id if ocr_id else '无效'}，涂卡识别结果={card_id if card_id else '无效'}")

    # 按规则判定最终学号
    if card_id and ocr_id:
        # 规则1：双有效→一致用一致，不一致用涂卡
        if card_id == ocr_id:
            print(f"  最终学号：双识别一致→{card_id}")
            return card_id
        else:
            print(f"  最终学号：双识别不一致，以涂卡为准→{card_id}")
            return card_id
    elif card_id and not ocr_id:
        # 规则2：仅涂卡有效→用涂卡
        print(f"  最终学号：仅涂卡有效→{card_id}")
        return card_id
    elif not card_id and ocr_id:
        # 规则3：仅OCR有效→用OCR
        print(f"  最终学号：仅OCR有效→{ocr_id}")
        return ocr_id
    else:
        # 规则4：双无效→无学号
        print(f"  最终学号：双识别均无效→无")
        return None

def rename_pdf_based_on_ocr(pdf_path, corrected_image):
    """根据OCR识别结果重命名PDF文件"""
    dir_name = os.path.dirname(pdf_path)
    base_name = os.path.basename(pdf_path)
    file_name = base_name  # 简化文件名变量

    # ---------------------- 成绩识别 ----------------------
    grade = recognize_grade(corrected_image)
    print(f"【文件 {file_name}】")
    print(f"  识别到成绩: {grade}")

    # ---------------------- 学号识别：复用全局坐标 ----------------------
    # 从全局常量获取坐标，调用extract_text_from_region
    x1_s, y1_s, x2_s, y2_s = STUDENT_ID_REGION
    student_raw_text = extract_text_from_region_with_preview(corrected_image, x1_s, y1_s, x2_s, y2_s, "学号区域")
    # 调用原有函数验证（逻辑不变）
    ocr_student_id = recognize_student_id(corrected_image)
    # 打印原始结果和验证结果
    print(f"  学号原始识别结果: {repr(student_raw_text)}")  # repr()显示特殊字符
    print(f"  学号验证结果: {'有效（4位数字）' if ocr_student_id else '无效'}")

    # ---------------------- 学号识别：涂卡方式 ----------------------
    print(f"\n【文件 {file_name}】涂卡式学号识别开始：")
    card_student_id = recognize_student_id_by_card(corrected_image)

    # ---------------------- 确定最终学号 ----------------------
    final_student_id = get_final_student_id(ocr_student_id, card_student_id)
    print(f"  最终有效学号：{final_student_id if final_student_id else '无'}")

    # ---------------------- 过关编号识别：复用全局坐标 ----------------------
    # 从全局常量获取坐标，调用extract_text_from_region
    x1_p, y1_p, x2_p, y2_p = PASS_ID_REGION
    pass_raw_text = extract_text_from_region_with_preview(corrected_image, x1_p, y1_p, x2_p, y2_p, "过关编号区域")
    # 调用原有函数验证（逻辑不变）
    pass_id = recognize_pass_id(corrected_image)
    # 打印原始结果和验证结果
    print(f"  过关编号原始识别结果: {repr(pass_raw_text)}")
    print(f"  过关编号验证结果: {'有效（8位数字）' if pass_id else '无效'}\n")

    # ---------------------- 原有重命名逻辑（仅修改学号部分） ----------------------
    if grade:
        base_name = f"《{grade}》-{base_name}"
    if final_student_id:  # 替换为最终学号
        base_name = f"[{final_student_id}]-{base_name}"
    if pass_id:
        base_name = f"({pass_id})-{base_name}"

    new_path = os.path.join(dir_name, base_name)
    if new_path != pdf_path:
        # 用shutil.move替换os.rename
        shutil.move(pdf_path, new_path)
        return new_path, f"已重命名为: {base_name}"
    return pdf_path, "未进行重命名"


def process_pdf(pdf_path):
    """处理单个PDF文件：转换为图像→处理每一页→合并为新PDF"""
    try:
        # 将PDF转换为图像
        pages = convert_from_path(pdf_path, dpi=300)
        processed_pages = []

        # 处理每一页
        for page in pages:
            open_cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            detected_corners = detect_markers(open_cv_image)
            corrected_img = correct_to_a4_300dpi(open_cv_image, detected_corners)
            pil_image = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
            processed_pages.append(pil_image)

        if processed_pages:
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
            processed_pages[0].save(temp_path, save_all=True, append_images=processed_pages[1:])

            # 用shutil.move替换os.rename和os.remove（支持跨卷移动）
            # 先删除目标文件（如果存在）
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            # 使用shutil.move处理跨卷移动
            shutil.move(temp_path, pdf_path)

            # 后续OCR识别
            last_page_cv = cv2.cvtColor(np.array(processed_pages[0]), cv2.COLOR_RGB2BGR)
            new_path, rename_msg = rename_pdf_based_on_ocr(pdf_path, last_page_cv)

            return True, f"成功处理PDF: {pdf_path}，共{len(processed_pages)}页\n{rename_msg}"
        else:
            return False, f"PDF文件{pdf_path}没有页面内容"

    except Exception as e:
        return False, f"处理PDF {pdf_path} 时出错: {str(e)}"


def monitor_and_process(base_path, interval=300):
    """后台监测指定路径并处理新文件"""
    print(f"\n进入后台监测模式，将每隔{interval}秒检查新文件...")
    print(f"监测路径: {base_path}")
    print("按Ctrl+C可退出程序")

    # 记录已处理的文件路径，避免重复处理
    processed_files = set()

    while True:
        try:
            # 查找指定路径下的所有PDF文件
            pdf_files = glob.glob(os.path.join(base_path, "*.pdf"))
            new_files = [f for f in pdf_files if f not in processed_files]

            if new_files:
                print(f"\n发现{len(new_files)}个新文件，开始处理...")
                for pdf_path in new_files:
                    print(f"\n处理新文件: {pdf_path}")

                    # 1. 拆分PDF
                    split_dir, keep_dir, pre_identified_dir, pre_non_file_dir = create_directory_structure(base_path)
                    success, result = split_pdf_by_two_pages(pdf_path, split_dir)
                    if not success:
                        print(f"拆分失败: {result}")
                        continue

                    split_files = result
                    print(f"成功拆分为{len(split_files)}个PDF文件")

                    # 2. 将源文件移动到留存目录
                    try:
                        dest_path = os.path.join(keep_dir, os.path.basename(pdf_path))
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.move(pdf_path, dest_path)
                        print(f"源文件已移动到: {dest_path}")
                    except Exception as e:
                        print(f"移动源文件失败: {str(e)}")
                        continue

                    # 3. 处理拆分后的每个PDF文件
                    for split_file in split_files:
                        print(f"\n处理拆分后的文件: {split_file}")
                        success, message = process_pdf(split_file)
                        print(message)

                        # 获取重命名后的新路径
                        new_path = split_file
                        if success and "已重命名为: " in message:
                            new_name = message.split("已重命名为: ")[1].strip()
                            new_dir = os.path.dirname(split_file)
                            new_path = os.path.join(new_dir, new_name)
                            if not os.path.exists(new_path):
                                new_path = split_file

                        # 分类移动
                        try:
                            if os.path.exists(new_path):
                                file_name = os.path.basename(new_path)
                                has_id = bool(re.search(r'\[\d{4}\]', file_name))
                                has_pass = bool(re.search(r'\(\d{8}\)', file_name))
                                has_grade = bool(re.search(r'《[ABCDE]》', file_name))

                                dest_dir = pre_identified_dir if (
                                            has_id and has_pass and has_grade) else pre_non_file_dir
                                dest_path = os.path.join(dest_dir, file_name)

                                if os.path.exists(dest_path):
                                    os.remove(dest_path)
                                shutil.move(new_path, dest_path)
                                print(f"文件已移动到: {dest_path}")
                        except Exception as e:
                            print(f"移动文件时出错: {str(e)}")

                    # 标记为已处理
                    processed_files.add(pdf_path)

            else:
                print(f"{time.ctime()} - 未发现新文件，等待下次检查...")

            # 等待指定间隔时间
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n用户中断，退出后台监测")
            break
        except Exception as e:
            print(f"监测过程出错: {str(e)}，将继续监测")
            time.sleep(interval)

# def main():
#     pdf_paths = select_pdf_files()
#     if not pdf_paths:
#         print("未选择PDF文件，程序退出")
#         return
#
#     print(f"已选择{len(pdf_paths)}个PDF文件，开始处理...")
#
#     # 基础路径
#     base_path = r"\\192.168.110.248\guoguan"
#
#     # 创建目录结构
#     split_dir, keep_dir, pre_identified_dir, pre_non_file_dir = create_directory_structure(base_path)
#     print(f"拆分文件将保存到: {split_dir}")
#     print(f"源文件将保存到: {keep_dir}")
#     print(f"识别完整文件将保存到: {pre_identified_dir}")
#     print(f"识别不完整文件将保存到: {pre_non_file_dir}")
#
#     # 处理每个选中的PDF文件
#     for pdf_path in pdf_paths:
#         print(f"\n处理文件: {pdf_path}")
#
#         # 拆分PDF
#         success, result = split_pdf_by_two_pages(pdf_path, split_dir)
#         if not success:
#             print(f"拆分失败: {result}")
#             continue
#
#         split_files = result
#         print(f"成功拆分为{len(split_files)}个PDF文件")
#
#         # 将源文件移动到留存目录
#         try:
#             dest_path = os.path.join(keep_dir, os.path.basename(pdf_path))
#             if os.path.exists(dest_path):
#                 os.remove(dest_path)
#             shutil.move(pdf_path, dest_path)
#             print(f"源文件已移动到: {dest_path}")
#         except Exception as e:
#             print(f"移动源文件失败: {str(e)}")
#             continue
#
#         # 处理拆分后的每个PDF文件
#         for split_file in split_files:
#             print(f"\n处理拆分后的文件: {split_file}")
#             success, message = process_pdf(split_file)
#             print(message)
#
#             # 获取重命名后的新路径
#             new_path = split_file  # 初始为拆分后的原始路径
#             if success and "已重命名为: " in message:
#                 # 从message中提取新文件名
#                 new_name = message.split("已重命名为: ")[1].strip()
#                 # 构建新文件的完整路径（与原拆分文件同目录）
#                 new_dir = os.path.dirname(split_file)
#                 new_path = os.path.join(new_dir, new_name)
#                 # 验证新路径是否真的存在（防止解析错误）
#                 if not os.path.exists(new_path):
#                     new_path = split_file  # 若不存在，回退到原始路径
#
#             # 分类移动逻辑（基于新路径）
#             try:
#                 if os.path.exists(new_path):  # 确保文件存在再移动
#                     file_name = os.path.basename(new_path)
#                     # 检查文件名是否包含全部3类信息（学号、过关编号、成绩）
#                     has_id = bool(re.search(r'\[\d{4}\]', file_name))  # 匹配 [1234] 格式学号
#                     has_pass = bool(re.search(r'\(\d{8}\)', file_name))  # 匹配 (12345678) 格式过关编号
#                     has_grade = bool(re.search(r'《[ABCDE]》', file_name))  # 匹配 《A》 格式成绩
#
#                     # 决定目标目录
#                     if has_id and has_pass and has_grade:
#                         dest_dir = pre_identified_dir
#                     else:
#                         dest_dir = pre_non_file_dir
#                     dest_path = os.path.join(dest_dir, file_name)
#
#                     # 移动前删除目标位置的同名文件（避免冲突）
#                     if os.path.exists(dest_path):
#                         os.remove(dest_path)
#                     shutil.move(new_path, dest_path)
#                     print(f"文件已移动到: {dest_path}")
#                 else:
#                     print(f"警告：文件 {new_path} 不存在，跳过移动")
#             except Exception as e:
#                 print(f"移动文件时出错: {str(e)}")
#
#     print("\n所有文件处理完成！")

# 修改原main函数，在初始处理后进入监测模式
def main_with_monitor():
    # 先执行一次初始处理
    pdf_paths = select_pdf_files()
    if pdf_paths:
        print(f"已选择{len(pdf_paths)}个PDF文件，开始初始处理...")

        base_path = r"\\192.168.110.248\guoguan"
        split_dir, keep_dir, pre_identified_dir, pre_non_file_dir = create_directory_structure(base_path)
        print(f"拆分文件将保存到: {split_dir}")
        print(f"源文件将保存到: {keep_dir}")
        print(f"识别完整文件将保存到: {pre_identified_dir}")
        print(f"识别不完整文件将保存到: {pre_non_file_dir}")

        # 处理初始选择的文件
        for pdf_path in pdf_paths:
            print(f"\n处理文件: {pdf_path}")
            success, result = split_pdf_by_two_pages(pdf_path, split_dir)
            if not success:
                print(f"拆分失败: {result}")
                continue

            split_files = result
            print(f"成功拆分为{len(split_files)}个PDF文件")

            # 移动源文件
            try:
                dest_path = os.path.join(keep_dir, os.path.basename(pdf_path))
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(pdf_path, dest_path)
                print(f"源文件已移动到: {dest_path}")
            except Exception as e:
                print(f"移动源文件失败: {str(e)}")
                continue

            # 处理拆分文件
            for split_file in split_files:
                print(f"\n处理拆分后的文件: {split_file}")
                success, message = process_pdf(split_file)
                print(message)

                new_path = split_file
                if success and "已重命名为: " in message:
                    new_name = message.split("已重命名为: ")[1].strip()
                    new_dir = os.path.dirname(split_file)
                    new_path = os.path.join(new_dir, new_name)
                    if not os.path.exists(new_path):
                        new_path = split_file

                # 分类移动
                try:
                    if os.path.exists(new_path):
                        file_name = os.path.basename(new_path)
                        has_id = bool(re.search(r'\[\d{4}\]', file_name))
                        has_pass = bool(re.search(r'\(\d{8}\)', file_name))
                        has_grade = bool(re.search(r'《[ABCDE]》', file_name))

                        dest_dir = pre_identified_dir if (has_id and has_pass and has_grade) else pre_non_file_dir
                        dest_path = os.path.join(dest_dir, file_name)

                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.move(new_path, dest_path)
                        print(f"文件已移动到: {dest_path}")
                except Exception as e:
                    print(f"移动文件时出错: {str(e)}")

        print("\n初始文件处理完成")

    # 进入后台监测模式
    base_monitor_path = r"\\192.168.110.248\guoguan"  # 监测的目标路径
    monitor_and_process(base_monitor_path, 60)

if __name__ == "__main__":
    main_with_monitor()
