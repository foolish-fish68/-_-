import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from pdf2image import convert_from_path
from PIL import Image
import tempfile


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
    else:
        return image.copy()


def select_pdf_files():
    """选择PDF文件（支持多选）"""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="选择PDF文件",
        filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
    )

    # 过滤无效路径
    valid_paths = [path for path in file_paths if os.path.isfile(path)]
    return valid_paths if valid_paths else None


def detect_markers(image):
    """检测并筛选定位点（顶2圆，底2方）"""
    img_copy = image.copy()

    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # 找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            circles.append((cX, cY))
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
                continue

            # 过滤过小的轮廓
            area = cv2.contourArea(contour)
            if area < 100:
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