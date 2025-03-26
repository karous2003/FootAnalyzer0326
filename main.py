import io
import json
import math
import os
import time
import uuid
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, ImageDraw, ImageOps
from rembg import remove
from scipy.spatial import KDTree
import onnxruntime as ort
from flask_cors import CORS

# 初始化 Flask 應用程式，設置靜態文件夾
app = Flask(__name__, static_folder='static')

# 定義檔案路徑常量
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
INSOLE_OUTPUT_FOLDER = 'static/insole'
HALLUX_OUTPUT_FOLDER = 'static/hallux'
FOOT_REMOVEBG_FOLDER = 'static/hallux/foot_removebg'
FOOTOUTLINE_FOLDER = 'footoutline'
IOU_FOLDER = 'IoU'

# 確保必要的資料夾存在，若不存在則創建
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, INSOLE_OUTPUT_FOLDER, 
               HALLUX_OUTPUT_FOLDER, FOOT_REMOVEBG_FOLDER, IOU_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 啟用跨域資源共享（CORS）
CORS(app)

# 載入 YOLO 模型，用於雙足去背
hallux_yolo_model = YOLO("model/best.pt")

# 載入 ONNX 模型，用於雙足關鍵點檢測
model_path = "model/keypoint_rcnn/labelme-keypoint-eyes-noses-dataset-keypointrcnn_resnet50_fpn.onnx"
session = ort.InferenceSession(model_path)

# ================= 靜態檔案處理與檔案刪除 =================
@app.route("/static/insole/<filename>")
def get_insole_file(filename):
    """提供鞋墊圖片的靜態檔案存取"""
    return send_from_directory(INSOLE_OUTPUT_FOLDER, filename)

@app.route("/static/hallux/<filename>")
def get_hallux_file(filename):
    """提供姆趾外翻處理後圖片的靜態檔案存取"""
    return send_from_directory(HALLUX_OUTPUT_FOLDER, filename)

@app.route("/footoutline/<filename>")
def get_footoutline_file(filename):
    """提供足部輪廓圖片的靜態檔案存取"""
    return send_from_directory(FOOTOUTLINE_FOLDER, filename)

@app.route("/static/foot_removebg/<filename>")
def get_foot_removebg_file(filename):
    """提供去背後足部圖片的靜態檔案存取"""
    return send_from_directory(FOOT_REMOVEBG_FOLDER, filename)

@app.route("/delete-image", methods=["DELETE"])
def delete_image():
    """根據檔案名稱與類型刪除指定圖片"""
    filename = request.args.get("filename")
    file_type = request.args.get("type")
    
    if not filename or not file_type:
        return jsonify({'success': False, 'error': '缺少檔案名稱或類型'}), 400
    
    filename = os.path.basename(filename)
    print("delet_url", filename)
    file_path = {
        "insole": os.path.join(INSOLE_OUTPUT_FOLDER, filename).replace("\\", "/"),
        "hallux": os.path.join(HALLUX_OUTPUT_FOLDER, filename).replace("\\", "/"),
        "foot_removebg": os.path.join(FOOT_REMOVEBG_FOLDER, filename).replace("\\", "/")
    }.get(file_type)
    
    print("file_path", file_path)
    
    if not file_path:
        return jsonify({'success': False, 'error': '無效的檔案類型'}), 400
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'success': True, 'message': '檔案已刪除'})
    return jsonify({'success': False, 'error': '檔案不存在'}), 404

# ================= 首頁路由 =================
@app.route("/", methods=["GET", "POST"])
def home():
    """渲染應用程式首頁"""
    if request.method == "POST":
        pass  # 目前無 POST 處理邏輯
    return render_template("home.html", image_url=None, result_data=None)

# ================= 鞋墊圖片處理 =================
@app.route("/insole", methods=["GET", "POST"])
def insole():
    """處理鞋墊圖片的上傳與分析"""
    if request.method == "POST":
        print("收到圖片上傳請求")
        if "file" not in request.files:
            print("未上傳檔案")
            return jsonify({'success': False, 'error': '未上傳檔案'}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({'success': False, 'error': '未選擇檔案'}), 400
        
        # 為攝影機捕捉的圖片生成唯一名稱
        unique_filename = f"camera_{uuid.uuid4().hex[:8]}.jpg" if file.filename == 'camera-capture.jpg' else file.filename
        filename = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filename)
        
        try:
            points_data = request.form.get("points")
            if points_data:
                # 若有傳入四點座標，執行透視轉換
                points = json.loads(points_data)
                processor = ImageProcessor()
                warped_path = processor.A4_transform(filename, points)
                output_filename, result_data = process_insole(warped_path)
            else:
                output_filename, result_data = process_insole(filename)
            
            # 根據請求類型返回 JSON 或 HTML
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'image_url': url_for('get_insole_file', filename=output_filename, _external=True),
                    'result_data': result_data
                })
            return render_template("insole.html", image_url=output_filename, result_data=result_data)
        except Exception as e:
            error_message = str(e)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': error_message}), 400
            return f"Error processing image: {error_message}", 400
        finally:
            # 清理臨時檔案
            if os.path.exists(filename):
                os.remove(filename)
    return render_template("insole.html", image_url=None, result_data=None)

class ImageProcessor:
    """圖像處理工具類，提供透視轉換等功能"""
    def __init__(self):
        self.points = []
    
    def order_points(self, pts):
        """將四個點按順時針排序（左上、右上、右下、左下）"""
        rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]  # 左上與右下
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]  # 右上與左下
        return rect

    def four_point_transform(self, image, pts):
        """根據四點座標執行透視轉換，生成 A4 大小圖像"""
        rect = self.order_points(pts)
        dst = np.array([[0, 0], [2479, 0], [2479, 3507], [0, 3507]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (2480, 3508))

    def A4_transform(self, image_path, points):
        """將圖片轉換為 A4 大小並保存"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取圖片 {image_path}")
        
        # 處理前端傳來的點座標格式
        if isinstance(points[0], dict):
            points = [(p['x'], p['y']) for p in points]
        points_array = np.array(points, dtype="float32")
        if points_array.shape != (4, 2):
            raise ValueError("請提供四個點，每個點需包含 x 與 y 座標")
        
        warped_image = self.four_point_transform(image, points_array)
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{original_filename}_transformed.png"
        output_path = os.path.join(INSOLE_OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, warped_image)
        print(f"轉換後的圖片已儲存至: {output_path}")
        return output_path

def process_insole(image_path):
    """
    處理鞋墊圖片：利用 rembg 進行背景去背，使用二值化影像進行輪廓分析與旋轉校正，
    並繪製鞋墊總長、前掌寬、中足寬與後跟寬等尺寸到去背後且旋轉校正後的圖像上。
    最終圖片將根據初始方向進行調整，確保頭在上尾在下。
    """
    start_time = time.time()
    
    # 讀取圖片
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise Exception("無法讀取圖片檔案")
    
    # 調整尺寸
    max_dimension = 2000
    height, width = input_image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        input_image = cv2.resize(input_image, None, fx=scale, fy=scale)
    
    # 背景去背：使用 rembg
    segmented_image = remove(input_image)
    
    # 將去背後的圖像轉換為 3 通道（BGR）
    if segmented_image.shape[2] == 4:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGRA2BGR)
    
    # 灰階與二值化處理
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=5, beta=0)  # 增強對比度
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形態學閉運算填補小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 輪廓檢測
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("圖片中未找到輪廓")
    max_contour = max(contours, key=cv2.contourArea)
    
    # 利用 KDTree 找出最遠的兩點進行旋轉校正
    points = np.array([pt[0] for pt in max_contour])
    kdtree = KDTree(points)
    max_distance = 0
    topmost, bottommost = None, None
    for point in points:
        dist, idx = kdtree.query(point, k=points.shape[0])
        farthest_point = points[idx[-1]]
        distance = np.linalg.norm(point - farthest_point)
        if distance > max_distance:
            max_distance = distance
            topmost, bottommost = point, farthest_point
    
    # 計算旋轉角度
    def calculate_rotation_angle(topmost, bottommost):
        vector_line = (bottommost[0] - topmost[0], bottommost[1] - topmost[1])
        vector_mid_line = (0, 1)
        dot_product = vector_line[0] * vector_mid_line[0] + vector_line[1] * vector_mid_line[1]
        magnitude_line = math.sqrt(vector_line[0] ** 2 + vector_line[1] ** 2)
        magnitude_mid_line = math.sqrt(vector_mid_line[0] ** 2 + vector_mid_line[1] ** 2)
        cos_angle = dot_product / (magnitude_line * magnitude_mid_line)
        cos_angle = max(-1, min(1, cos_angle))
        angle_radians = math.acos(cos_angle)
        return math.degrees(angle_radians)
    
    # 旋轉圖像
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    # 根據最遠兩點進行旋轉校正
    if topmost[1] < bottommost[1]:
        angle_degrees = calculate_rotation_angle(topmost, bottommost)
        rotated_image = rotate_image(segmented_image, -angle_degrees)
        rotated_binary = rotate_image(binary, -angle_degrees)
    elif topmost[1] > bottommost[1]:
        angle_degrees = 180 - calculate_rotation_angle(topmost, bottommost)
        rotated_image = rotate_image(segmented_image, angle_degrees)
        rotated_binary = rotate_image(binary, angle_degrees)
    else:
        rotated_image = segmented_image
        rotated_binary = binary
    
    # 重新取得旋轉後的輪廓
    rotated_contours, _ = cv2.findContours(rotated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not rotated_contours:
        raise Exception("旋轉後的圖片中未找到輪廓")
    rotated_max_contour = max(rotated_contours, key=cv2.contourArea)
    
    # 計算鞋墊長度
    bottom_point = tuple(rotated_max_contour[rotated_max_contour[:, :, 1].argmax()][0])
    front_point = tuple(rotated_max_contour[rotated_max_contour[:, :, 1].argmin()][0])
    foot_length_pixels = math.dist(bottom_point if 'custom_point' in locals() else bottom_point, front_point)
    
    # 利用 A4 紙比例估算每公分的像素數
    a4_width_cm = 21.0
    a4_length_cm = 29.7
    image_height, image_width = rotated_image.shape[:2]
    pixels_per_cm = ((image_width / a4_width_cm) + (image_height / a4_length_cm)) / 2
    insole_length_cm = foot_length_pixels / pixels_per_cm
    
    # 繪製線段（不含文字）
    def draw_line(image, point1, point2, color, line_thickness=10):
        cv2.line(image, point1, point2, color, line_thickness)
    
    # 繪製鞋墊長（藍線）
    draw_line(rotated_image, bottom_point, front_point, (255, 135, 0))
    cv2.circle(rotated_image, bottom_point, 12, (255, 135, 0), -1)
    cv2.circle(rotated_image, front_point, 12, (255, 135, 0), -1)
    
    # 檢測鞋墊方向
    y_mid = (front_point[1] + bottom_point[1]) // 2
    upper_points = [pt[0] for pt in rotated_max_contour if pt[0][1] <= y_mid]
    lower_points = [pt[0] for pt in rotated_max_contour if pt[0][1] > y_mid]
    orientation = "normal"
    if upper_points and lower_points:
        upper_width = max(upper_points, key=lambda x: x[0])[0] - min(upper_points, key=lambda x: x[0])[0]
        lower_width = max(lower_points, key=lambda x: x[0])[0] - min(lower_points, key=lambda x: x[0])[0]
        if upper_width > lower_width:
            orientation = "normal"
        else:
            orientation = "inverted"
    
    # 根據方向調整計算參數
    if orientation == "normal":
        y_threshold = front_point[1] + int(foot_length_pixels * 0.5)
        front_half_points = [pt[0] for pt in rotated_max_contour if pt[0][1] <= y_threshold]
        midfoot_y = int(bottom_point[1] - 0.4 * foot_length_pixels if 'custom_point' in locals() else bottom_point[1] - 0.4 * foot_length_pixels)
        heel_y = bottom_point[1] - int(0.15 * foot_length_pixels)
    else:
        y_threshold = bottom_point[1] - int(foot_length_pixels * 0.5)
        front_half_points = [pt[0] for pt in rotated_max_contour if pt[0][1] >= y_threshold]
        midfoot_y = int(front_point[1] + 0.4 * foot_length_pixels)
        heel_y = front_point[1] + int(0.15 * foot_length_pixels)
    
    # 前掌寬
    forefoot_width = 0
    forefoot_center = None
    if front_half_points:
        left_most = tuple(min(front_half_points, key=lambda x: x[0]))
        right_most = tuple(max(front_half_points, key=lambda x: x[0]))
        forefoot_width = math.dist(left_most, right_most) / pixels_per_cm
        draw_line(rotated_image, left_most, right_most, (0, 0, 255))
        forefoot_center = ((left_most[0] + right_most[0]) // 2, (left_most[1] + right_most[1]) // 2)
        cv2.circle(rotated_image, forefoot_center, 12, (0, 0, 255), -1)
    
    # 尋找輪廓最近點
    def find_nearest_contour_point(image, start_point, direction, contour, target_y):
        x, y = start_point
        step = 5 if direction == "right" else -5
        closest_point = None
        min_dist = float('inf')
        for offset_y in range(-20, 20, 1):
            x_temp, y_temp = x, y + offset_y
            while 0 <= x_temp < image.shape[1]:
                point = (float(x_temp), float(y_temp))
                dist = cv2.pointPolygonTest(contour, point, True)
                if dist >= 0 and abs(dist) < min_dist:
                    min_dist = abs(dist)
                    closest_point = (int(x_temp), target_y)
                x_temp += step
        return closest_point
    
    # 中足寬
    midfoot_pt = (bottom_point[0], midfoot_y)
    left_point = find_nearest_contour_point(rotated_image, midfoot_pt, "left", rotated_max_contour, midfoot_pt[1])
    right_point = find_nearest_contour_point(rotated_image, midfoot_pt, "right", rotated_max_contour, midfoot_pt[1])
    midfoot_width = 0
    if left_point and right_point:
        midfoot_width = math.dist(left_point, right_point) / pixels_per_cm
        draw_line(rotated_image, left_point, right_point, (255, 0, 155))
        cv2.circle(rotated_image, left_point, 12, (255, 0, 155), -1)
        cv2.circle(rotated_image, right_point, 12, (255, 0, 155), -1)
    
    # 後跟寬
    heel_center = (bottom_point[0], heel_y)
    heel_width_left = find_nearest_contour_point(rotated_image, heel_center, "left", rotated_max_contour, heel_y)
    heel_width_right = find_nearest_contour_point(rotated_image, heel_center, "right", rotated_max_contour, heel_y)
    heel_width = 0
    if heel_width_left and heel_width_right and forefoot_center:
        heel_width = math.dist(heel_width_left, heel_width_right) / pixels_per_cm
        draw_line(rotated_image, heel_width_left, heel_width_right, (0, 255, 255))
        cv2.circle(rotated_image, heel_width_left, 12, (0, 255, 255), -1)
        cv2.circle(rotated_image, heel_width_right, 12, (0, 255, 255), -1)
        heel_center_point = ((heel_width_left[0] + heel_width_right[0]) // 2, heel_y)
        green_line_start = heel_center_point
        green_line_end = forefoot_center
        line_direction = np.array([green_line_end[0] - green_line_start[0],
                                   green_line_end[1] - green_line_start[1]], dtype=float)
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length
        step_size = 15
        green_line_endpoint = None
        current_end = green_line_end
        while True:
            current_end = (int(current_end[0] + line_direction[0] * step_size),
                           int(current_end[1] + line_direction[1] * step_size))
            if cv2.pointPolygonTest(rotated_max_contour, current_end, True) < 0:
                green_line_endpoint = current_end
                break
        if green_line_endpoint is not None:
            cv2.circle(rotated_image, green_line_endpoint, 12, (0, 255, 0), -1)
            cv2.circle(rotated_image, green_line_start, 12, (0, 255, 255), -1)
            cv2.line(rotated_image, green_line_start, green_line_endpoint, (0, 255, 0), 10)
    else:
        print("無法取得繪製中心線的必要點")
    
    # 如果鞋墊初始為顛倒狀態，則將整張圖上下翻轉
    if orientation == "inverted":
        rotated_image = cv2.flip(rotated_image, 0)  # 上下翻轉
    
    # 最終調整尺寸為 A4 (2480 × 3508)
    if rotated_image.shape[:2] != (3508, 2480):
        rotated_image = cv2.resize(rotated_image, (2480, 3508), interpolation=cv2.INTER_AREA)
    
    # 統一在右上角繪製所有尺寸
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  
    thickness = 4     

    # 設置右上角文字位置
    start_x = 1580 
    start_y = 150  
    line_spacing = 80  
    # 定義測量項目與對應的線段顏色 (BGR 格式)
    measurements = [
        {"text": f"Insole Length: {insole_length_cm:.2f} cm", "color": (255, 135, 0)},  # 藍線
        {"text": f"Forefoot Width: {forefoot_width:.2f} cm", "color": (0, 0, 255)},  # 紅線
        {"text": f"Midfoot Width: {midfoot_width:.2f} cm", "color": (255, 0, 155)},  # 紫線
        {"text": f"Heel Width: {heel_width:.2f} cm", "color": (0, 255, 255)}  # 黃線
    ]

    # 繪製文字標籤（無背景色）
    for i, measurement in enumerate(measurements):
        text = measurement["text"]
        text_color = measurement["color"]
        y_pos = start_y + i * line_spacing
        cv2.putText(rotated_image, text, (start_x, y_pos), 
                    font, font_scale, text_color, thickness)

    # 保存結果
    output_filename = f"result_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(INSOLE_OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, rotated_image)
    
    elapsed_time = time.time() - start_time
    result_data = {
        "insole_length_cm": float(round(insole_length_cm, 2)),
        "forefoot_width_cm": float(round(forefoot_width, 2)),
        "midfoot_width_cm": float(round(midfoot_width, 2)),
        "heel_width_cm": float(round(heel_width, 2)),
        "processing_time": float(round(elapsed_time, 2))
    }
    return output_filename, result_data

# ================= 姆趾外翻處理與去背顯示 =================
@app.route("/static/<filename>")
def get_file(filename):
    """提供靜態檔案存取"""
    return send_from_directory(OUTPUT_FOLDER, filename)

def foot_removebg(image_path):
    """對雙足圖片進行去背處理，分離左右腳"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    
    # 使用 YOLO 模型檢測足部
    results = hallux_yolo_model.predict(image_path)
    result = results[0]
    masks, boxes = result.masks, result.boxes
    print("原圖大小:", img.size)
    
    if not masks:
        return None, None, None
    
    foot_detections = []
    for i, mask in enumerate(masks):
        conf = boxes[i].conf.item()
        if conf <= 0.95:
            return None, None, None
        
        mask_data = mask.data[0].numpy()
        mask_img = (mask_data * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        
        min_x = min([pt[0][0] for contour in contours for pt in contour])
        foot_detections.append({"conf": conf, "mask": mask_img, "min_x": min_x})
    
    foot_detections.sort(key=lambda x: x["min_x"])
    if len(foot_detections) != 2:
        return None, None, None
    
    # 標記左右腳
    foot_detections[0]["cls_name"] = "left-foot"
    foot_detections[1]["cls_name"] = "right-foot"
    
    # 分離左右腳並生成去背圖
    image_rgba = img.convert("RGBA")
    transparent_left = Image.new("RGBA", img.size, (0, 0, 0, 0))
    transparent_right = Image.new("RGBA", img.size, (0, 0, 0, 0))
    transparent_full = Image.new("RGBA", img.size, (0, 0, 0, 0))
    
    for foot in foot_detections:
        cls_name = foot["cls_name"]
        mask_img = Image.fromarray(foot["mask"]).resize(img.size)
        if cls_name == "left-foot":
            transparent_left = Image.composite(image_rgba, transparent_left, mask_img)
        elif cls_name == "right-foot":
            transparent_right = Image.composite(image_rgba, transparent_right, mask_img)
        transparent_full = Image.composite(image_rgba, transparent_full, mask_img)
    
    # 保存去背結果到 FOOT_REMOVEBG_FOLDER
    basename = os.path.splitext(os.path.basename(image_path))[0]
    def save_image(img, suffix):
        filename = f"{basename}_{suffix}.png"
        img_path = os.path.join(FOOT_REMOVEBG_FOLDER, filename)
        img.save(img_path, format="PNG")
        return f"/static/foot_removebg/{filename}"
    
    left_image_url = save_image(transparent_left, "left_foot")
    right_image_url = save_image(transparent_right, "right_foot")
    full_image_url = save_image(transparent_full, "full")
    
    return left_image_url, right_image_url, full_image_url

def calculate_iou(box1, box2):
    """計算兩個邊界框的交並比（IoU）"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

@app.route("/iou", methods=["POST"])
def iou_computation():
    """計算前端傳來的邊界框與模型檢測框的 IoU"""
    start_time = time.time()
    if "file" not in request.files:
        return jsonify({"success": False, "error": "沒有圖片"})
    
    file = request.files["file"]
    unique_filename = f"camera_{uuid.uuid4().hex[:8]}.jpg" if file.filename == 'camera-capture.jpg' else file.filename
    filename = os.path.join(IOU_FOLDER, unique_filename)
    file.save(filename)
    img = Image.open(filename)
    img = ImageOps.exif_transpose(img)
    
    try:
        left_bbox = json.loads(request.form.get("left_bbox", "[0,0,0,0]"))
        right_bbox = json.loads(request.form.get("right_bbox", "[0,0,0,0]"))
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "bounding box 解析錯誤"})
    
    print("前端左腳 BBox:", left_bbox)
    print("前端右腳 BBox:", right_bbox)
    
    # 使用 YOLO 模型檢測足部
    results = hallux_yolo_model.predict(filename)
    num_foot_contours = len(results[0].boxes)
    print(f"Detected {num_foot_contours} foot-contours")
    
    foot_boxes, foot_centers = [], []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                foot_boxes.append([x1, y1, x2, y2])
                foot_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    if len(foot_boxes) != 2:
        os.remove(filename)
        return jsonify({"success": False, "error": "未能偵測到兩隻腳"})
    
    # 根據中心點判斷左右腳
    print("中心點:", foot_centers)
    left_foot, right_foot = (foot_boxes if foot_centers[0][0] < foot_centers[1][0] 
                             else foot_boxes[::-1])
    
    iou_left = calculate_iou(left_foot, left_bbox)
    iou_right = calculate_iou(right_foot, right_bbox)
    
    # 記錄處理時間並清理檔案
    processing_time = time.time() - start_time
    print("共花費:", processing_time)
    print("左腳iou:", iou_left)
    print("右腳iou:", iou_right)
    os.remove(filename)
    
    return jsonify({
        "success": True,
        "iou_left": round(iou_left, 4),
        "iou_right": round(iou_right, 4)
    })

def calculate_angle(p1, p2, p3):
    """計算三點間的角度"""
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = a[0] * b[0] + a[1] * b[1]
    len_a, len_b = math.sqrt(a[0]**2 + a[1]**2), math.sqrt(b[0]**2 + b[1]**2)
    if len_a == 0 or len_b == 0:
        return None
    cos_theta = dot_product / (len_a * len_b)
    return math.degrees(math.acos(max(-1, min(1, cos_theta))))
def draw_keypoints(test_img, keypoints, left_hva, right_hva):
    """在圖像上繪製關鍵點並標註 HVA 角度"""
    image_cv = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
    
    # 繪製關鍵點和編號
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(image_cv, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image_cv, str(i+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # 文字樣式參數
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8 
    thickness = 5    
    text_color = (0, 0, 255) 
    line_spacing = 60 
    
    start_x = 10  
    start_y = 50  
    if left_hva is not None:
        cv2.putText(image_cv, f"Left HVA: {left_hva:.2f}", (start_x, start_y), 
                    font, font_scale, text_color, thickness)
    
    if right_hva is not None:
        cv2.putText(image_cv, f"Right HVA: {right_hva:.2f}", (start_x, start_y + line_spacing), 
                    font, font_scale, text_color, thickness)
    
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

@app.route('/hallux', methods=['GET', 'POST'])
def hallux():
    """處理姆趾外翻圖片，包括去背與關鍵點分析"""
    def resize_img(image, target_sz, divisor=1):
        """根據目標尺寸調整圖片大小"""
        width, height = image.size
        if width < height:
            new_width, new_height = target_sz, int(height * (target_sz / width))
        else:
            new_height, new_width = target_sz, int(width * (target_sz / height))
        return image.resize((new_width, new_height), Image.BILINEAR)
    
    if request.method == 'POST':
        stage = request.form.get('stage', 'removebg')
        
        try:
            if stage == 'removebg':
                # 第一階段：去背處理
                if 'file' not in request.files:
                    return jsonify({"error": "未上傳檔案", "success": False}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({"error": "未選擇檔案", "success": False}), 400
                
                unique_filename = f"hallux_{uuid.uuid4().hex[:8]}.png"
                filename = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(filename)
                
                img = Image.open(filename)
                img = ImageOps.exif_transpose(img)
                
                # 使用 YOLO 模型進行去背
                results = hallux_yolo_model.predict(filename)
                result = results[0]
                masks, boxes = result.masks, result.boxes
                if not masks:
                    return jsonify({"error": "未偵測到腳部", "success": False}), 400
                
                for i, mask in enumerate(masks):
                    if boxes[i].conf.item() <= 0.95:
                        return jsonify({"error": "去背結果不佳，請保持背景乾淨並重新拍照", "success": False}), 400
                
                image_rgba = img.convert("RGBA")
                transparent = Image.new("RGBA", img.size, (0, 0, 0, 0))
                for mask in masks:
                    mask_data = mask.data[0].numpy()
                    mask_img = Image.fromarray((mask_data * 255).astype(np.uint8)).resize(img.size)
                    transparent = Image.composite(image_rgba, transparent, mask_img)

                # 保存去背結果到 FOOT_REMOVEBG_FOLDER
                background_removed_filename = f"background_removed_{uuid.uuid4().hex[:8]}.png"
                background_removed_path = os.path.join(FOOT_REMOVEBG_FOLDER, background_removed_filename)
                transparent.save(background_removed_path, format="PNG")
                image_url = url_for('get_foot_removebg_file', filename=background_removed_filename, _external=True)
                print("full_url", image_url)
                
                return jsonify({
                    "success": True,
                    "image_url": image_url,
                    "stage": "removebg_done"
                })
            
            elif stage == 'analyze':
                # 第二階段：關鍵點分析
                background_removed_filename = request.form.get('bg_removed_filename')
                background_removed_path = os.path.join(FOOT_REMOVEBG_FOLDER, background_removed_filename)
                if not os.path.exists(background_removed_path):
                    return jsonify({"error": "找不到去背後的圖片", "success": False}), 400
                
                test_img = Image.open(background_removed_path).convert('RGB')
                test_sz = 512
                input_img = resize_img(test_img, target_sz=test_sz)
                min_img_scale = min(test_img.size) / min(input_img.size)
                input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None] / 255
                model_output = session.run(None, {"input": input_tensor_np})
                
                # 過濾關鍵點並縮放回原始尺寸
                conf_threshold = 0.8
                scores_mask = model_output[2] > conf_threshold
                predicted_keypoints = (model_output[3][scores_mask])[:, :, :-1].reshape(-1, 2) * min_img_scale
                if len(predicted_keypoints) < 6:
                    return jsonify({"error": "關鍵點數量不足，無法計算 HVA！", "success": False}), 400
                
                # 計算左右腳 HVA 角度
                predicted_keypoints = predicted_keypoints[:6]
                p1, p2, p3 = predicted_keypoints[0], predicted_keypoints[1], predicted_keypoints[2]
                p4, p5, p6 = predicted_keypoints[3], predicted_keypoints[4], predicted_keypoints[5]
                left_angle = calculate_angle(p1, p2, p3)
                right_angle = calculate_angle(p4, p5, p6)
                left_hva_angle = 180 - left_angle if left_angle else 0
                right_hva_angle = 180 - right_angle if right_angle else 0
                
                # 判斷嚴重程度與顏色
                left_severity, left_color = get_severity_and_color(left_hva_angle)
                right_severity, right_color = get_severity_and_color(right_hva_angle)
                
                # 繪製關鍵點並保存
                annotated_image = draw_keypoints(test_img, predicted_keypoints,left_hva_angle, right_hva_angle)
                output_filename = f"hallux_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join(HALLUX_OUTPUT_FOLDER, output_filename)
                annotated_image.save(output_path)
                
                return jsonify({
                    "success": True,
                    "image_url": url_for('get_hallux_file', filename=output_filename, _external=True),
                    "result": {
                        "left_hva_angle": round(left_hva_angle, 2) if left_hva_angle else None,
                        "right_hva_angle": round(right_hva_angle, 2) if right_hva_angle else None,
                        "left_severity": left_severity,
                        "right_severity": right_severity,
                        "left_color": left_color,
                        "right_color": right_color
                    }
                })
            
            return jsonify({"error": "無效的階段", "success": False}), 400
        
        except Exception as e:
            return jsonify({"error": f"處理過程中發生錯誤: {str(e)}", "success": False}), 500
        
        finally:
            if 'filename' in locals() and os.path.exists(filename):
                os.remove(filename)
    
    return render_template("hallux.html")

def get_severity_and_color(angle):
    """根據角度判斷姆趾外翻嚴重程度與對應顏色"""
    if angle is None or angle == 0:
        return "未知", "gray"
    elif angle < 15:
        return "正常", "green"
    elif 15 <= angle <= 20:
        return "輕微", "yellow"
    elif 20 < angle <= 35:
        return "中度", "orange"
    return "嚴重", "red"

@app.route('/health')
def health():
    return render_template('health.html', image_url=None, result_data=None)

if __name__ == "__main__":
    """啟動 Flask 應用程式"""
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)