import os
import json
import warnings
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tqdm import tqdm

# 过滤无关警告
warnings.filterwarnings("ignore", message="No ccache found.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FrameAnalyzer:
    def __init__(self):
        """初始化模型（直接处理已截取的帧图片）"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本目录

        # 加载YOLO模型（模型放脚本同目录）
        yolo_model_path = os.path.join(self.base_dir, "yolov8m.pt")
        self.yolo_model = YOLO(yolo_model_path)

        # 初始化PaddleOCR（适配版本）
        try:
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                lang="ch",
                show_log=False
            )
        except ValueError:
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                lang="ch"
            )

        print("模型初始化完成，准备处理帧图片...\n")

    def analyze_single_frame(self, frame_path):
        """分析单张帧图片（识别物品+文字）"""
        interested_classes = {
            "bottle", "cup", "cell phone", "book", "laptop",
            "tv", "keyboard", "mouse", "remote", "scissors",
            "teddy bear", "hair drier", "toothbrush", "backpack",
            "umbrella", "handbag", "tie", "suitcase"
        }

        result = {
            "frame_path": os.path.relpath(frame_path, self.base_dir),  # 相对路径
            "objects": [],
            "status": "success"
        }

        # 1. 物品识别（YOLO）
        try:
            yolo_results = self.yolo_model(frame_path, conf=0.5, verbose=False)
            objects = []
            for res in yolo_results:
                class_names = [
                    res.names[int(cls)]
                    for cls in res.boxes.cls.tolist()
                    if res.names[int(cls)] in interested_classes
                ]
                objects.extend(class_names)
            result["objects"] = list(set(objects))  # 去重
        except Exception as e:
            result["status"] = f"error: {str(e)}"
            return result

        # 2. 文字识别（PaddleOCR，可选保留）
        # if you need OCR, uncomment below
        # try:
        #     ocr_results = self.ocr.ocr(frame_path)
        #     texts = [line[1][0] for line in ocr_results[0] if len(line[1][0].strip()) > 1]
        #     result["texts"] = texts
        # except Exception as e:
        #     result["status"] += f" | OCR失败: {str(e)}"

        return result

    def process_frame_folder(self, frame_folder_rel="video_processed_20250817_1226",
                             output_root_rel="analysis_results"):
        """处理已截取好的帧图片文件夹"""
        # 转换为绝对路径
        frame_folder = os.path.join(self.base_dir, frame_folder_rel)

        # 检查文件夹是否存在
        if not os.path.exists(frame_folder):
            raise FileNotFoundError(
                f"帧图片文件夹不存在（相对路径：{frame_folder_rel}）\n"
                f"绝对路径：{frame_folder}"
            )

        # 结果输出目录（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_rel = os.path.join(output_root_rel, f"results_{timestamp}")
        output_dir = os.path.join(self.base_dir, output_dir_rel)
        os.makedirs(output_dir, exist_ok=True)
        print(f"结果将保存至（相对路径）：{output_dir_rel}\n")

        # 获取所有帧图片（只处理.jpg格式）
        frame_extensions = (".jpg", ".jpeg", ".png")  # 常见图片格式
        frame_files = [
            os.path.join(frame_folder, f)
            for f in os.listdir(frame_folder)
            if f.lower().endswith(frame_extensions)
        ]

        if not frame_files:
            print(f"警告：在 {frame_folder_rel} 中未找到帧图片")
            return

        # 批量分析帧图片
        all_results = []
        for frame_path in tqdm(frame_files, desc="分析帧图片"):
            res = self.analyze_single_frame(frame_path)
            all_results.append(res)

        # 保存结果到JSON
        self.save_results(all_results, output_dir_rel, output_dir)

    def save_results(self, all_results, output_dir_rel, output_dir):
        """保存识别结果到JSON文件"""
        # 控制台输出汇总
        print("\n" + "=" * 50)
        print("最终识别结果汇总")
        print("=" * 50)
        for res in all_results:
            frame_name = os.path.basename(res["frame_path"])
            objects = res["objects"] if res["objects"] else "无"
            print(f"帧 {frame_name}：识别到物品 -> {objects}")

        # 保存到JSON
        json_rel_path = os.path.join(output_dir_rel, "frame_analysis_results.json")
        json_path = os.path.join(self.base_dir, json_rel_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存为JSON文件（相对路径）：{json_rel_path}")


if __name__ == "__main__":
    # 直接处理已截取的帧图片文件夹
    analyzer = FrameAnalyzer()
    analyzer.process_frame_folder(
        frame_folder_rel="video_processed_20250817_1226",  # 帧图片文件夹相对路径
        output_root_rel="analysis_results"  # 结果输出目录
    )