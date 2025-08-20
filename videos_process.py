import cv2
from pydub import AudioSegment
import os
from datetime import datetime


def process_video(video_path, output_dir=f"video_processed_{datetime.now().strftime('%Y%m%d_%H%M')}", a=10):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 提取视频帧（每a秒抽1帧）
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_interval = int(fps * a)  # 默认每10秒1帧
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 每隔指定帧保存1张
        if frame_count % frame_interval == 0:
            frame_path = f"{output_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        frame_count += 1
    cap.release()
    print(f"提取了 {len(saved_frames)} 张关键帧")

    # 2. 提取音频（保存为wav格式，供后续语音识别）
    audio_path = f"{output_dir}/audio.wav"
    # 先将视频转成临时音频（需ffmpeg）
    AudioSegment.from_file(video_path).export(audio_path, format="wav")
    print(f"音频提取完成：{audio_path}")

    return saved_frames, audio_path


if __name__ == '__main__':
    video_path = './videos/test.mp4'
    print('开始处理')
    process_video(video_path, a=10)
    print('处理成功')
