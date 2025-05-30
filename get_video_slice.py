import argparse
import os

import cv2
import torch
from deepface import DeepFace
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos based on emotions")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to video folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--frame_interval", type=int, default=150, help="Process every N frames")
    parser.add_argument("--detector", type=str, default="retinaface", help="Face detector backend")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [f for f in os.listdir(args.video_folder) if f.lower().endswith(video_extensions)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    total_saved_frames = 0

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        saved_frames = 0

        pbar = tqdm(total=total_frames, desc=f"{video_name}", leave=False)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % args.frame_interval == 0:
                try:
                    result = DeepFace.analyze(
                        frame, actions=["emotion"], enforce_detection=False, detector_backend=args.detector
                    )
                    emotion = result[0]["dominant_emotion"]

                    emotion_folder = os.path.join(args.output_dir, video_name, emotion)
                    os.makedirs(emotion_folder, exist_ok=True)

                    frame_filename = os.path.join(emotion_folder, f"frame_{frame_index}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_frames += 1
                    pbar.set_postfix(saved=saved_frames, emotion=emotion)

                except Exception:
                    pbar.set_postfix(saved=saved_frames, status="detection_failed")

            frame_index += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        total_saved_frames += saved_frames

    print(f"\nDone. Total frames saved: {total_saved_frames}")


if __name__ == "__main__":
    main()
