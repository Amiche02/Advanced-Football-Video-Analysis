import cv2

def read_video(video_path):
    """
    Reads a video file and returns a list of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Saves a list of frames to a video file
    """
    if not output_video_frames:
        return

    # Get the width and height from the first frame
    width, height = output_video_frames[0].shape[1], output_video_frames[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()