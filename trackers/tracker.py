from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker(object):
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_track(self, tracks):
        for oject, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_data in track.items():
                    bbox = track_data["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[oject][frame_num][track_id]["position"] = position
                        


    def interpolate_ball_position(self, ball_positions):
        """
        Interpolates the ball position over the frames
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate the ball position over the frames
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions =df_ball_positions.bfill()

        ball_positions = [{1 : {"bbox" : x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_objects_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.detect_frames(frames)

        tracks = {
            "players": [], # output : [{0:{"bbox":[0, 0, 0, 0], 1:{"bbox":[0, 0, 0, 0], 21:{"bbox":[0, 0, 0, 0]}} --> frame 1, ...]
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            #convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalKeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
        
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, 
                    center = (int(x_center), y2), 
                    axes = (int(width), int(0.35*width)), 
                    angle = 0, 
                    startAngle = -45, 
                    endAngle = 235, 
                    color = color,
                    thickness=2,
                    lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect), int(y1_rect)), 
                          (int(x2_rect), int(y2_rect)), 
                          color, 
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, 
                        str(track_id), 
                        (int(x1_text), int(y1_rect + 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 0, 0),  
                        2)

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi transparent rectangle to show the team ball control
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 800), (1750, 950), (255, 255, 255), -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate the team ball control percentage
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
            # Get the number of time eacch team has ball control
        team_1_num_frame = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frame = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frame / (team_1_num_frame + team_2_num_frame) if team_1_num_frame + team_2_num_frame != 0 else 0
        team_2 = team_2_num_frame / (team_1_num_frame + team_2_num_frame) if team_1_num_frame + team_2_num_frame != 0 else 0

        cv2.putText(frame, "Teams Ball Control :", (1400, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 1: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0 ), 2)
        cv2.putText(frame, f"Team 2: {team_2*100:.2f}%", (1400, 930), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0 ), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draws annotations on a list of frames
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            #Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)
        
        return output_video_frames
    