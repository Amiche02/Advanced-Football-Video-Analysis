from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    # read the video
    model_path ='models/Soccer_YoloV5/best.pt'
    input_video_path = 'input_videos/08fd33_4.mp4'
    output_video_path = 'output_videos/output.avi'
    video_frames = read_video(input_video_path)

    #Initialize the tracker
    tracker = Tracker(model_path=model_path)
    tracks = tracker.get_objects_tracks(video_frames, read_from_stub=True,
                                        stub_path="stubs/track_stubs1.pkl")
    
    # Get objects positions
    tracker.add_position_to_track(tracks)
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path="stubs/camera_movement_stubs1.pkl")
    # Add the ajusted position to the tracks
    camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate the ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_track(tracks)

    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Assign Ball to players and team
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        elif len(team_ball_control) > 0: 
            team_ball_control.append(team_ball_control[-1])
        else:
            team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)


    # Draw output
    ## Draw objects tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)


    # save the video
    save_video(output_video_frames, output_video_path)


if __name__ == "__main__":
    main()