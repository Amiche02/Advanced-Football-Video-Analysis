import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from utils import get_center_of_bbox, get_bbox_width, mesure_distance

class PlayerBallAssigner(object):
    def __init__(self):
        self.max_player_ball_distance = 40


    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 9999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = mesure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = mesure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

        # print(player_id, distance)
            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

                if distance < self.max_player_ball_distance:
                    if distance < minimum_distance:
                        minimum_distance = distance
                        assigned_player = player_id

        return assigned_player