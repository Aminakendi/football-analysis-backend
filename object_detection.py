import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from team_functions import (
    get_jersey_color
)

def draw_blue_triangle(frame, bbox):
    """Draw a blue triangle to represent the ball on the frame with professional styling."""
    x1, y1, x2, y2 = bbox
    # Points for the triangle (using the center of the bbox)
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    points = np.array([(center_x, center_y - 20), (center_x - 20, center_y + 20), (center_x + 20, center_y + 20)])
    
    # Draw a filled triangle with semi-transparent blue
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color=(255, 0, 0))  # Blue fill
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)  # Outline

def draw_possession_stats(frame, team1_percent, team2_percent):
    """Display possession stats with a professional design."""
    height, width, _ = frame.shape
    overlay = frame.copy()
    bar_width = 500
    bar_height = 30
    bar_x = (width - bar_width) // 2
    bar_y = 10
    
    # Draw a semi-transparent background
    cv2.rectangle(overlay, (bar_x - 10, bar_y - 10), (bar_x + bar_width + 10, bar_y + bar_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw possession text
    text = f"Team 1: {team1_percent:.1f}% | Team 2: {team2_percent:.1f}%"
    cv2.putText(frame, text, (bar_x, bar_y + bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def annotate_player(frame, bbox, label, team_color):
    """Annotate players with professional styling."""
    x1, y1, x2, y2 = map(int, bbox)
    overlay = frame.copy()
    # Draw semi-transparent rectangle for the bounding box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), team_color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    # Draw bounding box outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
    # Add text label with a shadow
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_x, text_y = x1, y1 - 10
    shadow_offset = 2
    cv2.putText(frame, label, (text_x + shadow_offset, text_y + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_yolo_video_with_teams(model_path, video_path, output_path, club1, club2):
    model = YOLO(model_path)
    tracker = sv.ByteTrack(lost_track_buffer=30)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    possession_team1 = 0  # Time the ball is close to Team 1
    possession_team2 = 0  # Time the ball is close to Team 2
    total_frames = 0      # Count total frames processed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Convert frame for YOLO and run inference
        batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        results = model(batch)

        # Get the results
        result = results[0]

        # Build detections for tracker
        bboxes, confidences, class_ids = [], [], []
        for detection in result.boxes:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]

            # Exclude referee (class_id == 3)
            if class_id != 3 and confidence >= 0.25:
                x1, y1, x2, y2 = detection.xyxy[0]
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Create detection object
        detections = sv.Detections(xyxy=np.array(bboxes), confidence=np.array(confidences), class_id=np.array(class_ids))

        # Update tracker
        tracks = tracker.update_with_detections(detections)

        ball_position = None
        players = []

        # Annotate frame with team colors and blue triangle for the ball
        for track in tracks:
            bbox = track[0]  # [x1, y1, x2, y2]
            track_id = track[4]  # Track ID (correct index)
            class_id = track[3]  # Get class_id

            is_goalkeeper = (class_id == 1)  # Assuming class_id 1 is goalkeeper; adjust as needed

            if class_id == 0:  # Ball class (ID = 0)
                # Draw blue triangle for the ball
                draw_blue_triangle(frame, bbox)
                ball_position = bbox
            elif class_id != 3:  # Skip annotating referees (class_id == 3)
                # Get the jersey color using the provided functions
                jersey_color = get_jersey_color(frame, bbox, is_goalkeeper=is_goalkeeper)

                # Determine the club using the jersey color
                distances_to_club1 = np.linalg.norm(np.array(jersey_color) - np.array(club1.player_jersey_color))
                distances_to_club2 = np.linalg.norm(np.array(jersey_color) - np.array(club2.player_jersey_color))

                team_id = 0 if distances_to_club1 < distances_to_club2 else 1
                team_color = club1.player_jersey_color if team_id == 0 else club2.player_jersey_color

                # Assign a color for the team and annotate
                team_color = tuple(int(c) for c in team_color)  # Convert to int (BGR format)

                # Add tracker ID to the label
                label = f"T{team_id + 1}-ID:{track_id}"  # Label the player by team number and tracker ID
                annotate_player(frame, bbox, label, team_color)

                # Collect player information for possession calculation
                players.append((bbox, team_id))

        # Calculate possession
        if ball_position is not None:
            ball_x, ball_y = (ball_position[0] + ball_position[2]) / 2, (ball_position[1] + ball_position[3]) / 2
            closest_team = None
            min_distance = float('inf')

            for player_bbox, team_id in players:
                player_x, player_y = (player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2
                distance = np.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_team = team_id

            if closest_team == 0:
                possession_team1 += 1
            elif closest_team == 1:
                possession_team2 += 1

        # Display possession percentages
        total_possession = possession_team1 + possession_team2
        if total_possession > 0:
            possession_percent_team1 = (possession_team1 / total_possession) * 100
            possession_percent_team2 = (possession_team2 / total_possession) * 100
        else:
            possession_percent_team1 = possession_percent_team2 = 0

        draw_possession_stats(frame, possession_percent_team1, possession_percent_team2)

        # Write annotated frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from team_functions import Club

    # Define the two clubs
    club1 = Club(name="Team1", player_jersey_color=(232, 247, 248), goalkeeper_jersey_color=(6, 25, 21))
    club2 = Club(name="Team2", player_jersey_color=(172, 251, 145), goalkeeper_jersey_color=(239, 156, 132))

    process_yolo_video_with_teams(
        model_path='models/object.pt',
        video_path='input_video/08fd33_4 - Copy (2).mp4',
        output_path='output_video/teams_tracked_possession_bar.mp4',
        club1=club1,
        club2=club2
    )
