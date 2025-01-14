from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans

@dataclass
class Club:
    """
    A class to represent a football team.

    Attributes:
        name (str): The name of the club.
        player_jersey_color (Tuple[int, int, int]): The jersey color of the players in RGB format.
        goalkeeper_jersey_color (Tuple[int, int, int]): The jersey color of the goalkeeper in RGB format.
    """
    name: str
    player_jersey_color: Tuple[int, int, int]
    goalkeeper_jersey_color: Tuple[int, int, int]



def predict_club(model: Dict[str, np.ndarray], extracted_color: Tuple[int, int, int], is_goalkeeper: bool = False) -> int:
    """
    Predict the club for a given jersey color based on the centroids.

    Args:
        model (Dict[str, np.ndarray]): The centroids for club jerseys.
        extracted_color (Tuple[int, int, int]): The extracted jersey color in BGR format.
        is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

    Returns:
        int: The index of the predicted club (0 or 1).
    """
    centroids = model["goalkeeper_centroids"] if is_goalkeeper else model["player_centroids"]
    distances = np.linalg.norm(extracted_color - centroids, axis=1)
    return np.argmin(distances)

def apply_mask(image: np.ndarray, green_threshold: float = 0.08) -> np.ndarray:
    """
    Apply a mask to an image based on green color in HSV space.

    Args:
        image (np.ndarray): An image to apply the mask to.
        green_threshold (float): Threshold for green color coverage.

    Returns:
        np.ndarray: The masked image.
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    total_pixels = image.shape[0] * image.shape[1]
    masked_pixels = cv2.countNonZero(cv2.bitwise_not(mask))
    mask_percentage = masked_pixels / total_pixels

    if mask_percentage > green_threshold:
        return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    else:
        return image

def perform_clustering(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Perform K-Means clustering to identify the dominant jersey color.

    Args:
        image (np.ndarray): The input image.

    Returns:
        Tuple[int, int, int]: The jersey color in BGR format.
    """
    img_reshape = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(img_reshape)
    labels = kmeans.labels_.reshape(image.shape[:2])

    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    bg_cluster = max(set(corners), key=corners.count)
    player_cluster = 1 - bg_cluster

    jersey_color_bgr = kmeans.cluster_centers_[player_cluster]
    return (int(jersey_color_bgr[2]), int(jersey_color_bgr[1]), int(jersey_color_bgr[0]))

def get_jersey_color(frame: np.ndarray, bbox: Tuple[int, int, int, int], is_goalkeeper: bool = False) -> Tuple[int, int, int]:
    """
    Extract the jersey color from a player's bounding box in the frame.

    Args:
        frame (np.ndarray): The current video frame.
        bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
        is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

    Returns:
        Tuple[int, int, int]: The jersey color in BGR format.
    """
    # Crop the image to the region of interest (bounding box)
    img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    if is_goalkeeper:
        # If the player is a goalkeeper, use the predefined goalkeeper color
        return (int(img[0][0][0]), int(img[0][0][1]), int(img[0][0][2]))  # BGR format
    else:
        # Apply color clustering 
        img_top = img[:img.shape[0] // 2, :]  # Only consider the top part of the cropped image
        masked_img_top = apply_mask(img_top, green_threshold=0.08)  # Mask out green areas (e.g., grass)
        return perform_clustering(masked_img_top)





