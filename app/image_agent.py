import cv2
import numpy as np

# ---------- Helper Functions ----------
def extract_frame(video_path, time_str, output_path="frame.png"):
    """
    Extracts a frame from the video at the specified time and saves it as a PNG file.
    
    Parameters:
        video_path (str): Path to the video file.
        time_str (str): Time in "mm:ss" format.
        output_path (str): Path to save the extracted frame as a PNG file.
        
    Returns:
        bool: True if the frame was successfully extracted and saved, False otherwise.
    """
    # Parse the time string (mm:ss) into total seconds
    try:
        minutes, seconds = map(int, time_str.split(":"))
        target_time = minutes * 60 + seconds
    except Exception as e:
        print("Error parsing time string. Make sure it's in 'mm:ss' format.", e)
        return False

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS is 0.")
        return False

    # Calculate the frame number corresponding to the target time
    frame_number = int(target_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve the frame at the specified time.")
        cap.release()
        return False

    # Save the frame as a PNG file
    cv2.imwrite(output_path, frame)
    print(f"Frame saved successfully as {output_path}")
    cap.release()
    return True
'''
# Example usage:
if __name__ == "__main__":
    video_file = "example_video.mp4"  # Replace with your video file path
    time_stamp = "01:30"              # Example: 1 minute 30 seconds
    output_image = "frame_01_30.png"   # Desired output file name
    extract_frame(video_file, time_stamp, output_image)
'''

def extract_red_polygon(image):
    """
    Detects red regions in the image, finds contours, approximates polygons,
    and returns the vertices of the largest red polygon found.
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color ranges (red wraps in HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return None
    if not contours:
        return None

    # Assume the largest contour by area is the red polygon
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract vertices as (x, y) tuples
    vertices = [tuple(pt[0]) for pt in approx]
    return vertices

def load_yolo_model(cfg_path, weights_path, names_path):
    """
    Loads the YOLO model using OpenCV's DNN module.
    Returns the net and the list of class names.
    """
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load class names from coco.names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def detect_airplanes(image, net, classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect airplane icons in the image using YOLO.
    Returns a list of center coordinates for detections whose class is "airplane" (or "aeroplane").
    """
    # Prepare the image for YOLO
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # Forward pass
    detections = net.forward(out_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Check if detected object is an airplane (adjust class name as needed)
            if confidence > conf_threshold and classes[class_id].lower() in ["airplane", "aeroplane"]:
                # Compute bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    airplane_centers = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            center = (x + w // 2, y + h // 2)
            airplane_centers.append(center)
    return airplane_centers

def point_inside_polygon(point, polygon):
    """
    Determines if a point (x, y) is inside a polygon.
    Uses the ray casting algorithm.
    """
    x, y = point
    num = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(num + 1):
        p2x, p2y = polygon[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def extract_purple_polygon_edges(image):
    # Convert image to HSV color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define an HSV range for purple.
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Use Gaussian blur to reduce noise.
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Use Canny edge detection to capture edges.
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge image.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return None, edges

    # Filter out small contours by area.
    image_area = image.shape[0] * image.shape[1]
    min_area = 0.005 * image_area  # Consider only contours covering at least 0.5% of the image.
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
    else:
        largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon with a smaller epsilon to capture more details.
    arc_length = cv2.arcLength(largest_contour, True)
    epsilon = 0.005 * arc_length  # Lower epsilon preserves more vertices.
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract vertices as (x, y) tuples.
    vertices = [tuple(pt[0]) for pt in approx]
    return vertices, edges

def crop_to_polygon(image, vertices):
    """
    Creates a mask from the polygon vertices, applies it to the image,
    and then crops the image to the polygon's bounding rectangle.
    """
    # Create a blank mask.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
    
    # Fill the polygon on the mask.
    cv2.fillPoly(mask, [pts], 255)
    
    # Apply the mask to the image.
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop to the bounding rectangle of the polygon.
    x, y, w, h = cv2.boundingRect(pts)
    cropped_image = masked_image[y:y+h, x:x+w]
    return cropped_image

def extract_frame(video_path, time_stamp, output_image):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Convert timestamp to milliseconds
    minutes, seconds = map(int, time_stamp.split(':'))
    frame_time = (minutes * 60 + seconds) * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
    
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image, frame)
    else:
        print("Error: Could not extract frame.")
    
    cap.release()

def process_video(video_path, time_stamp, output_image):
    extract_frame(video_path, time_stamp, output_image)
    
    image = cv2.imread(output_image)
    if image is None:
        print("Error loading image")
        return
    
    vertices, edges = extract_purple_polygon_edges(image)
    if vertices is not None:
        print("Detected polygon vertices:", vertices)
        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
        
        cropped = crop_to_polygon(image, vertices)
        
        cropped_output_path = "cropped_polygon_area.jpg"
        cv2.imwrite(cropped_output_path, cropped)
        print(f"Cropped image saved as: {cropped_output_path}")
        
        cv2.imshow("Edges", edges)
        cv2.imshow("Detected Polygon", image)
        cv2.imshow("Cropped Area", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No suitable purple polygon detected.")

if __name__ == '__main__':
    video_file = "bandicam 2025-03-10 11-00-42-951.mp4"
    time_stamp = "00:32"
    output_image = "framed_video.png"
    
    process_video(video_file, time_stamp, output_image)








'''
if __name__ == '__main__':
    
    video_file = "bandicam 2025-03-10 11-00-42-951.mp4"  # Replace with your video file path
    time_stamp = "00:32"              # Example: 1 minute 30 seconds
    output_image = "framed_video.png"   # Desired output file name
    extract_frame(video_file, time_stamp, output_image)
    
    # Replace with your image path.
    image = cv2.imread("framed_video.png")
    if image is None:
        print("Error loading image")
        exit()

    # Extract the polygon vertices.
    vertices, edges = extract_purple_polygon_edges(image)
    if vertices is not None:
        print("Detected polygon vertices:", vertices)
        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
        # Draw the detected polygon on the image for visualization.
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
        
        # Crop the image to the polygon area.
        cropped = crop_to_polygon(image, vertices)
        
        # Save the cropped image to a new file.
        output_path = "cropped_polygon_area.jpg"
        cv2.imwrite(output_path, cropped)
        print(f"Cropped image saved as: {output_path}")
        
        # Display the results.
        cv2.imshow("Edges", edges)
        cv2.imshow("Detected Polygon", image)
        cv2.imshow("Cropped Area", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No suitable purple polygon detected.")

'''
'''
def extract_purple_polygon_edges(image):
    # Convert image to HSV color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define an HSV range for purple.
    # Adjust these values to match your specific shade of purple.
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Reduce noise with Gaussian blur.
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Detect edges using Canny edge detection.
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return None, edges

    # Optionally filter out small contours.
    image_area = image.shape[0] * image.shape[1]
    min_area = 0.005 * image_area  # Only consider contours covering at least 0.5% of the image.
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
    else:
        largest_contour = max(contours, key=cv2.contourArea)

    # Use approxPolyDP with a smaller epsilon to capture more detail.
    arc_length = cv2.arcLength(largest_contour, True)
    epsilon = 0.005 * arc_length  # Lower value preserves more vertices.
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract vertices as (x, y) tuples.
    vertices = [tuple(pt[0]) for pt in approx]
    return vertices, edges

if __name__ == '__main__':
    # Replace with your image path containing the purple polygon.
    image = cv2.imread("polygon_purple.png")
    if image is None:
        print("Error loading image")
        exit()

    vertices, edges = extract_purple_polygon_edges(image)
    if vertices is not None:
        print("Detected polygon vertices:", vertices)
        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
        # Draw the polygon using purple color (BGR: 255, 0, 255).
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
    else:
        print("No polygon detected")

    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Polygon", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
'''
# ---------- Main Pipeline ----------
'''
def main():
    # Load an example image (replace with your image path)
    image_path = "polygon_5_crop.png" #"your_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Step 1: Extract the red polygon and its vertices
    polygon_vertices = extract_red_polygon(image)
    if polygon_vertices is None:
        print("No red polygon detected.")
        return
    print("Red Polygon Vertices:", polygon_vertices)

    # Draw the polygon for visualization
    pts = np.array(polygon_vertices, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Step 2: Load YOLO model for airplane detection
    yolo_cfg = "yolov3.cfg"       # Provide correct path
    yolo_weights = "yolov3.weights"  # Provide correct path
    yolo_names = "coco.names"     # Provide correct path
    net, classes = load_yolo_model(yolo_cfg, yolo_weights, yolo_names)

    # Step 3: Detect airplane icons and get their center coordinates
    airplane_centers = detect_airplanes(image, net, classes)
    print("Detected Airplane Centers:", airplane_centers)

    # Draw circles on detected airplane centers
    for center in airplane_centers:
        cv2.circle(image, center, 5, (255, 0, 0), -1)

    # Step 4: Check which airplane centers fall inside the red polygon
    count_inside = 0
    for center in airplane_centers:
        if point_inside_polygon(center, polygon_vertices):
            count_inside += 1

    print("Number of airplanes inside the red polygon:", count_inside)

    # Show the final image
    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''