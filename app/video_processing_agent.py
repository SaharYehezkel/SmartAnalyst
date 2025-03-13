import cv2
import numpy as np
import os
import vertexai
from vertexai.language_models import ChatModel
import google.generativeai as genai


class VideoProcessingAgent:
    def __init__(self):
        vertexai.init(project="txt2api", location="")
        self.model = ChatModel.from_pretrained("gemini")
        self.chat = self.model.start_chat()

    def extract_frame(self, video_path, time_stamp, output_image):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None
        
        try:
            minutes, seconds = map(int, time_stamp.split(':'))
            frame_time = (minutes * 60 + seconds) * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
        except ValueError:
            print("Error: Invalid timestamp format. Expected 'MM:SS'.")
            return None
        
        success, frame = cap.read()
        cap.release()
        
        if success:
            cv2.imwrite(output_image, frame)
            return output_image
        else:
            print("Error: Could not extract frame.")
            return None

    def process_video(self, video_path, time_stamp):
        output_image = "extracted_frame.png"
        frame_path = self.extract_frame(video_path, time_stamp, output_image)
        
        if frame_path is None:
            return None
        
        image = cv2.imread(frame_path)
        if image is None:
            print("Error loading image")
            return None
        
        vertices, edges = extract_purple_polygon_edges(image)
        if vertices is not None:
            print("Detected polygon vertices:", vertices)
            pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
            
            cropped = crop_to_polygon(image, vertices)
            cropped_output_path = "cropped_polygon_area.jpg"
            cv2.imwrite(cropped_output_path, cropped)
            print(f"Cropped image saved as: {cropped_output_path}")
            
            return self.analyze_cropped_image(cropped_output_path)
        else:
            print("No suitable purple polygon detected.")
            return None

    def analyze_cropped_image(self, image_path):
        prompt = "Analyze this image and count how many yellow airplane icons are inside the purple polygon. Return only the count."
        response = self.chat.send_message(prompt)
        
        try:
            count = int(response.text.strip())
            return f"Number of yellow airplane icons detected: {count}"
        except ValueError:
            return "Error: Could not determine the number of yellow airplane icons."

    def analyze_query_and_process(self, video_path, query):
        prompt = f"Extract the timestamp from this query: {query}. Return only the timestamp in 'MM:SS' format."
        response = self.chat.send_message(prompt)
        
        try:
            time_stamp = response.text.strip()
            return self.process_video(video_path, time_stamp)
        except Exception as e:
            print("Error processing query response:", e)
            return None

if __name__ == '__main__':
    agent = VideoProcessingAgent()
    video_path = input("data\\videos\\1.mp4")
    user_query = input("""Do you recognize any purple polygon in the video? if the answer is yes,
                       give me the time in the video. if the time is a range return the last time in the range.""")
    result = agent.analyze_query_and_process(video_path, user_query)
    if result:
        print(result)
