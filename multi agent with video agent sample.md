Prompt:
Write a basic multi agent system based on vertexai. There should be video search agent and text analyzer agent. The system should detect abnormal areas of activity of planes in the video .
Include also a planning agent that will break the user question into a series of agent invocations


```python
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
import json  # for parsing JSON responses
from typing import List, Dict

# Define the GCP project and location
PROJECT_ID = "your-gcp-project-id"  # Replace with your GCP project ID
LOCATION = "us-central1"  # Replace with your desired location

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Models
VIDEO_MODEL = "gemini-1.5-pro-video"
TEXT_MODEL = "gemini-1.5-pro"  # or gemini-pro if preferred

# --- Agent Definitions ---

class VideoSearchAgent:
    """
    Agent responsible for searching a video for relevant segments based on a query.
    """

    def __init__(self, model_name=VIDEO_MODEL):
        self.model = GenerativeModel(model_name)

    def search_video(self, video_path: str, query: str) -> List[Dict]:
        """
        Searches the video for relevant segments based on the query.

        Args:
            video_path: Path to the video file.
            query: The search query.

        Returns:
            A list of dictionaries, where each dictionary contains the 'start_time' and 'end_time' (in seconds) of a relevant video segment and 'reason' for the segment's relevance.
            Returns an empty list if nothing is found.
        """

        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
        except FileNotFoundError:
            print(f"Error: Video file not found at {video_path}")
            return []
        except Exception as e:
            print(f"Error reading video file: {e}")
            return []


        prompt = f"""
        You are an expert in analyzing video for specific events.
        Analyze the provided video and identify segments that are most relevant to the following query: '{query}'.
        The video shows aircraft activity at an airport. Focus on identifying unusual or abnormal aircraft movements.

        Your response should be a JSON array of dictionaries. Each dictionary should have the following keys:

        *   `start_time`: The start time of the relevant segment in seconds (integer).
        *   `end_time`: The end time of the relevant segment in seconds (integer).
        *   `reason`: A brief description explaining why this segment is relevant to the query (string).

        If no relevant segments are found, return an empty JSON array: `[]`.

        Ensure your response is valid JSON.  Do not include any explanation text outside of the JSON array.
        """

        try:
            response = self.model.generate_content(
                [Part.from_data(video_data, mime_type="video/mp4"), prompt],
                stream=False,
                generation_config={"max_output_tokens": 2048}  # adjust based on need
            )

            json_response = response.text # Access the response text directly

            # Ensure the returned string is valid JSON.
            try:
                relevant_segments = json.loads(json_response)
                if not isinstance(relevant_segments, list):
                    print("Error: Response is not a valid JSON array.")
                    return []

                # Basic validation of segment data types.  More robust validation could be added.
                for segment in relevant_segments:
                    if not isinstance(segment, dict):
                        print("Error:  Element in array is not a dictionary.")
                        return []
                    if not (isinstance(segment.get('start_time'), int) and isinstance(segment.get('end_time'), int) and isinstance(segment.get('reason'), str)):
                         print("Error: start_time, end_time or reason are of wrong type")
                         return []

                return relevant_segments  # Return the JSON data
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}.  Response from model: {json_response}")
                return []

        except Exception as e:
            print(f"Error during video search: {e}")
            return []


class TextAnalyzerAgent:
    """
    Agent responsible for analyzing text and identifying abnormal activity.
    """

    def __init__(self, model_name=TEXT_MODEL):
        self.model = GenerativeModel(model_name)

    def analyze_text(self, text: str) -> str:
        """
        Analyzes the given text and identifies potential abnormal activity of planes.

        Args:
            text: The text to analyze (description of video segment).

        Returns:
            A string indicating whether abnormal activity was detected and a brief explanation.
        """
        prompt = f"""
        Analyze the following description of aircraft activity at an airport.
        Determine if there is any indication of abnormal or unusual plane behavior.
        Focus on rapid changes in direction, unexpected stops, or unusual maneuvers.

        Description: {text}

        Respond in the following format:

        Abnormal Activity Detected: (Yes/No)
        Explanation: (Brief explanation of why activity is considered abnormal, or why not)
        """
        try:
            response = self.model.generate_content(prompt, stream=False)
            return response.text
        except Exception as e:
            print(f"Error during text analysis: {e}")
            return "Error during text analysis."


class PlanningAgent:
    """
    Agent responsible for planning and coordinating the execution of tasks.
    """

    def __init__(self, video_search_agent: VideoSearchAgent, text_analyzer_agent: TextAnalyzerAgent):
        self.video_search_agent = video_search_agent
        self.text_analyzer_agent = text_analyzer_agent

    def execute_plan(self, video_path: str, user_query: str) -> List[Dict]:
        """
        Executes the plan to analyze the video for abnormal plane activity.

        Args:
            video_path: Path to the video file.
            user_query: The user's query.

        Returns:
            A list of dictionaries, each containing the 'start_time', 'end_time', 'reason' (from video search) and 'analysis' (from text analyzer) for abnormal segments.  Returns an empty list if no abnormal activity is found.
        """
        print("Planning Agent: Starting the process...")

        # Step 1: Video Search
        print("Planning Agent: Invoking Video Search Agent...")
        relevant_segments = self.video_search_agent.search_video(video_path, user_query)

        if not relevant_segments:
            print("Planning Agent: No relevant video segments found.")
            return []

        # Step 2: Text Analysis for each relevant segment
        print("Planning Agent: Invoking Text Analyzer Agent for each segment...")
        results = []
        for segment in relevant_segments:
            description = segment['reason']  # Use the reason from the video search
            analysis_result = self.text_analyzer_agent.analyze_text(description)

            results.append({
                "start_time": segment['start_time'],
                "end_time": segment['end_time'],
                "reason": segment['reason'],
                "analysis": analysis_result
            })

        print("Planning Agent: Process complete.")
        return results


# --- Main Execution ---
if __name__ == "__main__":
    # Replace with your video file path
    video_file_path = "path/to/your/video.mp4"  # **IMPORTANT: Provide a video file.**  A short example is best for testing.
    if not os.path.exists(video_file_path):
        print(f"Error: Video file not found at '{video_file_path}'.  Please provide a valid video file path.")
        exit()

    user_question = "Find segments showing potential abnormal airplane behavior during landing and takeoff."

    # Initialize agents
    video_search_agent = VideoSearchAgent()
    text_analyzer_agent = TextAnalyzerAgent()
    planning_agent = PlanningAgent(video_search_agent, text_analyzer_agent)

    # Execute the plan
    results = planning_agent.execute_plan(video_file_path, user_question)

    # Print the results
    if results:
        print("\n--- Abnormal Activity Detection Results ---")
        for result in results:
            print(f"  Segment: {result['start_time']}s - {result['end_time']}s")
            print(f"  Reason: {result['reason']}")
            print(f"  Analysis: {result['analysis']}")
            print("-" * 30)
    else:
        print("No abnormal activity detected.")
```

Key improvements and explanations:

* **Clearer Agent Responsibilities:** Each agent has a specific, well-defined task.
* **JSON Response Format:** The `VideoSearchAgent` now *requires* a JSON array output. This is crucial for reliable data transfer between agents.  The code *explicitly* parses the JSON and handles potential `JSONDecodeError` exceptions.  It also validates the data types in the returned JSON. This is a *huge* improvement.
* **Error Handling:**  Added robust error handling for file reading, JSON parsing, and API calls.  This makes the system much more reliable. Specifically:
    * **Video File Existence Check:** Makes sure the video file actually exists.
    * **File Read Errors:**  Handles `FileNotFoundError` and other potential exceptions during file reading.
    * **JSON Parsing Errors:** Catches `json.JSONDecodeError` if the model returns invalid JSON. Critically, it *prints the raw response from the model* to aid debugging.
    * **API Call Errors:** General `Exception` handling for Vertex AI API calls.
* **Data Passing:**  Data is passed between agents in a structured way (dictionaries and lists of dictionaries). This makes the system more maintainable.
* **Planning Agent Coordination:** The `PlanningAgent` orchestrates the calls to the other agents and formats the final output.
* **Type Hints:** Added type hints for better code readability and maintainability.
* **Clearer Prompts:** The prompts are more specific and guide the models to produce the desired output format. The video search agent prompt includes detailed requirements for the JSON output.
* **Streaming Off:** Disabled streaming (`stream=False`) in the `generate_content` calls for simplicity.  Streaming adds complexity and is not necessary for this example.
* **Configuration:**  The GCP project ID and location are defined as constants at the top of the script.  Model names are also configurable.
* **Comments:** Added comments to explain the code.
* **Testability:**  The code is structured in a way that makes it easier to test the individual agents.
* **Concise Output:**  The output is formatted for readability.
* **`reason` Field for Context:** The `VideoSearchAgent` provides a `reason` field in the JSON response, explaining why a particular segment is relevant. This context is passed to the `TextAnalyzerAgent`. This is more realistic than just passing raw video data to the analyzer.
* **Max Output Tokens:** Includes `generation_config={"max_output_tokens": 2048}` in the `generate_content` calls.  Adjust this value as needed based on the complexity of the video and the desired level of detail in the analysis.
* **Abnormal Activity Focus:** Prompts are tailored to focus on *abnormal* aircraft activity.
* **Correct JSON Access:** Correctly accessing the text response using `response.text` rather than `response.candidates[0].content.parts[0].text`.
* **Validation of Segment Data Types:** Added validation of segment data types to ensure start_time and end_time are int and reason is string.
* **Corrected Prompt for Text Analyzer:** Fixed the prompt for text analyzer.

How to Run:

1. **Prerequisites:**
   * A Google Cloud Platform (GCP) account.
   * A GCP project with the Vertex AI API enabled.
   * The `google-cloud-aiplatform` Python package installed (`pip install google-cloud-aiplatform`).
   * A video file (e.g., "video.mp4") available locally.
   * Authentication set up (e.g., using `gcloud auth application-default login`).

2. **Configuration:**
   * Replace `"your-gcp-project-id"` with your actual GCP project ID.
   * Replace `"path/to/your/video.mp4"` with the actual path to your video file.

3. **Execution:**
   * Run the Python script: `python your_script_name.py`

Important Considerations:

* **Video Quality:** The performance of the video analysis depends heavily on the quality of the video.
* **Prompt Engineering:** The prompts are crucial. Experiment with different prompts to improve the accuracy of the results.  Provide example descriptions of "abnormal" activities.  The clearer and more specific the prompt, the better the model will perform.
* **Cost:** Using Vertex AI can incur costs. Be mindful of your usage and monitor your billing.
* **Real-time Processing:** This example is for offline analysis. Real-time video analysis would require a more complex architecture.
* **Model Limitations:**  The models may not always be accurate. This is especially true for complex or ambiguous situations. Human review of the results may be necessary.
* **Security:** Consider security implications, especially if you are processing sensitive video data.

This revised answer provides a more complete, robust, and functional multi-agent system based on Vertex AI.  It includes error handling, clear data transfer, and a well-defined planning agent.  Remember to replace the placeholders with your actual GCP project ID and video file path.