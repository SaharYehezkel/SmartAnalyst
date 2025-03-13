import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextGenerationModel
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Configure Vertex AI
PROJECT_ID = os.getenv("PROJECT_ID")  # Replace with your project ID
LOCATION = "us-central1"  # Replace with your location
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize generative models
llm = GenerativeModel("gemini-1.0-pro")
text_generation_model = TextGenerationModel.from_pretrained("text-bison@002")

# --- Agent Definitions ---

class DbSearchAgent:
    """Agent that creates sql query from free text, and uses it to fetch results from db."""

    def __init__(self, llm_model):
        self.llm = llm_model  # Use the shared GenerativeModel
        self.search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not self.search_api_key or not self.search_engine_id:
            raise ValueError(
                "GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID must be set in the environment."
            )

    def search(self, query, num_results=3):
        """Performs a Google search and returns snippets."""
        try:
            from googleapiclient.discovery import build

            service = build("customsearch", "v1", developerKey=self.search_api_key)
            result = (
                service.cse()
                .list(q=query, cx=self.search_engine_id, num=num_results)
                .execute()
            )
            snippets = [item["snippet"] for item in result["items"]]
            return snippets
        except Exception as e:
            print(f"Error during Google Search: {e}")
            return []

    def run(self, query):
        """Runs the Google Search Agent."""
        print(f"Google Search Agent: Searching for '{query}'...")
        snippets = self.search(query)
        if snippets:
            print(f"Google Search Agent: Found {len(snippets)} snippets.")
            return "\n".join(snippets)
        else:
            return "Google Search Agent: No results found."


class ImageAnalyzerAgent:
    """Agent that analyzes images for flights information."""

    def __init__(self, llm_model):
        self.llm = llm_model  # Use the shared GenerativeModel

    def analyze_violence(self, text):
        """Analyzes the text for violent content using the LLM."""
        prompt = f"""You are an AI assistant specializing in content moderation.  Your task is to analyze the provided text and determine if it contains violent content.  Provide a brief explanation of why you think the content is violent or not, and then provide a score from 0 to 1, where 0 means no violence detected and 1 means highly violent content.

        Text: {text}

        Your analysis should be structured as follows:

        Analysis: [Your analysis of the text]
        Violence Score: [0-1]
        """
        response = self.llm.generate_content(prompt)
        return response.text

    def run(self, text):
        """Runs the Text Analyzer Agent."""
        print("Text Analyzer Agent: Analyzing text for violence...")
        analysis = self.analyze_violence(text)
        print(f"Text Analyzer Agent: Analysis:\n{analysis}")
        return analysis


class PlanningAgent:
    """Agent that plans the execution of the multi-agent system."""

    def __init__(self, llm_model):
        self.llm = llm_model  # Use the shared GenerativeModel

    def create_plan(self, user_query):
        """Creates a plan based on the user query."""

        prompt = f"""You are a planning agent responsible for creating a step-by-step plan to answer a user's question. You have access to the following agents:

        1. Google Search Agent: This agent can search Google for information.  It takes a search query as input and returns relevant snippets.
        2. Text Analyzer Agent: This agent can analyze text for violent content.  It takes text as input and returns an analysis and a violence score (0-1).

        Your goal is to use these agents in the most effective way to answer the user's question.

        User Question: {user_query}

        Your plan should be a numbered list of steps.  Each step should specify which agent to use and what input to provide to that agent.  For example:

        1. Use Google Search Agent with query: "search query here"
        2. Use Text Analyzer Agent with text: "[result from Google Search Agent]"

        After listing the steps, provide a final statement on what the final answer to the users query will look like.

        Here is your plan:
        """

        response = self.llm.generate_content(prompt)
        return response.text

    def run(self, user_query):
        """Runs the Planning Agent."""
        print("Planning Agent: Creating a plan...")
        plan = self.create_plan(user_query)
        print(f"Planning Agent: Plan:\n{plan}")
        return plan


# --- Main Execution ---
def main():
    # Initialize agents with the shared GenerativeModel instance
    db_search_agent = DbSearchAgent(llm)
    image_analyzer_agent = ImageAnalyzerAgent(llm)
    planning_agent = PlanningAgent(llm)

    user_query = "Find examples of violent content online and analyze them."

    # 1. Create a plan using the Planning Agent
    plan = planning_agent.run(user_query)

    # Parse the plan (very basic parsing, improve for robustness)
    # This is a rudimentary way to parse the plan; in a real system,
    # use a more robust parser.  The LLM is generating text; we need
    # to turn that text back into executable actions.
    plan_lines = plan.split("\n")
    actions = []
    for line in plan_lines:
        if "Use Db Search Agent" in line:
            query = line.split("query:")[1].strip().replace('"', "")
            actions.append({"agent": "db_search", "input": query})
        elif "Use Image Analyzer Agent" in line:
            text = line.split("text:")[1].strip().replace('"', "")
            actions.append({"agent": "image_analyzer", "input": text})

    # 2. Execute the plan
    results = {}
    for action in actions:
        if action["agent"] == "db_search":
            results["db_search"] = db_search_agent.run(action["input"])
        elif action["agent"] == "image_analyzer":
            # Replace "[result from Google Search Agent]" with the actual result
            input_text = action["input"].replace(
                "[result from Db Search Agent]", results.get("db_search", "")
            )
            results["image_analysis"] = text_analyzer_agent.run(input_text)

    # 3. (Optional) Combine the results and present to the user.  This will
    # depend on the specific plan.
    print("\n--- Final Results ---")
    if "text_analysis" in results:
        print(results["text_analysis"])
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()