import os
#load_dotenv() reads .env and puts its values into environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import gradio as gr
from tavily import TavilyClient
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For calling AI
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Is used for web search 

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing. Check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Simple helper: search the web and return a short text summary of results.
    """
    try:
        resp = tavily.search(query=query, search_depth="basic", max_results=max_results)
        results = resp.get("results", [])
        lines = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            lines.append(f"- {title}\n  {content}\n  {url}")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"

# build the prompt fo the AI
def build_prompt(origin, destination, days, budget, interests, pace, flights_text, hotels_text, pois_text):
    return f"""
You are a travel planner.

User info:
- Origin: {origin}
- Destination: {destination}
- Trip length (days): {days}
- Budget: {budget}
- Interests: {interests}
- Pace : {pace}
Pace rules:
- Relaxed: fewer activities per day, more breaks, avoid rushing.
- Normal: balanced schedule, moderate walking.
- Packed: more activities per day, early starts, efficient routes.

Search results (use these to make the plan realistic):
Flights:
{flights_text}

Hotels:
{hotels_text}

Points of interest:
{pois_text}

Task:
Create a clear day-by-day itinerary for {days} days.
Include:
- morning / afternoon / evening
- 2 to 4 activities per day (use the pace rules)
- 2 food suggestion per day
Keep it simple and readable in Markdown.
"""


def plan_trip(origin, destination, days, budget, interests,pace):
    flights_q = f"best flights from {origin} to {destination}"
    hotels_q = f"best hotels in {destination} under {budget} budget"
    pois_q = f"top attractions in {destination} for {interests}"

    flights_text = tavily_search(flights_q)
    hotels_text = tavily_search(hotels_q)
    pois_text = tavily_search(pois_q)

    prompt = build_prompt(origin, destination, days, budget, interests,pace,flights_text, hotels_text, pois_text)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful travel planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,# cotrol randomness # lower means more consistent
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"

# create the Gradio UI
demo = gr.Interface(
    fn=plan_trip,
    inputs=[
        gr.Textbox(label="🌍 Origin (city)", placeholder="Atlanta"),
        gr.Textbox(label="📍 Destination (city)", placeholder="New York"),
        gr.Number(label="🗓️ Trip length (days)", value=3),
        gr.Textbox(label="💰 Budget (example: 800 dollars)", placeholder="800"),
        gr.Textbox(label="🎯 Interests (example: museums, food)", placeholder="museums, food"),
        gr.Dropdown(
            label="⚡ Pace",
            choices=["Relaxed", "Normal", "Packed"],
            value="Normal"
        ),
    ],
    outputs=gr.Markdown(label="🧾 Your Trip Plan"),
    title="🧳 TripSmith Travel Planner",
    description="🔎 Tavily web search + 🤖 OpenAI LLM itinerary generation",
)


if __name__ == "__main__":
    demo.launch()
