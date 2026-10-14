import json
import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load configuration from .env
load_dotenv()

# load model
MODEL = os.getenv("MIMOE_MODEL", "smollm-360m")
print(f"loading {MODEL}")

client = OpenAI(
    base_url=os.environ['MIMOE_BASE_URL'],
    api_key=os.environ['MIMOE_API_KEY'],
    timeout=60.0
)

def get_coordinates(city: str) -> dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    
    data = response.json()
    locations = data.get("results", [])

    if not locations:
        raise ValueError(f"City not found: {city}")

    return locations[0] 


def get_weather(city:str, day:str) -> dict: 
    location = get_coordinates(city)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum"
        ],
        "timezone": location["timezone"],
        "forecast_days": 2
    }

    response = requests.get(
        url, 
        params=params,
        timeout=15
    )
    response.raise_for_status()

    data = response.json()
    daily  = data['daily']

    if day not in ('today', 'tomorrow'):
        raise ValueError('Day must be either today or tomorrow')

    index = 0 if day == "today" else 1
    
    return {
    "city": location["name"],
    "date": daily["time"][index],
    "temperature_max": daily["temperature_2m_max"][index],
    "temperature_min": daily["temperature_2m_min"][index],
    "peak_hourly_rain_probability": (
        daily["precipitation_probability_max"][index]
    ),
    "precipitation_mm": daily["precipitation_sum"][index]
}


def parse_question(question: str) -> tuple[str, str]:

    # Extract the city after "in" or "for"
    match = re.search(
        r"\b(?:in|for)\s+([A-Za-z][A-Za-z '\-]*?)"
        r"(?=\s+(?:today|tomorrow)\b|[?!.]|$)",
        question,
        re.IGNORECASE
    )

    city = match.group(1).strip() if match else "Vancouver"

    # Determine the requested day
    if re.search(r"\btomorrow\b", question, re.IGNORECASE):
        day = "tomorrow"
    else:
        day = "today"

    return city, day


def needs_weather_tool(question: str) -> bool:
    """Determine whether a question needs weather data."""

    weather_keywords = [
        "weather",
        "forecast",
        "rain",
        "raining",
        "temperature",
        "snow",
        "wind",
        "windy",
        "umbrella",
        "cycle",
        "cycling",
        "bike",
        "biking",
        "outdoor",
    ]

    for keyword in weather_keywords:
        if re.search(rf"\b{keyword}\b", question, re.IGNORECASE):
            return True

    return False



def llm_select_tool(question: str) -> tuple[bool, str]:
    """Ask Qwen3 to select whether weather retrieval is needed."""

    prompt = f"""
Classify the question as WEATHER or NONE.

WEATHER: Requires a weather forecast.
NONE: Does not require a weather forecast.

Examples:
Will it rain tomorrow? -> WEATHER
Should I wear a ski jacket? -> WEATHER
Tell me a joke. -> NONE
What can you do? -> NONE

Question: {question}
Classification:

/no_think
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=128,
        stream=True
    )

    tokens = []
    finish_reason = None

    for chunk in response:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        token = choice.delta.content

        if token:
            tokens.append(token)

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    raw_output = "".join(tokens).strip()

    # Extract text after the final thinking block
    if "</think>" in raw_output:
        final_text = raw_output.rsplit("</think>", 1)[-1].strip()
    else:
        final_text = raw_output

    print(f"[LLM final text] {final_text[:200]!r}")
    print(f"[Finish reason] {finish_reason}")

    # Accept only a valid classification
    match = re.fullmatch(
        r"(?:Action:\s*)?(WEATHER|NONE)\.?",
        final_text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper() == "WEATHER", "llm"

    print("[Agent] Invalid decision; using fallback.")
    return needs_weather_tool(question), "fallback"


    
def local_llm_reply(question: str, forecast: dict | None = None) -> str:
    """Generate a response using the local mimOE model."""

    if forecast is None:
        system_message = (
            "You are a weather assistant. "
            "Explain your capabilities briefly. "
            "Do not invent weather forecasts."
        )
        user_message = question

    else:
        system_message = (
            "You are a helpful weather assistant. "
            "Answer the user's question using ONLY the "
            "provided forecast data. "
            "Use 1-3 short sentences. "
            "Do not invent weather conditions or numbers. "
            "Peak hourly precipitation probability is not "
            "the probability of rain at any time during the day."
        )

        user_message = (
            f"Question: {question}\n"
            f"Weather data: {json.dumps(forecast)}"
        )

    
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": user_message + "\n/no_think"
            }
        ],
        temperature=0,
        max_tokens=512,
        stream=True
    )

    tokens = []
    finish_reason = None

    for chunk in response:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]

        if choice.delta.content:
            tokens.append(choice.delta.content)

        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    raw_output = "".join(tokens)

    print(f"[Generation finish reason] {finish_reason}")

    if finish_reason == "length":
        raise ValueError(
            "Model output truncated. "
            "Increase max_tokens or disable thinking."
        )

    return extract_final_answer(raw_output)




def run_agent(question: str) -> str:
    """Run tool selection, weather retrieval, and LLM inference."""

    use_weather, decision_source = llm_select_tool(question)

    print(f"[Agent] Routing source: {decision_source}")

    if use_weather:
        city, day = parse_question(question)

        print(f"[Agent] Calling get_weather({city!r}, {day!r})")

        forecast = get_weather(city, day)

        print(f"[Tool result] {json.dumps(forecast, indent=2)}")

        answer = local_llm_reply(question, forecast)

    else:
        print("[Agent] No weather tool required")
        answer = local_llm_reply(question)

    return answer


def extract_final_answer(raw_output: str) -> str:
    """Extract the final answer from Qwen3 output."""

    if "</think>" in raw_output:
        # Extract everything after the closing thinking tag
        answer = raw_output.rsplit("</think>", 1)[-1].strip()

    elif "<think>" in raw_output:
        # Reasoning started but was never completed
        raise ValueError(
            "Incomplete thinking block. "
            "Increase max_tokens or disable thinking mode."
        )

    else:
        # No thinking block
        answer = raw_output.strip()

    if not answer:
        raise ValueError("Model returned an empty final answer.")

    return answer



def main():
    print("\nLocalWeatherAgent powered by mimOE")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in ("quit", "exit"):
                break

            if not question:
                continue

            answer = run_agent(question)
            print(f"\nAssistant: {answer}\n")

        except (KeyboardInterrupt, EOFError):
            break

        except Exception as exc:
            print(f"[Error] {type(exc).__name__}: {exc}")

    print("Goodbye!")


if __name__ == "__main__":
    main()
