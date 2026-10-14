# LocalWeatherAgent

A small **LLM-directed weather assistant** built with raw Python, [mimOE Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2), and [Open-Meteo](https://open-meteo.com/en/docs). A locally served **Qwen3-4B** model selects whether a weather lookup is needed; Python executes the request and sends the forecast back to the model for a natural-language response.

This project is a focused demonstration of an OpenAI-compatible local inference endpoint, model-directed tool selection, external API use, and explicit fallback logic. It does **not** require an agent framework.

## Architecture

```text
User question (CLI)
       |
       v
llm_select_tool() -- Qwen3-4B via mimOE
       |                     |
       | WEATHER             | NONE
       v                     |
parse_question()             |
       |                     |
get_weather()                |
       |                     |
Open-Meteo geocoding         |
+ forecast APIs              |
       |                     |
Structured weather data      |
       |                     |
       +----------+----------+
                  |
                  v
     local_llm_reply() -- Qwen3-4B via mimOE
                  |
                  v
         extract_final_answer()
                  |
                  v
             CLI answer
```

If the LLM does not return a valid `WEATHER` or `NONE` classification, the controller falls back to keyword-based routing (`needs_weather_tool()`). Python, not the LLM, currently parses the city/date and executes `get_weather()`. This is **LLM-directed action selection using a constrained text protocol**, not native OpenAI function calling or an autonomous multi-step planning loop.

**Execution boundary:** Model inference runs locally through mimOE; geocoding and forecast retrieval require internet access.

## Requirements

- Python **3.10+** (for `dict | None` and `tuple[str, str]` annotations).
- mimOE Studio with a model downloaded and **loaded** in Model View.
- An endpoint URL, model ID, and API-key value from Studio's **API** screen.
- Internet access to Open-Meteo.

Tested during development with `qwen3-4b`. The initially bundled `smollm-360m` was also evaluated for tool selection; see [Observed routing results](#observed-routing-results).

## Quick start (Windows / PowerShell)

1. Open the project directory in VS Code. Create an environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Edit `.env` using the **exact** values displayed in mimOE Studio's API screen:

   ```dotenv
   MIMOE_BASE_URL=http://10.0.0.94:8083/mimik-ai/openai/v1
   MIMOE_API_KEY=1234
   MIMOE_MODEL=qwen3-4b
   ```

   The address and key above are **development examples**, not universal settings. `127.0.0.1` or `localhost` may also work when the service is accessible on your own machine. Use Studio's reported model ID, not a guessed identifier.

4. With the model loaded, start the interactive assistant:

   ```powershell
   .\.venv\Scripts\python.exe weather_agent_v1.py
   ```

   If you rename the final script to `weather_agent.py` for GitHub, update this command accordingly.

On macOS/Linux, create the environment with `python3 -m venv .venv`, activate it with `source .venv/bin/activate`, install `requirements.txt`, and run the script using `python`.

## Example questions

```text
Will it rain in Vancouver tomorrow?
Should I wear a ski jacket tomorrow?
What's the temperature in Toronto today?
Tell me a joke.
What can you do?
quit
```

For a weather query, the CLI logs the classification source (`llm` or `fallback`), the executed weather tool call, and the structured forecast. Non-weather queries do not invoke Open-Meteo. The model then generates the user-facing response.

## Main functions

| Function | Responsibility |
| --- | --- |
| `get_coordinates(city)` | Resolve a city name using Open-Meteo's geocoding API. |
| `get_weather(city, day)` | Retrieve today's or tomorrow's daily forecast for the resolved location. |
| `parse_question(question)` | Extract a city and `today`/`tomorrow`, defaulting to Vancouver/today when not recognized. |
| `llm_select_tool(question)` | Ask the local model to return `WEATHER` or `NONE`; validate the classification. |
| `needs_weather_tool(question)` | Keyword-based fallback if classification is invalid. |
| `local_llm_reply(question, forecast)` | Generate an answer with or without retrieved forecast data. |
| `extract_final_answer(raw_output)` | Remove a completed Qwen3 `<think>...</think>` block and return the final text; reject incomplete blocks. |
| `run_agent(question)` | Coordinate selection, retrieval, logging, and response generation. |
| `main()` | Run the interactive command-line loop. |

The forecast currently contains daily maximum/minimum temperature (°C), daily precipitation sum (mm), and **maximum hourly precipitation probability** (%). This probability is **not** the probability of precipitation occurring at least once during the entire day. The present code does not fetch wind speed or hourly conditions.

## Observed routing results

Four hand-selected prompts were used to check whether the model returned a valid classification and chose the appropriate branch:

| Question | Expected | Qwen3-4B classification | Decision source |
| --- | --- | --- | --- |
| Will it rain in Vancouver tomorrow? | `WEATHER` | `WEATHER` | LLM |
| Should I wear a ski jacket tomorrow? | `WEATHER` | `WEATHER` | LLM |
| Tell me a joke. | `NONE` | `NONE` | LLM |
| What can you do? | `NONE` | `NONE` | LLM |

**Qwen3-4B: 4/4 valid and correct decisions** under the tested prompt and generation settings, without fallback. Earlier **SmolLM-360M: 0/4 valid tool-selection outputs** under different prompts and generation settings; the controller used fallback routing. These results are exploratory smoke tests, **not** a controlled model comparison or an estimate of general routing accuracy.

The final weather responses have **not** been systematically evaluated for factual grounding or recommendation accuracy. Successful tool selection alone does not establish answer reliability.

## Design decisions and limitations

- **Raw Python and OpenAI SDK:** The controller remains simple and auditable, without LangChain or CrewAI.
- **Constrained LLM decision:** Qwen3 produces the label `WEATHER` or `NONE`; Python validates it before taking an action.
- **Fallback:** Keyword matching is used when the model returns invalid text. It can miss implicit queries (such as clothing questions) or unnecessarily trigger on unrelated keywords.
- **Qwen3 reasoning output:** Streaming responses can include `<think>...</think>` text. The program extracts text after a closing `</think>` tag and raises an error for an incomplete thinking block. This is output handling, not validation of the final answer.
- **Limited parsing:** City/date parsing supports a narrow set of English patterns and only today/tomorrow. The default city is Vancouver; ambiguous city names are not disambiguated.
- **Answer quality:** The model can misinterpret questions (e.g., wearing a ski jacket versus going skiing) or generate unsupported claims. No automatic factual-grounding validator or deterministic answer fallback has been implemented.
- **Data availability:** Weather retrieval depends on a third-party online API, even though model inference is local.

Potential extensions include structured tool arguments, more reliable date/location parsing, automated tests, response validation against forecast fields, and broader evaluation across questions and models.

## Troubleshooting

- **Missing environment variable (`KeyError`):** Create `.env` and ensure `load_dotenv()` runs before accessing `os.environ`.
- **Connection refused or timeout:** Confirm Studio is running, the model is loaded, and the API address matches the Model View → API screen.
- **HTTP 403 on chat completions:** Check the endpoint and credentials. In local testing, the Studio-advertised address with `stream=True` succeeded; the specific cause of the earlier 403 was not isolated.
- **Model not found:** Match `MIMOE_MODEL` to the ID returned by Studio's API or `/models` endpoint.
- **Incomplete `<think>` block / truncated response:** Inspect the generation finish reason, request a larger output budget, or investigate whether thinking mode can be disabled in the runtime.
- **Incorrect or invented weather answer:** Compare the answer with the printed JSON tool result. Do not treat an LLM response as verified weather data.

## Data and credentials

The `.env` file must remain out of Git; `.env.example` may be committed. Use `.gitignore` and do not commit real authentication secrets. Keeping values out of Git does not, by itself, secure an inference server accessible on your local network.

Weather data are provided by [Open-Meteo](https://open-meteo.com/en/docs). Check its licensing and attribution terms before reuse or commercial deployment.
