
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Supply test-only configuration before importing the agent.
# No actual inference request is made.
os.environ["MIMOE_BASE_URL"] = (
    "http://127.0.0.1:8083/mimik-ai/openai/v1"
)
os.environ["MIMOE_API_KEY"] = "test-only-key"

import weather_agent_v1 as agent


class TestWeatherAgent(unittest.TestCase):

    # Test 1: Extract city and day
    def test_parse_question(self):
        city, day = agent.parse_question(
            "Will it rain in Vancouver tomorrow?"
        )

        self.assertEqual(city, "Vancouver")
        self.assertEqual(day, "tomorrow")

    # Test 2: Default city
    def test_default_location(self):
        city, day = agent.parse_question(
            "Should I cycle to work tomorrow?"
        )

        self.assertEqual(city, "Vancouver")
        self.assertEqual(day, "tomorrow")

    # Test 3: Recognize weather questions
    def test_weather_routing(self):
        result = agent.needs_weather_tool(
            "Will it rain tomorrow?"
        )

        self.assertTrue(result)

    # Test 4: Recognize implicit weather questions
    def test_clothing_routing(self):
        result = agent.needs_weather_tool(
            "Should I wear a ski jacket tomorrow?"
        )

        self.assertTrue(result)

    # Test 5: Skip unnecessary weather lookups
    def test_non_weather_routing(self):
        result = agent.needs_weather_tool(
            "Tell me a joke."
        )

        self.assertFalse(result)

    # Test 6: Extract Qwen3's final answer
    def test_extract_final_answer(self):
        raw = (
            "<think>Analyzing the forecast.</think>"
            "Wear a light jacket."
        )

        answer = agent.extract_final_answer(raw)

        self.assertEqual(answer, "Wear a light jacket.")

    # Test 7: Reject incomplete thinking blocks
    def test_incomplete_thinking(self):
        raw = "<think>Analyzing the forecast."

        with self.assertRaises(ValueError):
            agent.extract_final_answer(raw)

    # Test 8: Parse a mocked LLM tool decision
    def test_llm_tool_selection(self):
        def chunk(content=None, finish_reason=None):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content),
                        finish_reason=finish_reason
                    )
                ]
            )

        fake_stream = iter([
            chunk("<think>\n</think>\n"),
            chunk("WEATHER"),
            chunk(finish_reason="stop")
        ])

        with patch.object(
            agent.client.chat.completions,
            "create",
            return_value=fake_stream
        ):
            use_weather, source = agent.llm_select_tool(
                "Will it rain tomorrow?"
            )

        self.assertTrue(use_weather)
        self.assertEqual(source, "llm")

    # Test 9: Verify weather tool execution
    def test_agent_calls_weather_tool(self):
        forecast = {
            "city": "Vancouver",
            "temperature_max": 15.8,
            "precipitation_mm": 5.0
        }

        with (
            patch.object(
                agent, "llm_select_tool",
                return_value=(True, "llm")
            ),
            patch.object(
                agent, "get_weather",
                return_value=forecast
            ) as mock_weather,
            patch.object(
                agent, "local_llm_reply",
                return_value="Bring a rain jacket."
            ) as mock_reply
        ):
            answer = agent.run_agent(
                "Will it rain in Vancouver tomorrow?"
            )

        mock_weather.assert_called_once_with(
            "Vancouver", "tomorrow"
        )
        mock_reply.assert_called_once_with(
            "Will it rain in Vancouver tomorrow?",
            forecast
        )
        self.assertEqual(answer, "Bring a rain jacket.")

    # Test 10: Verify no weather tool is called
    def test_agent_skips_weather_tool(self):
        with (
            patch.object(
                agent, "llm_select_tool",
                return_value=(False, "llm")
            ),
            patch.object(
                agent, "get_weather"
            ) as mock_weather,
            patch.object(
                agent, "local_llm_reply",
                return_value="I can answer weather questions."
            )
        ):
            answer = agent.run_agent("What can you do?")

        mock_weather.assert_not_called()
        self.assertEqual(
            answer,
            "I can answer weather questions."
        )


if __name__ == "__main__":
    unittest.main()
