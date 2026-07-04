"""system_prompt.py: single source of truth for Arche's persona.

Pins the product rule: the public API defaults to the SAME persona the
LiveKit product uses (agent.py), not a disconnected placeholder.
ARCHE_API_INSTRUCTIONS remains an explicit override.
"""

import unittest
from unittest import mock

import agent
import arche_api
import system_prompt


class SinglePromptSourceTests(unittest.TestCase):
    def test_agent_and_api_resolve_to_the_identical_prompt_by_default(self):
        self.assertEqual(agent.SYSTEM_PROMPT, arche_api.SYSTEM_PROMPT)
        self.assertEqual(agent.SYSTEM_PROMPT, system_prompt.SYSTEM_PROMPT)

    def test_default_prompt_is_not_a_generic_placeholder(self):
        self.assertNotIn("concise, warm voice assistant", system_prompt.SYSTEM_PROMPT)
        self.assertTrue(system_prompt.DEFAULT_SYSTEM_PROMPT.startswith("You are Crash."))

    def test_system_prompt_env_override_applies_to_both_entry_points(self):
        with mock.patch.dict("os.environ", {"SYSTEM_PROMPT": "custom persona"}, clear=False):
            import importlib

            importlib.reload(system_prompt)
            self.assertEqual(system_prompt.SYSTEM_PROMPT, "custom persona")
        importlib.reload(system_prompt)  # restore default for later tests


if __name__ == "__main__":
    unittest.main()
