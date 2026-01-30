import unittest

from agenticevals.schema import ToolEndpointSpec, ToolSpec
from agenticevals.tools import ToolDispatcher


class ToolDispatcherTests(unittest.TestCase):
    def test_rejects_missing_required_argument_before_dispatch(self):
        dispatcher = ToolDispatcher(
            tools=[
                ToolSpec(
                    name="send",
                    description="Send a message",
                    input_schema={"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]},
                )
            ],
            endpoints=[ToolEndpointSpec(tool_name="send", url="http://127.0.0.1:1/send")],
        )

        result = dispatcher.dispatch("send", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 500)
        self.assertIn("missing required", result.error)
        self.assertEqual(len(dispatcher.records), 1)

    def test_rejects_wrong_json_type_before_dispatch(self):
        dispatcher = ToolDispatcher(
            tools=[
                ToolSpec(
                    name="list",
                    description="List messages",
                    input_schema={"type": "object", "properties": {"max_results": {"type": "integer"}}},
                )
            ],
            endpoints=[ToolEndpointSpec(tool_name="list", url="http://127.0.0.1:1/list")],
        )

        result = dispatcher.dispatch("list", {"max_results": "ten"})

        self.assertFalse(result.ok)
        self.assertIn("invalid type", result.error)


if __name__ == "__main__":
    unittest.main()
