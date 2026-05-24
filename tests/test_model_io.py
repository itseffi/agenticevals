import unittest
import urllib.error

from agenticevals.model_io import _cache_key, _gemini_request, _is_retryable, _with_retries


class RetryTests(unittest.TestCase):
    def test_retries_transient_errors_then_succeeds(self):
        calls = {"n": 0}
        sleeps: list[float] = []

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.URLError("connection reset")
            return "ok"

        result = _with_retries(flaky, attempts=3, sleep=sleeps.append)

        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(sleeps), 2)  # slept before each of the two retries

    def test_does_not_retry_client_errors(self):
        calls = {"n": 0}

        def bad_request():
            calls["n"] += 1
            raise urllib.error.HTTPError("u", 400, "bad request", {}, None)

        with self.assertRaises(urllib.error.HTTPError):
            _with_retries(bad_request, attempts=3, sleep=lambda _s: None)
        self.assertEqual(calls["n"], 1)

    def test_retries_then_raises_after_exhausting_attempts(self):
        calls = {"n": 0}

        def always_503():
            calls["n"] += 1
            raise urllib.error.HTTPError("u", 503, "unavailable", {}, None)

        with self.assertRaises(urllib.error.HTTPError):
            _with_retries(always_503, attempts=3, sleep=lambda _s: None)
        self.assertEqual(calls["n"], 3)

    def test_retryable_classification(self):
        self.assertTrue(_is_retryable(urllib.error.HTTPError("u", 429, "", {}, None)))
        self.assertTrue(_is_retryable(urllib.error.HTTPError("u", 500, "", {}, None)))
        self.assertTrue(_is_retryable(urllib.error.URLError("down")))
        self.assertFalse(_is_retryable(urllib.error.HTTPError("u", 404, "", {}, None)))


class CacheKeyTests(unittest.TestCase):
    def test_same_inputs_produce_same_key(self):
        a = _cache_key("openai", "gpt-4o-mini", "hello", {"temperature": 0})
        b = _cache_key("openai", "gpt-4o-mini", "hello", {"temperature": 0})
        self.assertEqual(a, b)

    def test_differing_params_produce_different_keys(self):
        cold = _cache_key("openai", "gpt-4o-mini", "hello", {"temperature": 0})
        hot = _cache_key("openai", "gpt-4o-mini", "hello", {"temperature": 1})
        self.assertNotEqual(cold, hot)


class GeminiRequestTests(unittest.TestCase):
    def test_api_key_is_sent_as_header_not_in_url(self):
        request = _gemini_request("gemini-2.5-flash", {"contents": []}, "secret-key")
        self.assertNotIn("secret-key", request.full_url)
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "secret-key")


if __name__ == "__main__":
    unittest.main()
