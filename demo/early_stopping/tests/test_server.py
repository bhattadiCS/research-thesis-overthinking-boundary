"""Local HTTP/SSE boundary and worker lifecycle tests without a real model."""
from __future__ import annotations

from pathlib import Path
import http.client
import json
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import ScriptedGenerator, ScriptedPredictor
from demo.early_stopping.runtime import ComparisonManager, TERMINAL
from demo.early_stopping.server import MAX_BODY, create_server


class ServerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.generator = ScriptedGenerator()
        self.manager = ComparisonManager(model_path=ROOT / "demo/early_stopping/tests", device="cpu",
                                         generator_factory=lambda: self.generator)
        self.manager.predictor = ScriptedPredictor()
        self.server = create_server(self.manager, port=0)
        self.server.RequestHandlerClass.log_message = lambda *args: None
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.manager.close()
        self.generator.release.set()
        if self.manager.worker:
            self.manager.worker.join(2)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, payload=None, *, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        data = body if body is not None else (json.dumps(payload).encode() if payload is not None else None)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        try:
            connection.request(method, path, body=data, headers=request_headers)
            response = connection.getresponse()
            status, result_headers, content = response.status, dict(response.getheaders()), response.read()
            return status, result_headers, content
        finally:
            connection.close()

    def start(self, **settings):
        status, _, content = self.request("POST", "/api/runs", {"prompt": "Compute 12 + 13", **settings})
        self.assertEqual(status, 201, content)
        run_id = json.loads(content)["run_id"]
        return self.manager.get(run_id)

    def terminal(self, state):
        deadline = time.monotonic() + 2
        with state.condition:
            while state.status not in TERMINAL:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "worker failed to terminate")
                state.condition.wait(remaining)
        self.manager.worker.join(2)
        self.assertFalse(self.manager.worker.is_alive())
        return state.snapshot()

    def test_status_does_not_load_or_schedule_a_model(self):
        status, headers, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertFalse(data["busy"])
        self.assertFalse(data["model"]["loaded"])
        self.assertIsNone(self.manager.generator)
        self.assertEqual(self.generator.calls, [])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_only_explicit_loopback_bindings_are_allowed(self):
        for host in ("0.0.0.0", "::", "192.168.1.1", "example.com"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                create_server(self.manager, host=host, port=0)

    def test_foreign_host_origin_and_cross_site_controls_are_rejected(self):
        cases = [{"Host": "example.com"}, {"Host": f"127.0.0.1:{self.port + 1}"},
                 {"Origin": "https://example.com"}, {"Origin": "null"},
                 {"Origin": f"https://127.0.0.1:{self.port}"}, {"Sec-Fetch-Site": "cross-site"}]
        for headers in cases:
            with self.subTest(headers=headers):
                status, _, _ = self.request("POST", "/api/runs", {"prompt": "12+13"}, headers=headers)
                self.assertEqual(status, 403)
        self.assertEqual(self.manager.runs, {})

    def test_malformed_origin_is_rejected_without_handler_failure(self):
        status, _, body = self.request("POST", "/api/runs", {"prompt": "12+13"}, headers={"Origin": "http://["})
        self.assertEqual(status, 403, body)
        self.assertEqual(self.manager.runs, {})

    def test_path_whitelist_blocks_traversal_and_arbitrary_research_files(self):
        for path in ("/../research/prefix_stopping_model.py", "/%2e%2e/research/prefix_stopping_model.py",
                     "/%252e%252e/research/prefix_stopping_model.py", "/runtime.py", "/server.py",
                     "/research/outputs/semester2/prefix_model_v1/prefix_model.json", "/static/../runtime.py"):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404, body)
        self.assertEqual(self.generator.calls, [])

    def test_json_size_encoding_and_content_type_are_checked_before_work(self):
        status, _, _ = self.request("POST", "/api/runs", body=b"", headers={"Content-Length": str(MAX_BODY + 1)})
        self.assertEqual(status, 413)
        status, _, _ = self.request("POST", "/api/runs", body=b"{}", headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in (b"not-json", b"\xff", b'{"prompt":"12+13","step_cost":NaN}', b"[]", b'"prompt"', b"null",
                     b'{"prompt":"12+13","secret_option":1}', b"{}"):
            with self.subTest(body=body):
                status, _, _ = self.request("POST", "/api/runs", body=body)
                self.assertEqual(status, 400)
        self.assertEqual(self.manager.runs, {})
        self.assertEqual(self.generator.calls, [])

    def test_complete_run_has_one_terminal_sse_and_can_resume_by_sequence(self):
        state = self.start()
        snapshot = self.terminal(state)
        self.assertEqual(snapshot["status"], "complete")
        events = snapshot["events"]
        self.assertEqual([event["seq"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual([event["type"] for event in events if event["type"] in TERMINAL], ["complete"])
        self.assertEqual(snapshot["summary"]["early"]["stopped_at_step"], 2)
        self.assertFalse(self.manager.status()["busy"])
        status, headers, body = self.request("GET", f"/api/runs/{state.id}/events")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        streamed = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")]
        self.assertEqual(streamed, events)
        cursor = events[2]["seq"]
        status, _, resumed = self.request("GET", f"/api/runs/{state.id}/events", headers={"Last-Event-ID": str(cursor)})
        self.assertEqual(status, 200)
        after = [json.loads(line[6:]) for line in resumed.decode().splitlines() if line.startswith("data: ")]
        self.assertEqual(after, events[cursor:])

    def test_invalid_event_cursors_and_unknown_runs_are_bounded_errors(self):
        state = self.start()
        snapshot = self.terminal(state)
        for cursor in ("-1", "nope", str(snapshot["last_seq"] + 1)):
            with self.subTest(cursor=cursor):
                status, _, _ = self.request("GET", f"/api/runs/{state.id}/events?after={cursor}")
                self.assertEqual(status, 400)
        self.assertEqual(self.request("GET", "/api/runs/" + "0" * 32)[0], 404)
        self.assertEqual(self.request("POST", "/api/runs/" + "0" * 32 + "/cancel", {})[0], 404)

    def test_busy_run_rejects_overlap_and_cancel_releases_the_slot(self):
        self.generator.block_at = 1
        state = self.start()
        self.assertTrue(self.generator.entered.wait(2))
        status, _, _ = self.request("POST", "/api/runs", {"prompt": "another task"})
        self.assertEqual(status, 409)
        status, _, _ = self.request("POST", f"/api/runs/{state.id}/cancel", {})
        self.assertEqual(status, 200)
        snapshot = self.terminal(state)
        self.assertEqual(snapshot["status"], "cancelled")
        self.assertEqual([event["type"] for event in snapshot["events"] if event["type"] in TERMINAL], ["cancelled"])
        self.assertEqual([step for step, _, _ in self.generator.calls], [1])
        self.assertFalse(self.manager.status()["busy"])
        self.generator.block_at = None
        self.assertEqual(self.terminal(self.start())["status"], "complete")

    def test_generation_failure_emits_error_once_and_allows_another_run(self):
        self.generator.fail_at = 3
        with patch("traceback.print_exc"):
            snapshot = self.terminal(self.start())
        self.assertEqual(snapshot["status"], "error")
        self.assertIn("deterministic decoder failure", snapshot["error"])
        self.assertEqual([event["type"] for event in snapshot["events"] if event["type"] in TERMINAL], ["error"])
        self.assertIsNone(snapshot["summary"])
        self.assertFalse(self.manager.status()["busy"])
        self.generator.fail_at = None
        self.assertEqual(self.terminal(self.start())["status"], "complete")

    def test_stream_disconnect_does_not_cancel_work_and_reconnect_recovers_events(self):
        self.generator.block_at = 3
        state = self.start()
        self.assertTrue(self.generator.entered.wait(2))
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request("GET", f"/api/runs/{state.id}/events")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.readline().startswith(b"id: "))
            response.close()
        finally:
            connection.close()
        self.assertFalse(state.cancellation.cancelled)
        self.generator.release.set()
        snapshot = self.terminal(state)
        self.assertEqual(snapshot["status"], "complete")
        status, _, body = self.request("GET", f"/api/runs/{state.id}/events")
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")]
        self.assertEqual(events, snapshot["events"])


if __name__ == "__main__":
    unittest.main()
