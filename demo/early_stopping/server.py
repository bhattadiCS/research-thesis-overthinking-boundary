"""Loopback-only HTTP/SSE host for the live early-stopping comparison."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit, parse_qs

if __package__:
    from .runtime import ComparisonManager, DEFAULT_MODEL, DEFAULT_PREDICTOR
else:
    from runtime import ComparisonManager, DEFAULT_MODEL, DEFAULT_PREDICTOR

STATIC = Path(__file__).resolve().parent / 'static'
RUN_ROUTE = re.compile(r'^/api/runs/([0-9a-f]{32})(?:/(events|cancel))?$')
MAX_BODY = 32768


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(manager, host='127.0.0.1', port=8765):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('This local demo binds only to loopback.')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                # Browser refresh/cancel can close a keep-alive socket.
                pass

        def log_message(self, fmt, *args):
            # Prompts never enter the request path or disk logs.
            if args and isinstance(args[0], str) and '/events' in args[0]:
                return
            super().log_message(fmt, *args)

        def _headers(self, code, content_type, length=None):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if length is not None:
                self.send_header('Content-Length', str(length))
            if self.close_connection:
                self.send_header('Connection', 'close')
            self.end_headers()

        def _json(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._headers(code, 'application/json; charset=utf-8', len(data))
            self.wfile.write(data)

        def _reject(self, value, code):
            # On Windows, closing a socket with a small unread request body
            # can reset the connection before the client receives the error.
            # Drain only a bounded declared body, with a short deadline.
            self.close_connection = True
            if self.command == 'POST':
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    length = 0
                if 0 < length <= MAX_BODY:
                    previous_timeout = self.connection.gettimeout()
                    try:
                        self.connection.settimeout(.5)
                        self.rfile.read(length)
                    except (OSError, TimeoutError):
                        pass
                    finally:
                        self.connection.settimeout(previous_timeout)
            self._json(value, code)

        def _local(self):
            port = self.server.server_address[1]
            allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
            if self.headers.get('Host', '').lower() not in allowed:
                self._reject({'error': 'Use this demo through its localhost address.'}, 403)
                return False
            origin = self.headers.get('Origin')
            if origin:
                try:
                    parsed = urlsplit(origin)
                    valid_origin = parsed.scheme == 'http' and parsed.netloc.lower() in allowed
                except ValueError:
                    valid_origin = False
                if not valid_origin:
                    self._reject({'error': 'Cross-origin requests are not accepted.'}, 403)
                    return False
            if self.headers.get('Sec-Fetch-Site', '') == 'cross-site':
                self._reject({'error': 'Cross-site requests are not accepted.'}, 403)
                return False
            return True

        def do_GET(self):
            if not self._local():
                return
            parsed = urlsplit(self.path)
            path = parsed.path
            static_paths = {'/': ('index.html','text/html'), '/index.html': ('index.html','text/html'),
                            '/styles.css': ('styles.css','text/css'), '/app.js': ('app.js','text/javascript')}
            if path in static_paths:
                name, mime = static_paths[path]
                try:
                    data = (STATIC/name).read_bytes()
                except FileNotFoundError:
                    self._json({'error': 'UI file is missing.'}, 404)
                    return
                self._headers(200, mime+'; charset=utf-8', len(data))
                self.wfile.write(data)
                return
            if path == '/api/status':
                self._json(manager.status())
                return
            match = RUN_ROUTE.fullmatch(path)
            if not match or match[2] == 'cancel':
                self._json({'error':'Not found.'}, 404)
                return
            try:
                state = manager.get(match[1])
            except KeyError:
                self._json({'error':'Comparison not found. Start a new comparison.'}, 404)
                return
            if match[2] != 'events':
                self._json(state.snapshot())
                return
            try:
                seq = int(self.headers.get('Last-Event-ID') or parse_qs(parsed.query).get('after', ['0'])[0])
                if not 0 <= seq <= len(state.snapshot()['events']):
                    raise ValueError()
            except (ValueError, TypeError):
                self._json({'error':'Invalid event cursor.'}, 400)
                return
            self._headers(200, 'text/event-stream; charset=utf-8')
            self.close_connection = True
            try:
                while True:
                    events, terminal = state.events_since(seq)
                    for event in events:
                        data = json.dumps(event, ensure_ascii=False, allow_nan=False)
                        self.wfile.write(f'id: {event["seq"]}\ndata: {data}\n\n'.encode('utf-8'))
                        seq = event['seq']
                    if not events and not terminal:
                        self.wfile.write(b': keep-alive\n\n')
                    self.wfile.flush()
                    if terminal:
                        break
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Disconnect does not silently change the comparison. It can be
                # reattached using Last-Event-ID or explicitly cancelled.
                pass

        def do_POST(self):
            if not self._local():
                self.close_connection = True
                return
            try:
                length = int(self.headers.get('Content-Length', '-1'))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self.close_connection = True
                self._reject({'error':'Request body is missing or too large.'}, 413 if length > MAX_BODY else 411)
                return
            if self.headers.get('Content-Type','').split(';')[0].strip() != 'application/json':
                self.close_connection = True
                self._reject({'error':'Send a JSON request.'}, 415)
                return
            try:
                data = json.loads(self.rfile.read(length).decode('utf-8'), parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
            except (ValueError, UnicodeDecodeError):
                self._json({'error':'Invalid JSON request.'}, 400)
                return
            path = urlsplit(self.path).path
            try:
                if path == '/api/runs':
                    state = manager.start(data)
                    self._json({'run_id':state.id}, 201)
                    return
                match = RUN_ROUTE.fullmatch(path)
                if match and match[2] == 'cancel':
                    if not isinstance(data, dict) or data:
                        raise ValueError('Cancel request must be an empty JSON object.')
                    manager.cancel(match[1])
                    self._json({'ok':True})
                    return
                self._json({'error':'Not found.'}, 404)
            except (ValueError, TypeError) as exc:
                self._json({'error':str(exc)}, 400)
            except KeyError:
                self._json({'error':'Comparison not found.'}, 404)
            except FileNotFoundError as exc:
                self._json({'error':str(exc)}, 503)
            except RuntimeError as exc:
                self._json({'error':str(exc)}, 409)

    server = DemoServer((host, port), Handler)
    server.manager = manager
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1','localhost'])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--predictor-path', type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()
    manager = ComparisonManager(args.model_path, args.device, args.predictor_path)
    server = create_server(manager, args.host, args.port)
    print(f'Live early-stopping UI: http://{args.host}:{server.server_address[1]}', flush=True)
    print('Local model only. Prompts stay in memory. Ctrl+C stops the server.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        server.server_close()


if __name__ == '__main__':
    main()
