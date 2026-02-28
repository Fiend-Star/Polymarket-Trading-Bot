"""
MetricsHandler — HTTP request handler for Prometheus/Grafana metrics endpoint.

SRP: HTTP routing and response formatting only.
"""
import urllib.parse
from http.server import BaseHTTPRequestHandler
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger


class MetricsHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler: serves Prometheus metrics and handles Grafana probes."""

    exporter = None  # Set by GrafanaMetricsExporter

    def _send_json(self, data, status=200, cors=False):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        if cors:
            self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data if isinstance(data, bytes) else data.encode())

    def _serve_root(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Polymarket Bot Metrics</h1>"
                        b"<p><a href='/metrics'>/metrics</a> | <a href='/health'>/health</a></p>"
                        b"</body></html>")

    def _serve_metrics(self):
        try:
            data = generate_latest(REGISTRY)
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            self.send_response(500); self.end_headers()
            self.wfile.write(f"Error: {e}".encode())

    def _serve_api(self, path):
        if 'labels' in path:
            body = b'{"status":"success","data":[]}'
        elif 'query' in path:
            body = b'{"status":"success","data":{"resultType":"vector","result":[]}}'
        else:
            body = b'{"status":"success"}'
        self._send_json(body, cors=True)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ('/', ''):
            self._serve_root()
        elif path == '/health':
            self._send_json(b'{"status": "healthy"}')
        elif path == '/metrics':
            self._serve_metrics()
        elif path.startswith('/api/v1/'):
            self._serve_api(path)
        else:
            self.send_response(404); self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith('/api/v1/'):
            self._serve_api(parsed.path)
        elif parsed.path == '/metrics':
            self.do_GET()
        else:
            self.send_response(404); self.end_headers()
            self.wfile.write(b"Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Accept, Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def log_message(self, format, *args):
        try:
            if len(args) >= 2:
                sc = int(args[1]) if str(args[1]).isdigit() else 0
                if sc >= 400:
                    logger.debug(f"Metrics server: {format % args}")
        except Exception:
            pass

