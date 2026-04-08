from prometheus_client import Counter, Histogram

# Total number of HTTP requests
HTTP_REQUESTS_TOTAL = Counter(
    name="http_requests_total",
    documentation="Total number of HTTP requests",
    labelnames=("method", "path", "status_code"),
)


# Duration of HTTP requests in seconds
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    name="http_request_duration_seconds",
    documentation="Duration of HTTP requests in seconds",
    labelnames=("method", "path", "status_code"),
)

# Total number of HTTP error responses
HTTP_ERRORS_TOTAL = Counter(
    name="http_errors_total",
    documentation="Total number of HTTP error responses",
    labelnames=("path", "status_code", "error_code"),
)


RECOMMENDER_FALLBACK_TOTAL = Counter(
    "recommender_fallback_total",
    "Counts how often the recommender falls back instead of using the primary algorithm.",
    ["reason"],
)
