"""
Custom middleware for rate limiting and security
"""

import logging
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""

    def __init__(self, app, calls: int = 10, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Get client IP, fallback to 'unknown' if not available
        client_ip = request.client.host if request.client is not None else "unknown"

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Check rate limit
        now = time.time()
        client_requests = self.clients[client_ip]

        # Remove old requests outside the time window
        while client_requests and client_requests[0] <= now - self.period:
            client_requests.popleft()

        # Check if client exceeded rate limit
        if len(client_requests) >= self.calls:
            logger.warning("Rate limit exceeded for IP: %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error": "Rate limit exceeded",
                        "details": f"Maximum {self.calls} requests per {self.period} seconds",
                    }
                },
            )

        # Add current request
        client_requests.append(now)

        # Process request
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests for monitoring"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        client_host = request.client.host if request.client is not None else "unknown"
        logger.info(
            "Request: %s %s from %s", request.method, request.url.path, client_host
        )

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info("Response: %d in %.3fs", response.status_code, process_time)

        return response
