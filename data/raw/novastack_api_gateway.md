# NovaStack API Gateway Operations

All public REST traffic enters NovaStack through the Atlas API gateway. The default client rate limit is 600 requests per minute per tenant, while the reporting API is limited to 60 requests per minute because its queries are expensive. A 429 response includes a Retry-After header measured in seconds.

Every accepted request receives an `X-Request-ID`. The gateway preserves a valid request ID supplied by the client and otherwise creates a UUIDv7 identifier. Services must copy this value into structured logs and downstream requests so an operator can trace a transaction across the platform.

The gateway retries idempotent GET and HEAD requests at most two times for connection failures or HTTP 502, 503, and 504 responses. It never automatically retries POST requests, even when an idempotency key is present. The total gateway request timeout is 30 seconds; downstream services should use shorter budgets.

Requests larger than 10 MiB are rejected with status 413. WebSocket upgrades are allowed only on the `/events` route. Gateway deployments use a five-minute canary at ten percent of traffic before a region-wide rollout.

