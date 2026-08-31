# NovaStack Observability Standards

Services emit structured JSON logs with timestamp, severity, service, environment, region, trace ID, and request ID fields. Logs must not contain access tokens, refresh tokens, passwords, payment card data, or raw customer message bodies. Email addresses are treated as personal data and must be masked.

Distributed traces use W3C Trace Context headers. The default trace sampling rate is five percent, but errors and requests slower than two seconds are always retained. Trace data is kept for 14 days, application logs for 30 days, and security audit logs for 365 days.

NovaStack defines availability from successful requests divided by valid requests. The core API monthly availability objective is 99.9 percent. Alerts should consume an error budget at a meaningful rate: the standard page fires when both the one-hour burn rate exceeds 14.4 and the six-hour burn rate exceeds 6.

Metrics use base units and low-cardinality labels. Customer IDs, request IDs, email addresses, and other unbounded values are forbidden as metric labels. Dashboards are helpful for investigation, but every paging alert must link to an owned runbook.

