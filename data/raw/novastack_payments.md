# NovaStack Payments Handbook

Payment creation requires an `Idempotency-Key` header containing 16 to 64 printable ASCII characters. NovaStack stores the key and the normalized request hash for 24 hours. Repeating the same key and payload returns the original response, while changing the payload produces HTTP 409.

Card authorization is synchronous, but capture and settlement updates arrive asynchronously. A payment initially enters `pending`, then changes to `authorized`, `captured`, `failed`, or `cancelled`. Merchants should listen for the `payment.updated` webhook instead of polling more than once per minute.

Webhook signatures use HMAC-SHA256. The signature covers the exact raw request body plus the timestamp, and consumers must reject timestamps more than five minutes old. A webhook delivery is retried for 72 hours with exponential backoff, so handlers must be idempotent.

Refunds can be submitted for up to 180 days after capture. Multiple partial refunds are allowed, but their sum cannot exceed the captured amount. NovaStack does not store card primary account numbers; card data is exchanged through the payment processor's hosted fields.

