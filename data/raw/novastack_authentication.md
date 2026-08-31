# NovaStack Authentication and Sessions

NovaStack uses OpenID Connect for employee and customer sign-in. The identity service issues access tokens as JSON Web Tokens (JWTs), and each access token expires after 15 minutes. Refresh tokens expire after 30 days and are rotated whenever they are used. A reused refresh token revokes the entire token family because reuse is treated as evidence of theft.

The API gateway validates the token signature, issuer, audience, expiry, and not-before time before forwarding a request. Signing keys are published through the identity service JWKS endpoint and cached by the gateway for five minutes. Authorization is not decided at the gateway: the destination service evaluates scopes and tenant membership for the requested resource.

Browser sessions store the refresh token in a Secure, HttpOnly, SameSite=Lax cookie. Access tokens stay in memory and must never be written to local storage. Native clients use the authorization-code flow with PKCE and store refresh tokens in the operating system keychain.

Service-to-service traffic uses workload identities rather than shared API keys. Emergency break-glass accounts require phishing-resistant hardware security keys, and every use creates a high-severity security alert. Clock skew of up to 60 seconds is accepted during token validation; larger differences must be corrected through the platform time service.

