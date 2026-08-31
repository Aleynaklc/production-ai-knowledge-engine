# Incident 2026-001: Authentication Errors During Key Rotation

On 14 February 2026, customers in the eu-west region experienced elevated HTTP 401 responses between 09:12 and 09:31 UTC. The peak authentication failure rate was 18 percent. Other regions were not affected, and no customer data was lost or exposed.

The identity service rotated its JWT signing key at 09:10 UTC. A gateway configuration regression had increased the JWKS cache duration from five minutes to one hour, so affected gateway instances continued using a stale key set. The incident commander forced a cache refresh at 09:27, and error rates returned to normal four minutes later.

Monitoring detected the problem through the regional authentication-success SLO, but the first alert arrived seven minutes after impact began. The team reduced that alert window from ten minutes to three minutes and added a synthetic login check to every region.

Corrective actions include enforcing the five-minute JWKS cache maximum in code, testing overlapping signing keys during every rotation, and adding a deployment policy that rejects unapproved cache changes. The incident severity was SEV-1 because more than ten percent of regional requests failed.

