# NovaStack Security and Data Policy

NovaStack classifies information as Public, Internal, Confidential, or Restricted. Authentication secrets, payment credentials, and government identifiers are Restricted. Restricted data must be encrypted in transit with TLS 1.2 or newer and at rest with managed keys.

Production access follows least privilege and is granted just in time for a maximum of four hours. Engineers authenticate with single sign-on and phishing-resistant multi-factor authentication. Direct SSH access to production hosts is disabled; approved sessions pass through the audited access proxy.

Critical vulnerabilities exposed to the internet must be remediated within 24 hours. Other critical findings have a seven-day deadline, high-severity findings have 30 days, and medium-severity findings have 90 days. Exceptions require a documented risk owner and an expiration date.

Customer data is deleted from active systems within 30 days of an approved deletion request. Encrypted backups age out according to their normal retention schedule, and deleted records are not restored into active service. Security incidents must be reported immediately through the dedicated incident channel.

