# Security and artifact redaction

This repository is intended to make the compute pattern reproducible without exposing
the operator's management network.

Before committing a log or generated artifact, remove:

- management-network IP addresses and DNS names
- MAC addresses and hardware serial numbers
- usernames, home-directory paths and SSH configuration
- API keys, registry credentials, Hugging Face tokens and signed URLs
- container environment variables unrelated to the documented serving configuration

The `192.168.100.0/24`, `192.168.101.0/24`, `192.168.102.0/24` and
`192.168.200.1/32` addresses in this repository are example private fabric addresses.
Operators may reuse them only when they do not overlap an existing route.

Do not publish an unfiltered `docker inspect`, full process environment, `ip addr`, LLDP
dump or firmware inventory. Use the collection scripts as a starting point and inspect
their output manually before committing it.

Report security issues privately to the repository owner rather than placing secrets in
a public issue.
