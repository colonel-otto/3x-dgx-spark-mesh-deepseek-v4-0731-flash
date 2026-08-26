# Security and artifact redaction

This repository is intended to make the compute pattern reproducible without exposing
the operator's management network or hardware identity.

Before committing a log or generated artifact, remove:

- management-network IP addresses and DNS names
- MAC addresses, IPv6 link-local addresses and hardware serial numbers
- real machine hostnames
- usernames, home-directory paths and SSH configuration
- API keys, registry credentials, Hugging Face tokens and signed URLs
- container environment variables and images unrelated to the documented serving
  configuration

## Placeholder conventions used here

Every address, hostname and hardware identifier in this repository is a placeholder.
They are internally consistent, so the topology still reads correctly, but they do not
correspond to any real host.

| Placeholder | Meaning |
|---|---|
| `192.168.10.0/24` | management LAN (`.1`/`.2`/`.3` are nodes 0/1/2, `.9` an API client) |
| `192.168.99.0/24`, `192.168.100.0/24`, `192.168.101.0/24`, `192.168.102.0/24`, `192.168.110.0/30`, `192.168.200.1/32` | example private fabric addresses |
| `node0`, `node1`, `node2` | the three Sparks |
| `<mac-nodeN-pM>` | a redacted MAC address |
| `<link-local>` | a redacted IPv6 `fe80::` address |
| `<redacted>` | a redacted hardware serial |

Operators may reuse the fabric ranges only when they do not overlap an existing route.

## Enforcement

`scripts/check_no_sensitive.py` runs as a pre-commit hook (`make install-hooks`) and
fails the commit on serials, real emails, personal names, management-network addresses,
MAC addresses, IPv6 link-local addresses, real DGX hostnames, username-bearing home
paths, and credential-shaped strings.

The hook checks the working tree, not history. **A redaction commit does not remove a
secret that is already published** — the pre-redaction blob remains an ancestor and is
still fetchable. If a secret reaches a public branch, rewrite history, force-push, and
ask GitHub Support to purge the unreferenced objects; treat the value as disclosed and
rotate it where rotation is possible.

Do not publish an unfiltered `docker inspect`, full process environment, `ip addr`, LLDP
dump or firmware inventory. Use the collection scripts as a starting point and inspect
their output manually before committing it.

Report security issues privately to the repository owner rather than placing secrets in
a public issue.
