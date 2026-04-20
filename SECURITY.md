# Security Policy

## Scope

This project is a **local-run educational tool** — it performs no network requests, stores no user data, requires no authentication, and connects to no external services or databases. The attack surface is therefore limited, but dependency vulnerabilities and unsafe code practices are still taken seriously.

## Supported Versions

Only the latest commit on the `main` branch is actively maintained. No backported security fixes are provided for older versions.

| Version | Supported |
|:--------|:---------:|
| `main` (latest) | ✅ |
| Older commits | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities, as this exposes the issue before a fix is available.

Instead, use one of the following:

- **GitHub private vulnerability reporting** — go to the [Security tab](https://github.com/joe6302413/SSE-demo/security/advisories/new) of this repository and submit a private advisory.
- **Email** — **Contact:** Dr. Yi-Chun Chin **Email:** ycchin@ntu.edu.tw

Please include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof of concept
- The version or commit hash where the issue was found
- Any suggested mitigation, if known

## What to Expect

| Timeline | Action |
|:---------|:-------|
| Within **5 business days** | Acknowledgement of the report |
| Within **30 days** | Assessment and, if confirmed, a fix or mitigation plan |
| After fix is released | Public disclosure coordinated with the reporter |

If a reported issue is determined not to be a vulnerability, we will explain why and close the report.

## Out of Scope

The following are **not** considered security vulnerabilities for this project:

- Physics approximations or numerical precision in the simulation (these are educational, not safety-critical)
- Issues that require physical access to the machine running the app
- Vulnerabilities in Python itself or the operating system
- Self-inflicted issues from modifying the source code

## Dependencies

This project relies on `streamlit`, `numpy`, and `matplotlib`. If you discover a vulnerability in one of these upstream packages that affects this project, please report it to the upstream maintainers first, then notify us so we can update the dependency.
