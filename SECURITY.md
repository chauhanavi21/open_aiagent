# Security decisions

This file records how AgentForge meets the project's security checklist, including trade-offs and items that do not apply.

| Checklist item | Status | How |
|---|---|---|
| Packages maintained, latest stable, no critical vulns | Met | Zero runtime dependencies (stdlib only). Dev tools pinned in `requirements-dev.txt`; `pip-audit` reports none; Dependabot and a weekly CI scan watch for new ones. |
| Container images from Red Hat registry, non-root | N/A | The project ships no container image. If one is added it must use `registry.redhat.io/ubi9/python-311-minimal`, a non-root `USER`, and a read-only root filesystem. |
| No services bound to 0.0.0.0 | Met | `Config.validate` rejects any non-loopback host; test `test_config_rules`. |
| No hardcoded secrets | Met | Keys are read only from environment variables (`OPENROUTER_API_KEY`, `OLLAMA_API_KEY`, `AGENTFORGE_API_KEY`, `AGENTFORGE_TOKEN`); config stores only variable names; `.env*` is git-ignored. |
| TLS 1.2+ for all network communication | Met | Outbound TLS uses a verified context with `minimum_version = TLSv1_2` (`netutil.tls_context`). Provider URLs must be HTTPS; plain HTTP is allowed only for loopback (local Ollama). Redirects are refused. |
| Input validation and output encoding | Met | Every tool argument is type- and length-checked; the calculator parses the AST (no `eval`); the web UI renders output with `textContent` under a strict CSP; no SQL is used. |
| Non-root container execution | N/A | See container row. |
| Structured logging without sensitive data | Met | Server logs are JSON with a fixed field allow-list; keys, tokens and request bodies are never logged. Failed authentication is logged as an `auth_failure` event with path and client address. |
| Error handling without information disclosure | Met | Clients get generic messages (`invalid request`, `internal error`); stack traces go only to the server log. |
| SAST and dependency scanning in CI | Met | `.github/workflows/ci.yml` runs pytest, bandit and pip-audit on every push, pull request and weekly. |

## Known trade-offs

- **Authentication:** the local API uses a bearer token (constant-time compare). SAML/OpenID Connect and MFA are not implemented because the service is loopback-only and single-user. They would be required before exposing it beyond localhost.
- **Authorization:** there is a single privilege level for API callers; tools available to company roles are restricted per role (least privilege). Full RBAC for multiple users is not implemented.
- **Rate limiting and session timeout:** not implemented; the API has no sessions and relies on loopback binding.
- **Third-party LLMs:** when a hosted provider is in the brain chain, prompts leave the machine. Use `"brain": "local"` for sensitive data. Tool results are treated as untrusted in prompts and only registered tools can run.
- **Hashed requirements:** dev pins are exact versions but not hash-locked; `pip-audit` warns about this. Generating hashes with `pip-compile --generate-hashes` is a possible hardening step.
- **Log retention/alerting:** logs go to stderr; retaining them for 90 days and alerting on `auth_failure` is the deployer's responsibility.