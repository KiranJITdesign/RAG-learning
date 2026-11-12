---
id: identity-passwords-and-okta-sso
title: Single Sign-On (Okta SSO), 2FA and Security Standards
category: security
who: all-employees
level: internal
benefit: Unified identity management, enterprise password manager, and phishing defenses
source: https://security.internal.corp/policies/sso-passwords
---

# Single Sign-On (Okta SSO), 2FA & Security Standards

## Okta SSO & Login Guidelines

All corporate services (Google Workspace, Slack, GitHub, AWS, Workday, Figma) are federated through Okta Single Sign-On at `https://login.internal.corp`.
- **Master Password Requirements**: Minimum 16 characters, including uppercase, lowercase, numbers, and special symbols. Dictionary words or personal information are forbidden.
- **Password Rotation**: Passwords expire every 90 days. You will receive an automated prompt 7 days prior to expiry.
- **Two-Factor Authentication (2FA)**: Duo Mobile Push notification is mandatory on all corporate logins. SMS verification is prohibited due to SIM-swap vulnerabilities.

## Enterprise Password Manager (1Password)

- Every employee is provided an enterprise 1Password license.
- Store all team credentials, API tokens, and temporary keys inside assigned 1Password Shared Vaults.
- Never share plain text credentials via Slack, email, or Google Docs.

## Reporting Suspicious Emails or Phishing

If you receive an unexpected email requesting gift cards, password confirmations, or urgent wire transfers:
- Click the "Report Phishing" button in Google Workspace / Outlook immediately.
- Ping `#security-ops` on Slack or ask the Internal Ops Co-Pilot to alert the SecOps response team.
