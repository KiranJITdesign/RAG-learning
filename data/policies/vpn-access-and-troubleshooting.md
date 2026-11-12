---
id: vpn-access-and-troubleshooting
title: Global VPN Access and Troubleshooting Guide
category: it-support
who: all-employees
level: internal
benefit: Secure remote connection to internal codebases, databases, and staging environments
source: https://wiki.internal.corp/it/vpn-guide
---

# Global VPN Access & Troubleshooting Guide

## How to Get VPN Access

All employees and contractors requiring access to internal repositories, staging clusters, intranet dashboards, or developer databases must use the corporate VPN.

1. **Pre-requisites**: An active corporate Single Sign-On (SSO / Okta) account and Duo Mobile push authentication on your enrolled smartphone.
2. **Software**: Download the approved **GlobalProtect / Cisco AnyConnect** client from the internal portal (`https://software.internal.corp/vpn`).
3. **Gateway URL**:
   - Primary (Americas / EMEA): `gateway-primary.internal.corp`
   - Primary (APAC / India): `gateway-apac.internal.corp`
4. **Login Credentials**: Enter your corporate email username (e.g., `soudr@corp.com`) and your SSO password, then approve the Duo 2FA push notification.
5. **Access Approval**: Standard developer and staff access is auto-approved upon onboarding. Production environment access requires manager approval via an Access Request ticket.

## Step-by-Step Troubleshooting When VPN Fails

If your VPN connection drops or fails to connect with "Gateway Unreachable" or "Authentication Failed":

1. **Check Internet & DNS**: Ensure your local home WiFi is active. Try flushing your local DNS:
   - On Windows: Run `ipconfig /flushdns` in PowerShell.
   - On macOS: Run `sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`.
2. **Switch Gateway**: If `gateway-primary` fails, switch the gateway field to `gateway-apac.internal.corp` or `gateway-backup.internal.corp`.
3. **Re-sync MFA / Duo**: Open the Duo app, verify the push request matches your location, or generate a 6-digit one-time passcode from Duo and enter it after your password.
4. **Restart the VPN Agent Service**: Exit the VPN client completely, open Task Manager / Activity Monitor, terminate any hanging VPN daemon process, and restart the client as Administrator.
5. **Reboot Network Adapter**: Disable and re-enable your Wi-Fi or Ethernet adapter in network settings.

## Escalation & Automatic IT Ticket Logging

If none of the above troubleshooting steps resolve your issue within 10 minutes:
- Do not spend hours blocked. The **Internal Ops Co-Pilot** will automatically draft and log an **IT Support Incident (Category: Network / VPN)** directly to the IT Helpdesk queue.
- IT on-call engineers monitor the P2/P3 queue with a guaranteed response time under 30 minutes during working hours.
- Emergency hotline for critical release blockers: Slack channel `#it-emergencies` or extension `x4357`.
