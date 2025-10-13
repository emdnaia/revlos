# Description

revlos is an RL-powered web login form brute-forcer

What revlos CAN handle ✅:
Server-side authentication (PHP, Python, Node.js backends)
HTTP redirects (302, 301, 3xx)
Server-side error messages
Form-based POST/GET authentication

What revlos CANNOT handle (yet) ❌:
JavaScript-based authentication (like this site)
Single Page Applications (SPAs) with JWT/token auth
Sites requiring headless browsers

## Installation

```bash

go run revlos.go -L users.txt -P passwords.txt -f --auto 6.6.6.6

go build -o revlos revlos.go
./revlos -L users.txt -P passwords.txt -f --auto 6.6.6.6

```

## Quick Start

### Get Wordlists
```
curl -O https://raw.githubusercontent.com/danielmiessler/SecLists/master/Usernames/top-usernames-shortlist.txt
curl -O https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/2023-200_most_used_passwords.txt
```

### Auto Mode (Easiest)
```
./revlos -L top-usernames-shortlist.txt -P 2023-200_most_used_passwords.txt -f --auto 6.6.6.6
```

### Manual Mode (Hydra-compatible)
```
./revlos -L users.txt -P passwords.txt -f 6.6.6.6 -s 8080 \
  http-post-form "/:username=^USER^&password=^PASS^:F=Invalid credentials"
  ```

With Advanced Options
```
./revlos -L users.txt -P passwords.txt -f --auto \
  --method POST \
  --timeout 30 \
  --any-redirect \
  --header "Referer:http://6.6.6.6,X-Forwarded-For:127.0.0.1" \
  6.6.6.6:8080
  ```
