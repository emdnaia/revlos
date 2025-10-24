## ? Quick Start

### **Installation**

```bash
# Clone repository
git clone https://github.com/emdnaia/revlos.git
cd revlos

# Build
go build -o revlos revlos.go

# Optional: Install dependencies for full features
# Hashcat (for keyword password generation)
sudo apt install hashcat

# CUPP (for personalized password generation)
git clone https://github.com/Mebus/cupp.git

# username-anarchy (for OSINT username generation)
git clone https://github.com/urbanadventurer/username-anarchy.git
```

### **Basic Usage**

```bash
# Auto-detection mode (recommended)
./revlos --auto -L users.txt -P passwords.txt http://target.com

# Manual mode with form specification
./revlos -L users.txt -P passwords.txt \
  http://target.com \
  http-post-form "/login:username=^USER^&password=^PASS^:F=Invalid"

# HTTP Basic Authentication
./revlos --auto -L users.txt -P passwords.txt --basic-auth http://target.com

# Stop on first valid credential
./revlos --auto -L users.txt -P passwords.txt -f http://target.com

# Headless mode for JavaScript/SPA sites
./revlos --auto -L users.txt -P passwords.txt --headless http://target.com
```

---

## ? Usage Examples

### **Web Application Login**
```bash
./revlos --auto -L usernames.txt -P rockyou.txt \
  -f --timeout 15 http://example.com/login
```

### **Custom Success Detection**
```bash
./revlos -L users.txt -P passwords.txt \
  --success-text "Welcome" \
  --any-redirect \
  http://target.com/auth
```

### **OSINT-Enhanced Attack**
```bash
./revlos --osint --auto -L users.txt -P passwords.txt \
  --username-anarchy-path ./username-anarchy/username-anarchy \
  http://corporate-site.com/login
```

### **Rate-Limited Target**
```bash
./revlos --auto -L users.txt -P passwords.txt \
  -t 10 --timeout 30 \
  http://slow-target.com
```

### **Single Credential Test**
```bash
./revlos -l admin -p Admin123 --auto http://target.com
```


### **Core Options**
```
-l <user>          Single username (literal)
-L <file>          Username list file or URL
-p <pass>          Single password (literal)
-P <file>          Password list file or URL
-f                 Stop on first valid credential
-s <port>          Port (default: 80)
-t <threads>       Parallel tasks (default: 100)
-q                 Quiet mode
-v                 Verbose mode
```

### **Auto-Detection**
```
--auto             Auto-detect form fields and error messages
--headless         Use headless browser for JavaScript sites
--basic-auth       Force HTTP Basic Authentication mode
```

### **Success Detection**
```
--success-text <s>    Text in response indicating success
--success-cookie <c>  Cookie name indicating success
--success-code <c>    HTTP status codes for success (e.g., 200,302)
--any-redirect        Treat any 3xx redirect as success
```

### **Advanced Options**
```
--timeout <n>         Request timeout in seconds (default: 10)
--max-time <n>        Maximum total time in seconds
--method <m>          HTTP method: GET or POST (default: POST)
--header <h>          Custom headers (comma-separated)
--learning-rate <r>   RL learning rate 0.0-1.0 (default: 0.4)
--batch-size <n>      Credentials per batch (default: 150)
```

### **OSINT Options**
```
--osint                      Enable OSINT intelligence gathering
--username-anarchy-path <p>  Path to username-anarchy tool
```
