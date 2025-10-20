# revlos

**revlos** is a RL-powered web login form brute-forcer trying to beat Hydra at one single task

##  Installation

### Quick Build
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

# Or run directly
go run revlos.go -L users.txt -P passwords.txt -f --auto target.com
```

This tool is for:
- ✅ Authorized penetration testing
- ✅ Security research with permission
- ✅ Testing your own applications
- ✅ Capture The Flag competitions
- ✅ Educational purposes

Always:
- Obtain written permission before testing
- Respect rate limits and server resources
- Follow responsible disclosure practices
- Comply with local laws and regulations

## POC only

This is a Proof of Concept (POC) tool for educational and authorized security testing purposes only. Use only with permission.

##  Credits

- Built with Go and chromedp
- RL algorithms inspired by multi-armed bandit research
- Compatible with Hydra command-line syntax
- Wordlists from SecLists by Daniel Miessler

**Remember**: With great power comes great responsibility. Use wisely! ️
