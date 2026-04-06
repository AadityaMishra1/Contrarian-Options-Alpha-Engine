package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sync"
	"time"
)

const (
	polygonBase        = "https://api.polygon.io"
	snapshotPathFmt    = "/v3/snapshot/options/%s"
	defaultTimeout     = 15 * time.Second
	defaultConcurrency = 5
	defaultPageLimit   = 250
	maxPages           = 10
)

// Client fetches options snapshots from Polygon.io with concurrent fan-out
// across multiple underlying symbols.
type Client struct {
	apiKey      string
	httpClient  *http.Client
	concurrency int
	baseURL     string // override for testing; defaults to polygonBase
}

// NewClient creates an ingestion client. concurrency controls how many symbols
// are fetched in parallel (0 uses the default of 5).
func NewClient(apiKey string, concurrency int) *Client {
	if concurrency <= 0 {
		concurrency = defaultConcurrency
	}
	return &Client{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: defaultTimeout,
		},
		concurrency: concurrency,
		baseURL:     polygonBase,
	}
}

// FetchAll fetches options snapshots for every symbol in symbols concurrently,
// respecting the configured concurrency limit via a semaphore pattern.
func (c *Client) FetchAll(ctx context.Context, symbols []string) BatchResult {
	results := make([]IngestionResult, len(symbols))

	var wg sync.WaitGroup
	sem := make(chan struct{}, c.concurrency)

	for i, sym := range symbols {
		wg.Add(1)
		go func(idx int, symbol string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			snapshots, err := c.fetchSymbol(ctx, symbol)
			if err != nil {
				results[idx] = IngestionResult{
					Symbol: symbol,
					Error:  err.Error(),
				}
				return
			}
			results[idx] = IngestionResult{
				Symbol:    symbol,
				Count:     len(snapshots),
				Snapshots: snapshots,
			}
		}(i, sym)
	}

	wg.Wait()

	total := 0
	for _, r := range results {
		total += r.Count
	}
	return BatchResult{Results: results, Total: total}
}

// fetchSymbol fetches all pages of the options snapshot for a single symbol.
func (c *Client) fetchSymbol(ctx context.Context, symbol string) ([]OptionSnapshot, error) {
	url := fmt.Sprintf("%s%s?apiKey=%s&limit=%d",
		c.baseURL,
		fmt.Sprintf(snapshotPathFmt, symbol),
		c.apiKey,
		defaultPageLimit,
	)
	return c.fetchSymbolURL(ctx, url)
}

// fetchSymbolURL fetches all pages starting from a fully-formed URL.
// Extracted so tests can point at an httptest server.
func (c *Client) fetchSymbolURL(ctx context.Context, url string) ([]OptionSnapshot, error) {
	var all []OptionSnapshot
	for page := 0; page < maxPages && url != ""; page++ {
		snapshots, nextURL, err := c.fetchPage(ctx, url)
		if err != nil {
			return all, fmt.Errorf("page %d: %w", page, err)
		}
		all = append(all, snapshots...)
		url = nextURL
	}
	return all, nil
}

// polygonResponse mirrors the relevant fields of the Polygon.io snapshot
// response envelope.
type polygonResponse struct {
	Results []polygonContract `json:"results"`
	NextURL string            `json:"next_url"`
}

type polygonContract struct {
	Details struct {
		Ticker         string  `json:"ticker"`
		ContractType   string  `json:"contract_type"`
		StrikePrice    float64 `json:"strike_price"`
		ExpirationDate string  `json:"expiration_date"`
	} `json:"details"`
	Greeks struct {
		Delta float64 `json:"delta"`
		Gamma float64 `json:"gamma"`
		Theta float64 `json:"theta"`
		Vega  float64 `json:"vega"`
	} `json:"greeks"`
	Day struct {
		Volume       int `json:"volume"`
		OpenInterest int `json:"open_interest"`
	} `json:"day"`
	LastQuote struct {
		Bid float64 `json:"bid"`
		Ask float64 `json:"ask"`
	} `json:"last_quote"`
	ImpliedVolatility float64 `json:"implied_volatility"`
	UnderlyingAsset   struct {
		Ticker string `json:"ticker"`
	} `json:"underlying_asset"`
}

func (c *Client) fetchPage(ctx context.Context, url string) ([]OptionSnapshot, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", fmt.Errorf("building request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("polygon returned %d: %s", resp.StatusCode, string(body))
	}

	var parsed polygonResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, "", fmt.Errorf("decoding response: %w", err)
	}

	now := time.Now()
	snapshots := make([]OptionSnapshot, 0, len(parsed.Results))
	for _, c := range parsed.Results {
		snap := normalizeContract(c, now)
		if snap != nil {
			snapshots = append(snapshots, *snap)
		}
	}

	nextURL := parsed.NextURL
	if nextURL != "" {
		nextURL += "&apiKey=" + c.apiKey
	}
	return snapshots, nextURL, nil
}

func normalizeContract(c polygonContract, now time.Time) *OptionSnapshot {
	ct := c.Details.ContractType
	if ct != "call" && ct != "put" {
		return nil
	}

	expiry, err := time.Parse("2006-01-02", c.Details.ExpirationDate)
	if err != nil {
		return nil
	}
	dte := int(math.Ceil(expiry.Sub(now).Hours() / 24))
	if dte < 0 {
		return nil
	}

	bid := c.LastQuote.Bid
	ask := c.LastQuote.Ask
	if bid <= 0 || ask <= 0 || ask < bid {
		return nil
	}
	mid := (bid + ask) / 2.0

	underlying := c.UnderlyingAsset.Ticker
	if underlying == "" {
		// Fall back: strip prefix from option ticker (e.g. "O:AAPL..." → use caller)
		underlying = "UNKNOWN"
	}

	return &OptionSnapshot{
		Ticker:       c.Details.Ticker,
		Underlying:   underlying,
		ContractType: ct,
		StrikePrice:  c.Details.StrikePrice,
		Expiry:       c.Details.ExpirationDate,
		DTE:          dte,
		Bid:          bid,
		Ask:          ask,
		MidPrice:     math.Round(mid*10000) / 10000,
		Delta:        math.Abs(c.Greeks.Delta),
		Gamma:        c.Greeks.Gamma,
		Theta:        c.Greeks.Theta,
		Vega:         c.Greeks.Vega,
		IV:           c.ImpliedVolatility,
		Volume:       c.Day.Volume,
		OpenInterest: c.Day.OpenInterest,
	}
}
