package ingest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fakePolygonResponse builds a minimal Polygon-format JSON response with n contracts.
func fakePolygonResponse(t *testing.T, n int, nextURL string) []byte {
	t.Helper()
	type result struct {
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
		LastQuote struct {
			Bid float64 `json:"bid"`
			Ask float64 `json:"ask"`
		} `json:"last_quote"`
		Day struct {
			Volume       int `json:"volume"`
			OpenInterest int `json:"open_interest"`
		} `json:"day"`
		ImpliedVolatility float64 `json:"implied_volatility"`
		UnderlyingAsset   struct {
			Ticker string `json:"ticker"`
		} `json:"underlying_asset"`
	}

	expiry := time.Now().AddDate(0, 0, 30).Format("2006-01-02")
	results := make([]result, n)
	for i := range n {
		r := &results[i]
		r.Details.Ticker = "O:AAPL251219C00150000"
		r.Details.ContractType = "call"
		r.Details.StrikePrice = 150.0 + float64(i)
		r.Details.ExpirationDate = expiry
		r.Greeks.Delta = 0.30
		r.Greeks.Gamma = 0.02
		r.Greeks.Theta = -0.05
		r.Greeks.Vega = 0.12
		r.LastQuote.Bid = 3.50
		r.LastQuote.Ask = 3.80
		r.Day.Volume = 1200
		r.Day.OpenInterest = 5000
		r.ImpliedVolatility = 0.35
		r.UnderlyingAsset.Ticker = "AAPL"
	}

	resp := struct {
		Results []result `json:"results"`
		NextURL string   `json:"next_url,omitempty"`
	}{Results: results, NextURL: nextURL}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("failed to marshal fake response: %v", err)
	}
	return data
}

func TestFetchAll_SingleSymbol(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(fakePolygonResponse(t, 3, ""))
	}))
	defer srv.Close()

	// Override the polygon base URL by swapping the client's HTTP client to
	// redirect requests to the test server.
	client := NewClient("test-key", 2)
	client.httpClient = srv.Client()

	// Monkey-patch: use a custom fetchSymbol that hits the test server.
	snapshots, err := client.fetchSymbolURL(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(snapshots) != 3 {
		t.Errorf("expected 3 snapshots, got %d", len(snapshots))
	}
	for _, s := range snapshots {
		if s.ContractType != "call" {
			t.Errorf("expected contract_type=call, got %q", s.ContractType)
		}
		if s.Underlying != "AAPL" {
			t.Errorf("expected underlying=AAPL, got %q", s.Underlying)
		}
		if s.Delta != 0.30 {
			t.Errorf("expected delta=0.30, got %f", s.Delta)
		}
	}
}

func TestFetchAll_Pagination(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		if callCount == 1 {
			// First page: return 2 results + next_url pointing back to this server.
			nextURL := "http://" + r.Host + "/v3/snapshot/options/AAPL?cursor=page2"
			w.Write(fakePolygonResponse(t, 2, nextURL))
		} else {
			// Second page: return 1 result, no next_url.
			w.Write(fakePolygonResponse(t, 1, ""))
		}
	}))
	defer srv.Close()

	client := NewClient("test-key", 2)
	client.httpClient = srv.Client()

	snapshots, err := client.fetchSymbolURL(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(snapshots) != 3 {
		t.Errorf("expected 3 snapshots across 2 pages, got %d", len(snapshots))
	}
	if callCount != 2 {
		t.Errorf("expected 2 HTTP requests (pagination), got %d", callCount)
	}
}

func TestFetchAll_ConcurrentSymbols(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(fakePolygonResponse(t, 2, ""))
	}))
	defer srv.Close()

	client := NewClient("test-key", 3)
	client.httpClient = srv.Client()
	// Override base URL for FetchAll.
	client.baseURL = srv.URL

	result := client.FetchAll(context.Background(), []string{"AAPL", "TSLA", "NVDA"})
	if len(result.Results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result.Results))
	}
	if result.Total != 6 {
		t.Errorf("expected total=6, got %d", result.Total)
	}
	for _, r := range result.Results {
		if r.Error != "" {
			t.Errorf("unexpected error for %s: %s", r.Symbol, r.Error)
		}
		if r.Count != 2 {
			t.Errorf("expected 2 snapshots for %s, got %d", r.Symbol, r.Count)
		}
	}
}

func TestNormalizeContract_InvalidData(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name string
		mod  func(*polygonContract)
	}{
		{"bad contract type", func(c *polygonContract) { c.Details.ContractType = "swap" }},
		{"expired", func(c *polygonContract) { c.Details.ExpirationDate = "2020-01-01" }},
		{"zero bid", func(c *polygonContract) { c.LastQuote.Bid = 0 }},
		{"ask < bid", func(c *polygonContract) { c.LastQuote.Ask = 1.0; c.LastQuote.Bid = 5.0 }},
		{"bad date format", func(c *polygonContract) { c.Details.ExpirationDate = "not-a-date" }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := validContract()
			tt.mod(&c)
			if snap := normalizeContract(c, now); snap != nil {
				t.Error("expected nil for invalid contract, got non-nil")
			}
		})
	}
}

func validContract() polygonContract {
	var c polygonContract
	c.Details.Ticker = "O:AAPL251219C00150000"
	c.Details.ContractType = "call"
	c.Details.StrikePrice = 150.0
	c.Details.ExpirationDate = time.Now().AddDate(0, 0, 30).Format("2006-01-02")
	c.Greeks.Delta = 0.30
	c.LastQuote.Bid = 3.50
	c.LastQuote.Ask = 3.80
	c.UnderlyingAsset.Ticker = "AAPL"
	return c
}
