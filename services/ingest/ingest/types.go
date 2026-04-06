package ingest

// OptionSnapshot represents a single options contract from the Polygon.io
// snapshot API, normalized into the fields consumed by the Python pipeline.
type OptionSnapshot struct {
	Ticker       string  `json:"ticker"`
	Underlying   string  `json:"underlying"`
	ContractType string  `json:"contract_type"`
	StrikePrice  float64 `json:"strike_price"`
	Expiry       string  `json:"expiration_date"`
	DTE          int     `json:"dte"`
	Bid          float64 `json:"bid"`
	Ask          float64 `json:"ask"`
	MidPrice     float64 `json:"mid_price"`
	Delta        float64 `json:"delta"`
	Gamma        float64 `json:"gamma"`
	Theta        float64 `json:"theta"`
	Vega         float64 `json:"vega"`
	IV           float64 `json:"implied_volatility"`
	Volume       int     `json:"volume"`
	OpenInterest int     `json:"open_interest"`
}

// IngestionResult holds the output for a single underlying symbol.
type IngestionResult struct {
	Symbol    string           `json:"symbol"`
	Count     int              `json:"count"`
	Snapshots []OptionSnapshot `json:"snapshots"`
	Error     string           `json:"error,omitempty"`
}

// BatchResult is the top-level output of a multi-symbol ingestion run.
type BatchResult struct {
	Results []IngestionResult `json:"results"`
	Total   int               `json:"total"`
}
