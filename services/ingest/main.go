package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/AadityaMishra1/Contrarian-Options-Alpha-Engine/services/ingest/ingest"
)

func main() {
	symbols := flag.String("symbols", "", "comma-separated list of underlying tickers (e.g. AAPL,TSLA,NVDA)")
	concurrency := flag.Int("concurrency", 5, "max parallel API requests")
	outFile := flag.String("out", "", "output file path (default: stdout)")
	flag.Parse()

	apiKey := os.Getenv("POLYGON_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "error: POLYGON_API_KEY environment variable is required")
		os.Exit(1)
	}

	if *symbols == "" {
		fmt.Fprintln(os.Stderr, "error: -symbols flag is required (e.g. -symbols AAPL,TSLA,NVDA)")
		os.Exit(1)
	}

	tickers := strings.Split(*symbols, ",")
	for i := range tickers {
		tickers[i] = strings.TrimSpace(tickers[i])
	}

	client := ingest.NewClient(apiKey, *concurrency)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	fmt.Fprintf(os.Stderr, "ingesting options data for %d symbols (concurrency=%d)...\n", len(tickers), *concurrency)
	start := time.Now()
	result := client.FetchAll(ctx, tickers)
	elapsed := time.Since(start)

	fmt.Fprintf(os.Stderr, "done: %d contracts across %d symbols in %s\n", result.Total, len(result.Results), elapsed.Round(time.Millisecond))

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error marshaling output: %v\n", err)
		os.Exit(1)
	}

	if *outFile != "" {
		if err := os.WriteFile(*outFile, data, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "error writing to %s: %v\n", *outFile, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "output written to %s\n", *outFile)
	} else {
		fmt.Println(string(data))
	}
}
