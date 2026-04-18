# Cybersecurity Breach Disclosure Event Study

Replicating the finding from arXiv:2512.06144v1 that small-cap stocks drop ~7.5% after disclosing a cybersecurity breach via SEC Form 8-K Item 1.05.

## Results

Small caps (<$2B market cap) averaged -7.49% abnormal return in the 7-day window after disclosure. Statistically significant (t=-3.13, p=0.0019). Large caps showed no significant reaction (+0.43%).

## How it works

Fetches all Item 1.05 filings from SEC EDGAR since 2023 (when the rule took effect), resolves each to a stock ticker, downloads price data from Yahoo Finance, and computes market-adjusted abnormal returns using SPY as the benchmark.

Also runs a hypothetical short strategy on small caps to see if the finding is tradeable.

## How to run

```bash
pip install -r requirements.txt
python3 cybersecurity_event_study.py
```

Takes a few minutes on first run because of EDGAR rate limits and price downloads. Results get cached locally so subsequent runs are faster.

## Output

- `results/results.json` - Stats, t-tests, strategy metrics
- `results/events.csv` - Every filing with ticker, market cap, and abnormal returns

## Data sources

SEC EDGAR (free, public) for filings and ticker lookup. Yahoo Finance (free, via yfinance) for prices.

## License

MIT
