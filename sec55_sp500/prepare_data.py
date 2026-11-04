"""Prepare traceable daily data; no fitting, simulation, or model training."""
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/sp500_1950_2018_openintro.csv'
OUT = ROOT / 'data/processed'


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT/'docs').mkdir(exist_ok=True)
    with RAW.open() as f:
        rows = list(csv.DictReader(f))
    dates = [r['Date'] for r in rows]
    assert dates == sorted(set(dates)), 'Dates must be sorted and unique'
    for r in rows:
        date.fromisoformat(r['Date'])
        assert date.fromisoformat(r['Date']).weekday() < 5
        for col in ['Open', 'High', 'Low', 'Close', 'Adj.Close']:
            assert math.isfinite(float(r[col])) and float(r[col]) > 0
    selected = [r for r in rows if '2007-06-12' <= r['Date'] <= '2018-01-02']
    write_csv(OUT / 'sp500_prices_20070612_20180102.csv', selected, list(rows[0]))
    fields = ['date', 'previous_date', 'adjusted_close', 'log_return', 'simple_return']
    returns = []
    # First selected price is the base observation, never fabricated as a return.
    for previous, current in zip(selected[:-1], selected[1:]):
        ratio = float(current['Adj.Close']) / float(previous['Adj.Close'])
        returns.append(dict(date=current['Date'], previous_date=previous['Date'],
                            adjusted_close=current['Adj.Close'],
                            log_return=format(math.log(ratio), '.17g'),
                            simple_return=format(ratio - 1, '.17g')))
    write_csv(OUT / 'sp500_returns_20070613_20180102.csv', returns, fields)
    main_prices = [r for r in selected if r['Date'] >= '2013-03-05']
    main_returns = [r for r in returns if r['previous_date'] >= '2013-03-05']
    assert len(main_prices) == 1218 and len(main_returns) == 1217
    write_csv(OUT / 'paper_main_prices_20130305_20180102.csv', main_prices, list(rows[0]))
    main_returns = [dict(r, split_30d='calibration' if i < len(main_returns)-30 else 'historical_test')
                    for i, r in enumerate(main_returns)]
    write_csv(OUT / 'paper_main_returns_1217.csv', main_returns, fields + ['split_30d'])
    windows = []
    for window_id in range(300):
        cutoff = (date(2018, 1, 2) - timedelta(weeks=window_id)).isoformat()
        prices = [r for r in selected if r['Date'] <= cutoff][-1218:]
        assert len(prices) == 1218
        windows.append(dict(window_id=window_id, requested_end=cutoff,
                            price_start=prices[0]['Date'], price_end=prices[-1]['Date'],
                            calibration_last_date=prices[-31]['Date'],
                            test_first_return_date=prices[-30]['Date'],
                            calibration_returns=1187, test_returns=30))
    write_csv(OUT / 'paper_rolling_windows_300.csv', windows, list(windows[0]))
    audit = dict(
        source_page='https://www.openintro.org/data/index.php?data=sp500_1950_2018',
        download_url='https://www.openintro.org/data/csv/sp500_1950_2018.csv',
        acquisition_date='2026-10-28', source_attribution='OpenIntro snapshot; upstream Yahoo! Finance',
        raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
        raw_rows=len(rows), raw_first=dates[0], raw_last=dates[-1],
        selected_price_rows=len(selected), selected_return_rows=len(returns),
        main_price_rows=len(main_prices), main_return_rows=len(main_returns),
        main_calibration_returns=1187, main_test_returns=30,
        adjusted_close_equals_close_in_selected_period=all(r['Adj.Close']==r['Close'] for r in selected),
        latest_window=windows[0], earliest_window=windows[-1],
        validation='Unique increasing weekday dates; positive finite OHLC/adjusted prices; no missing required prices; exact endpoint counts.',
        caveats=[
            'Not the verified original author data file; paper does not name a vendor in Section 5.5.',
            'Adjusted Close label is preserved; equality to Close does not imply a dividend-reinvested total return series.',
            'Returns are decimal daily returns, not percentage units; no forward filling or interpolation.',
            'Rolling reconstruction: shift endpoint by calendar weeks, choose latest available trading day, retain 1218 prices. Inferred convention, not verified author code.',
            '30-day windows overlap; 300 windows are not 300 independent observations.',
            'All model calibration and hyperparameter selection must use pre-test observations only.'
        ],
    )
    (ROOT / 'docs/data_provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
