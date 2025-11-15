# api.py
# ================================================================
# 💰 AI Portfolio Advisory System — FastAPI + simple browser UI
# ================================================================
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Optional imports (safe fallbacks)
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

try:
    import scipy.optimize as sco
except Exception:
    sco = None


# -----------------------------
# Risk Profiler
# -----------------------------
class RiskProfiler:
    RISK_TOLERANCE_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8}

    def __init__(self, weight_risk_pref=0.6, weight_horizon=0.3, weight_s2i=0.1):
        self.w_risk = weight_risk_pref
        self.w_horizon = weight_horizon
        self.w_s2i = weight_s2i

    def score(self, risk_tolerance, horizon_years, savings_to_income=None):
        rt_val = self.RISK_TOLERANCE_MAP.get(risk_tolerance.lower(), 0.5)
        horizon_score = np.clip(horizon_years / 30.0, 0.0, 1.0)
        s2i_score = np.clip(savings_to_income or 0.2, 0.0, 1.0)
        raw = self.w_risk * rt_val + self.w_horizon * horizon_score + self.w_s2i * s2i_score
        return float(np.clip(raw, 0.0, 1.0))


# -----------------------------
# Sentiment Analyzer (stub)
# -----------------------------
class SentimentAnalyzer:
    def score_texts(self, texts):
        if not texts:
            return []
        rng = np.random.RandomState(42)
        return list(rng.uniform(-0.2, 0.2, size=len(texts)))


# -----------------------------
# Market Forecaster
# -----------------------------
class MarketForecaster:
    def __init__(self):
        self.scaler = StandardScaler() if StandardScaler else None

    @staticmethod
    def _annualize_returns(daily_returns, periods_per_year=252):
        if len(daily_returns) == 0:
            return 0.0, 0.0
        mean_daily = np.nanmean(daily_returns)
        vol_daily = np.nanstd(daily_returns, ddof=1)
        mean_ann = (1 + mean_daily) ** periods_per_year - 1
        vol_ann = vol_daily * np.sqrt(periods_per_year)
        return float(mean_ann), float(vol_ann)

    def forecast_arima(self, asset_series, steps=252):
        if asset_series.dropna().shape[0] < 10 or ARIMA is None:
            returns = asset_series.pct_change().dropna().values
            return self._annualize_returns(returns)
        try:
            y = np.log(asset_series.dropna())
            model = ARIMA(y, order=(1, 1, 0))
            fitted = model.fit()
            forecast_log = fitted.forecast(steps=steps)
            daily_forecast_returns = np.diff(forecast_log) / forecast_log[:-1]
            return self._annualize_returns(daily_forecast_returns)
        except Exception:
            returns = asset_series.pct_change().dropna().values
            return self._annualize_returns(returns)

    def batch_forecast(self, price_df):
        results = {}
        for col in price_df.columns:
            series = price_df[col].dropna()
            exp_r, vol = self.forecast_arima(series)
            results[col] = {"expected_return": exp_r, "volatility": vol}
        return results


# -----------------------------
# Portfolio Optimizer
# -----------------------------
class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.06):
        self.rf = risk_free_rate

    def maximize_sharpe(self, expected_returns, cov, bounds=None):
        if sco is None:
            raise RuntimeError("scipy is required for optimization")

        n = len(expected_returns)
        if bounds is None:
            bounds = tuple((0.0, 1.0) for _ in range(n))

        def portfolio_perf(w):
            ret = float(np.dot(w, expected_returns))
            vol = float(np.sqrt(w.T @ cov @ w))
            sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
            return ret, vol, sharpe

        def neg_sharpe(w):
            return -portfolio_perf(w)[2]

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        x0 = np.ones(n) / n
        opt = sco.minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method="SLSQP")
        w = np.clip(opt.x, 0, 1)
        ret, vol, sharpe = portfolio_perf(w)
        return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe}


# -----------------------------
# Explainability Engine
# -----------------------------
class ExplainabilityEngine:
    @staticmethod
    def explain(user_input, portfolio_metrics, allocations):
        risk = user_input.get("risk_tolerance", "medium").capitalize()
        goal = user_input.get("goal", "balanced").capitalize()
        risk_score = portfolio_metrics["risk_score"]
        sharpe = portfolio_metrics["sharpe"]

        reasoning = f"Based on your **{risk.lower()} risk tolerance** and goal of **{goal.lower()}**, "
        if risk_score < 0.4:
            reasoning += "the portfolio focuses on low-volatility assets."
        elif risk_score < 0.7:
            reasoning += "the portfolio balances stability and growth."
        else:
            reasoning += "the portfolio emphasizes high-return assets."

        sharpe_text = (
            "Excellent risk-adjusted performance." if sharpe > 1 else
            "Moderate performance." if sharpe > 0.5 else
            "High risk relative to reward."
        )

        top_allocs = sorted(allocations, key=lambda x: x["percent"], reverse=True)[:2]
        key_assets = ", ".join([a["asset"] for a in top_allocs]) if top_allocs else "N/A"
        return f"{reasoning} Key allocations: {key_assets}. {sharpe_text}"


# -----------------------------
# Advisory Pipeline
# -----------------------------
class AdvisoryPipeline:
    def __init__(self):
        self.profiler = RiskProfiler()
        self.sentiment = SentimentAnalyzer()
        self.forecaster = MarketForecaster()
        self.optimizer = PortfolioOptimizer()
        self.explainer = ExplainabilityEngine()

    def run(self, user_input, price_df, news_texts=None, amount=10000):
        risk_score = self.profiler.score(
            user_input.get("risk_tolerance", "medium"),
            user_input.get("investment_horizon", 3),
            user_input.get("savings_to_income", None)
        )

        sentiment_signals = {}
        if news_texts:
            for asset, texts in news_texts.items():
                sentiment_signals[asset] = np.mean(self.sentiment.score_texts(texts)) if texts else 0.0

        forecasts = self.forecaster.batch_forecast(price_df)
        combined = {}
        for a, f in forecasts.items():
            adj = f["expected_return"] + 0.5 * sentiment_signals.get(a, 0)
            combined[a] = {"expected_return": adj, "volatility": f["volatility"]}

        tickers = list(price_df.columns)
        returns_df = price_df.pct_change().dropna()
        cov = returns_df.cov().values
        exp_vec = np.array([combined[t]["expected_return"] for t in tickers])

        bounds = [(0, 0.6 if risk_score > 0.6 else 0.4 + 0.2 * risk_score)] * len(tickers)
        opt_res = self.optimizer.maximize_sharpe(exp_vec, cov, bounds=bounds)

        allocations = []
        for t, w in zip(tickers, opt_res["weights"]):
            allocations.append({
                "asset": t,
                "percent": float(w) * 100,
                "amount": float(w) * amount,
                "expected_return": combined[t]["expected_return"],
                "volatility": combined[t]["volatility"]
            })

        portfolio_metrics = {
            "expected_return": opt_res["return"],
            "volatility": opt_res["volatility"],
            "sharpe": opt_res["sharpe"],
            "risk_score": risk_score
        }

        explanation = self.explainer.explain(user_input, portfolio_metrics, allocations)

        return {"portfolio_metrics": portfolio_metrics, "allocations": allocations, "explanation": explanation}


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="AI Portfolio Advisory System")

class UserInput(BaseModel):
    amount: float
    investment_horizon: int
    risk_tolerance: str
    goal: str


@app.get("/", response_class=HTMLResponse)
def home():
    # Simple landing page with link to the UI
    html = """
    <html>
      <head><title>AI Portfolio Advisory</title></head>
      <body>
        <h2>AI Portfolio Advisory System</h2>
        <p><a href="/ui">Open web UI</a> | <a href="/docs">Open API docs</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/ui", response_class=HTMLResponse)
def ui():
    # Inline HTML + JS — posts JSON to /advise and shows the results
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Portfolio Advisor UI</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 2rem auto; }
          label { display:block; margin-top: 0.5rem; }
          input, select { padding: 0.5rem; width: 100%; box-sizing: border-box; }
          button { margin-top: 1rem; padding: 0.6rem 1rem; }
          pre { background:#f6f8fa; padding:1rem; overflow:auto; }
        </style>
      </head>
      <body>
        <h1>Portfolio Advisor</h1>
        <form id="adviseForm">
          <label>Amount (₹): <input type="number" id="amount" value="10000" required /></label>
          <label>Investment horizon (years): <input type="number" id="horizon" value="3" required /></label>
          <label>Risk tolerance:
            <select id="risk">
              <option value="low">Low</option>
              <option value="medium" selected>Medium</option>
              <option value="high">High</option>
            </select>
          </label>
          <label>Goal:
            <select id="goal">
              <option value="growth">Growth</option>
              <option value="income">Income</option>
              <option value="balanced" selected>Balanced</option>
            </select>
          </label>
          <button type="submit">Get Recommendation</button>
        </form>

        <h2>Result</h2>
        <div id="result">No recommendation yet</div>

        <script>
          const form = document.getElementById('adviseForm');
          form.addEventListener('submit', async (e) => {
            e.preventDefault();
            document.getElementById('result').textContent = 'Loading...';
            const payload = {
              amount: Number(document.getElementById('amount').value),
              investment_horizon: Number(document.getElementById('horizon').value),
              risk_tolerance: document.getElementById('risk').value,
              goal: document.getElementById('goal').value
            };
            try {
              const res = await fetch('/advise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
              });
              if (!res.ok) {
                const err = await res.json();
                document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(err, null, 2) + '</pre>';
                return;
              }
              const data = await res.json();
              // pretty-print result
              let html = '<h3>Portfolio Metrics</h3>';
              html += '<pre>' + JSON.stringify(data.portfolio_metrics, null, 2) + '</pre>';
              html += '<h3>Allocations</h3>';
              html += '<pre>' + JSON.stringify(data.allocations, null, 2) + '</pre>';
              html += '<h3>Explanation</h3>';
              html += '<p>' + data.explanation + '</p>';
              document.getElementById('result').innerHTML = html;
            } catch (err) {
              document.getElementById('result').textContent = 'Error: ' + err.toString();
            }
          });
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/advise")
def advise(user: UserInput):
    try:
        # Generate synthetic price data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq="B")
        np.random.seed(0)
        price_df = pd.DataFrame({
            "MutualFunds": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.001, len(dates))),
            "Stocks": 50 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates))),
            "FD": 100 * np.cumprod(1 + np.random.normal(0.00005, 0.0001, len(dates))),
            "Crypto": 10 * np.cumprod(1 + np.random.normal(0.001, 0.05, len(dates))),
        }, index=dates)

        news = {
            "Stocks": ["Market shows optimism"],
            "Crypto": ["Crypto volatility remains high but positive trend"],
            "MutualFunds": ["Steady mutual fund inflows"],
            "FD": ["Stable interest rates on FDs"]
        }

        pipeline = AdvisoryPipeline()
        results = pipeline.run(user.dict(), price_df, news_texts=news, amount=user.amount)
        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
