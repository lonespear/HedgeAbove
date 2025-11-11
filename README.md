# HedgeAbove

**Rise Above Market Uncertainty**

AI-powered finance analytics platform for market predictions, risk management, and hedging strategies.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- pip package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd HedgeAbove
```

2. **Create a virtual environment:**

Windows:
```bash
python -m venv venv
```

Mac/Linux:
```bash
python3 -m venv venv
```

3. **Activate the virtual environment:**

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Open your browser:**
The app will automatically open at `http://localhost:8501`

---

## 📋 Project Structure

```
HedgeAbove/
├── app.py                    # Main Streamlit application (all-in-one)
├── requirements.txt          # Python dependencies
├── hedge_above_logo.png     # Application logo
├── .gitignore               # Git ignore rules
├── README.md                # This file
│
├── venv/                    # Virtual environment (created during setup)
│
├── .streamlit/              # Streamlit configuration (optional)
│   └── config.toml
│
├── .env                     # Environment variables (optional, not in git)
│
├── models/                  # Saved ML models (auto-created)
├── data/                    # Cached market data (auto-created)
└── tests/                   # Unit tests (coming soon)
```

**Note:** Currently, all functionality is contained in `app.py` as a single-file Streamlit application. Future versions may split into multiple modules.

---

## 🎯 Current Features

### ✅ Implemented
- **Stock Screener** - Comprehensive fundamental analysis
  - 2,535+ global stocks (Russell 2000 + International)
  - 50+ fundamental metrics (BVPS, EPS, ROE, ROIC, margins, etc.)
  - 4 screening presets (Value, Growth, Quality, Dividend)
  - Advanced filtering by valuation and profitability
  - 6 organized tabs for different metric categories
  - Analyst targets and recommendations

- **Portfolio Builder** - Real-time portfolio management
  - Add/Edit/Delete positions with live market data
  - Automatic P&L tracking via yfinance
  - Quick-add from screener

- **Risk Analytics** - Comprehensive risk calculations
  - Value at Risk (Historical, Parametric, Monte Carlo)
  - Expected Shortfall (CVaR)
  - Correlation matrices and heatmaps
  - Portfolio metrics (Sharpe, beta, volatility)

- **Portfolio Optimization** - Modern Portfolio Theory
  - Efficient Frontier visualization
  - Max Sharpe Ratio optimization
  - Minimum Volatility optimization
  - Target Return optimization
  - Risk Parity allocation

- **AI Predictions** - Time series forecasting
  - LSTM neural networks
  - Facebook Prophet
  - ARIMA with auto parameter selection
  - Ensemble models

- **Advanced Models** - Sophisticated analytics
  - ARCH/GARCH volatility modeling
  - Copula analysis for tail dependence
  - Regime detection
  - Monte Carlo simulations

### 🚧 Coming Soon (Next 4-6 Weeks)
- Options pricing with Greeks (Black-Scholes)
- Enhanced backtesting framework
- User authentication and data persistence
- Freemium subscription gates
- Data export (CSV/Excel)
- Email/SMS alerts

---

## 📅 Development Roadmap

### Week 1-2: Foundation ✅ COMPLETE
- [x] Project setup and Streamlit demo
- [x] Market data integration (yfinance)
- [x] Portfolio data model and CRUD operations
- [x] Comprehensive stock screener (2,535+ stocks)

### Week 3-4: Risk Analytics ✅ COMPLETE
- [x] Implement VaR calculations (Historical, Parametric, Monte Carlo)
- [x] Expected Shortfall (CVaR)
- [x] Correlation matrix from real data
- [x] Portfolio metrics (Sharpe, beta, volatility)
- [x] Efficient frontier and optimization

### Week 5-6: AI/ML Models ✅ COMPLETE
- [x] Data pipeline for training
- [x] LSTM model for price prediction
- [x] Facebook Prophet integration
- [x] ARIMA with auto parameter selection
- [x] ARCH/GARCH volatility models
- [x] Copula analysis for tail risk
- [x] Ensemble model combining multiple approaches

### Week 7: Options & Hedging 🚧 IN PROGRESS
- [ ] Black-Scholes options pricing
- [ ] Greeks calculator (Delta, Gamma, Theta, Vega, Rho)
- [ ] Options strategy builder (Straddles, Spreads, Collars)
- [ ] Portfolio hedge recommendations

### Week 8: Polish & Launch 🔜 NEXT
- [ ] User authentication (Streamlit Authenticator)
- [ ] Freemium gates (usage limits)
- [ ] Database setup for data persistence
- [ ] UI/UX improvements
- [ ] Documentation and tutorials
- [ ] Beta testing and deployment

---

## 🔑 API Keys (Optional)

HedgeAbove works out-of-the-box with **Yahoo Finance** via yfinance - no API keys required!

### Optional API Keys (for extended functionality):

1. **Alpha Vantage** (Optional - for alternative data sources)
   - Free tier: 5 calls/min, 500 calls/day
   - Sign up: https://www.alphavantage.co/support/#api-key
   - Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key`

2. **Polygon.io** (Optional - for real-time enterprise data)
   - Free tier: 5 calls/min
   - Sign up: https://polygon.io/
   - Add to `.env`: `POLYGON_API_KEY=your_key`

### Setting up environment variables (optional):
```bash
# Create .env file in project root
echo "ALPHA_VANTAGE_API_KEY=your_key_here" > .env
echo "POLYGON_API_KEY=your_key_here" >> .env
```

**Note:** The app currently uses yfinance for all data, so API keys are not required for basic functionality.

---

## 💡 Usage Guide

HedgeAbove is a Streamlit web application - no coding required! Simply run the app and use the web interface.

### 1. Stock Screener
- Select **Stock Screener** from the sidebar
- Choose a screening strategy preset (Value, Growth, Quality, Dividend)
- Adjust filters (sector, P/E, ROE, ROIC, etc.)
- Browse results across 6 organized tabs
- Add stocks directly to your portfolio

### 2. Portfolio Builder
- Select **Portfolio Builder** from the sidebar
- Enter a stock ticker (e.g., AAPL, MSFT, TSLA)
- Specify shares and purchase price
- View real-time P&L and portfolio value

### 3. Risk Analytics
- Select **Risk Analytics** from the sidebar
- View VaR (Historical, Parametric, Monte Carlo)
- Analyze correlation matrix and portfolio metrics
- See Expected Shortfall (CVaR)

### 4. Portfolio Optimization
- Select **Portfolio Optimization** from the sidebar
- Visualize Efficient Frontier
- Find optimal allocations (Max Sharpe, Min Volatility, etc.)
- Implement Risk Parity strategies

### 5. AI Predictions
- Select **AI Predictions** from the sidebar
- Choose a ticker and prediction model (LSTM, Prophet, ARIMA, etc.)
- Set forecast horizon
- View predictions with confidence intervals

### Command Line Usage (Advanced)
```bash
# Run the app
streamlit run app.py

# Run on a specific port
streamlit run app.py --server.port 8502

# Run with custom config
streamlit run app.py --server.headless true
```

---

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

---

## 📊 Technology Stack

| Category | Technology |
|----------|-----------|
| **Framework** | Streamlit |
| **Language** | Python 3.9+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | Plotly, matplotlib |
| **Machine Learning** | TensorFlow, scikit-learn, Prophet |
| **Market Data** | yfinance, Alpha Vantage, Polygon |
| **Options Pricing** | py_vollib (Black-Scholes) |
| **Database** | PostgreSQL / SQLite |
| **Deployment** | Streamlit Cloud / AWS / Heroku |

---

## 🎨 Design Philosophy

1. **Speed to Market** - MVP in 8 weeks, iterate based on feedback
2. **ML-First** - AI predictions and analytics as core differentiator
3. **Freemium Model** - Free tier to attract users, premium for power features
4. **Data-Driven** - Everything backed by real market data and backtesting
5. **Professional UX** - Clean, intuitive interface for retail and pros

---

## 📈 Monetization Strategy

### Free Tier
- 1 portfolio
- 5 predictions per day
- Basic risk analytics
- Delayed market data (15 minutes)

### Pro Tier ($15/month)
- Unlimited portfolios
- Unlimited predictions
- Real-time market data
- Advanced ML models
- Options Greeks and strategies
- Data export (CSV/Excel)

### Premium Tier ($49/month)
- Everything in Pro
- Custom ML model training
- API access
- Priority data feeds
- Advanced hedging strategies
- Email/SMS alerts

---

## 🤝 Contributing

This is currently a solo project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is proprietary. All rights reserved.

---

## 🐛 Known Issues

- [ ] No user authentication - single-user mode only
- [ ] No data persistence between sessions (portfolio resets on refresh)
- [ ] Options pricing and Greeks not yet implemented
- [ ] Large screener queries (100+ stocks) can take 1-2 minutes
- [ ] Some international stocks may have incomplete fundamental data
- [ ] pmdarima not compatible with Python 3.13+ (falls back to statsmodels)

---

## 📞 Support

For questions or issues:
- Email: support@hedgeabove.com (placeholder)
- GitHub Issues: [Create an issue]
- Documentation: [Coming soon]

---

## 🎯 Success Metrics (First 3 Months)

- [ ] 100+ active users
- [ ] 10+ paying subscribers
- [ ] 80%+ prediction accuracy (backtested)
- [ ] < 2 second page load time
- [ ] 4.5+ star user rating

---

## 🔮 Future Vision

### Phase 2: Desktop App (Months 4-6)
- Electron wrapper for power users
- Offline analytics
- Advanced charting

### Phase 3: Mobile App (Months 7-10)
- React Native companion app
- Push notifications
- Quick portfolio checks

### Phase 4: Social Features (Months 11-12)
- Share strategies
- Leaderboards
- Community predictions

---

**Ready to rise above market uncertainty? 📈**

---

*Last Updated: 2025-01-11*
