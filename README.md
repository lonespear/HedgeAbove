# HedgeAbove

**Rise Above Market Uncertainty**

AI-powered finance analytics platform for market predictions, risk management, and hedging strategies.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Navigate to the project directory:**
```bash
cd "C:\Users\jonathan.day\Documents\HedgeAbove"
```

2. **Create a virtual environment:**
```bash
python -m venv venv
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
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore rules
├── PROJECT_OPTIONS.md       # Detailed project planning document
├── README.md                # This file
│
├── pages/                   # Multi-page Streamlit app (coming soon)
│   ├── 1_Portfolio.py
│   ├── 2_Risk_Analytics.py
│   ├── 3_Predictions.py
│   └── 4_Options.py
│
├── src/                     # Source code modules (coming soon)
│   ├── data/               # Data fetching and processing
│   ├── models/             # ML models
│   ├── analytics/          # Risk calculations
│   └── utils/              # Utility functions
│
├── models/                  # Saved ML models
├── data/                    # Cached market data
└── tests/                   # Unit tests
```

---

## 🎯 Current Features (MVP Demo)

### ✅ Implemented
- **Home Dashboard** - Overview of features
- **Portfolio Tracker** - Sample portfolio with P&L tracking
- **Risk Analytics** - VaR, correlation matrix, Sharpe ratio (demo)
- **AI Predictions** - ML prediction interface with visualizations (demo)
- **Options Calculator** - P&L diagrams and break-even analysis (demo)
- **Hedging Strategies** - AI-recommended hedge strategies (demo)

### 🚧 Coming Soon (Next 8 Weeks)
- Real market data integration (yfinance, Alpha Vantage)
- Working ML prediction models (LSTM, Prophet)
- Actual risk calculations (VaR, portfolio optimization)
- Options pricing with Greeks (Black-Scholes)
- User authentication and data persistence
- Freemium subscription system

---

## 📅 Development Roadmap

### Week 1-2: Foundation
- [x] Project setup and Streamlit demo
- [ ] Market data integration (yfinance)
- [ ] Portfolio data model and CRUD operations
- [ ] Database setup (SQLite or PostgreSQL)

### Week 3-4: Risk Analytics
- [ ] Implement VaR calculations (Historical, Monte Carlo)
- [ ] Correlation matrix from real data
- [ ] Portfolio metrics (Sharpe, beta, volatility)
- [ ] Efficient frontier and optimization

### Week 5-6: AI/ML Models
- [ ] Data pipeline for training
- [ ] LSTM model for price prediction
- [ ] Facebook Prophet integration
- [ ] Ensemble model combining multiple approaches
- [ ] Backtesting framework

### Week 7: Options & Hedging
- [ ] Black-Scholes options pricing
- [ ] Greeks calculator
- [ ] Options strategy builder
- [ ] Portfolio hedge recommendations

### Week 8: Polish & Launch
- [ ] User authentication (Streamlit Authenticator)
- [ ] Freemium gates (usage limits)
- [ ] UI/UX improvements
- [ ] Documentation
- [ ] Beta testing and deployment

---

## 🔑 API Keys Required

To use real market data, you'll need API keys from:

1. **Alpha Vantage** (Free tier: 5 calls/min, 500 calls/day)
   - Sign up: https://www.alphavantage.co/support/#api-key
   - Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key`

2. **Polygon.io** (Free tier: 5 calls/min)
   - Sign up: https://polygon.io/
   - Add to `.env`: `POLYGON_API_KEY=your_key`

3. **Yahoo Finance** (via yfinance - no key required)
   - Free, unlimited for delayed data

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

---

## 💡 Usage Examples

### Adding a Portfolio Position
```python
import yfinance as yf

# Fetch real-time data
ticker = yf.Ticker("AAPL")
current_price = ticker.info['currentPrice']

# Add to portfolio
portfolio.add_position("AAPL", shares=50, avg_price=150.00)
```

### Running ML Prediction
```python
from src.models.lstm_predictor import LSTMPredictor

# Train model
predictor = LSTMPredictor(symbol="AAPL")
predictor.train(periods=252)  # 1 year of data

# Make prediction
forecast = predictor.predict(horizon=30)  # 30 days ahead
```

### Calculating VaR
```python
from src.analytics.risk import calculate_var

# Historical VaR
var_95 = calculate_var(portfolio, confidence=0.95, method='historical')

# Monte Carlo VaR
var_95_mc = calculate_var(portfolio, confidence=0.95, method='monte_carlo', simulations=10000)
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

- [ ] Demo data only - no real market integration yet
- [ ] ML models not trained - showing simulated predictions
- [ ] No user authentication - single-user mode only
- [ ] Options pricing uses placeholder values
- [ ] No data persistence between sessions

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

**Ready to rise above market uncertainty? Let's build HedgeAbove! 📈**

---

*Last Updated: 2025-11-11*
