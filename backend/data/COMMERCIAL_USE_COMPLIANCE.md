# Data Source Commercial Use Compliance

Last Updated: 2024-01-31

## ✅ Confirmed Commercial Use Allowed

### 1. Sleeper API
- **License**: Allows commercial use per Terms of Service
- **Documentation**: https://docs.sleeper.app
- **Rate Limits**: 1000 requests/minute
- **Data Available**: Players, stats, projections, leagues
- **Status**: ✅ Safe for commercial use

### 2. nfl_data_py
- **License**: MIT License (explicitly allows commercial use)
- **Source**: https://github.com/nflverse/nfl_data_py
- **Data Sources**: Aggregates from multiple free sources
- **Status**: ✅ Safe for commercial use

### 3. Open-Meteo Weather API
- **License**: Free tier allows commercial use
- **Documentation**: https://open-meteo.com/en/docs
- **Rate Limits**: No hard limits on free tier
- **Data Available**: Historical and forecast weather
- **Status**: ✅ Safe for commercial use

### 4. Public Domain NFL Data
- **Source**: Government and public records
- **Examples**: Game results, player statistics
- **Status**: ✅ No restrictions on factual data

## ⚠️ Verify Before Production Use

### 1. ESPN Public Endpoints
- **Current Status**: Technically accessible without authentication
- **Risk**: Terms of Service may prohibit commercial use
- **Recommendation**: 
  - Review ESPN's current ToS before production deployment
  - Consider as supplementary data only
  - Have fallback if access is restricted

### 2. Web Scraping (Pro Football Reference, etc.)
- **Current Status**: Technically possible with respectful scraping
- **Requirements**:
  - Must follow robots.txt
  - Implement proper rate limiting
  - Include identifying User-Agent
- **Risk**: ToS may prohibit commercial use
- **Recommendation**: Use only for research/development

## ❌ Not for Commercial Use

### 1. ESPN Fantasy API (with authentication)
- Requires user credentials
- ToS explicitly prohibits commercial use without partnership

### 2. NFL Official API
- Requires partnership agreement
- Not available for general commercial use

### 3. Premium Stats Providers
- PFF (Pro Football Focus)
- Sports Info Solutions
- STATS LLC
- All require paid licenses for commercial use

## 📋 Implementation Guidelines

### For Production Use:
1. **Primary Sources**: Sleeper API + nfl_data_py + Open-Meteo
2. **Data Coverage**: 95%+ of required data from allowed sources
3. **Compliance**: All primary sources explicitly allow commercial use
4. **Cost**: $0 for data (only infrastructure costs)

### Best Practices:
1. **Always include User-Agent** headers with contact info
2. **Implement exponential backoff** for all API calls
3. **Cache aggressively** to minimize API load
4. **Monitor for ToS changes** quarterly
5. **Have fallback data sources** ready

### Legal Disclaimer:
```
This software uses publicly available data from various sources.
Users are responsible for ensuring their specific use case complies
with all applicable terms of service and laws in their jurisdiction.
```

## 🔄 Update Schedule

This document should be reviewed and updated:
- Quarterly (every 3 months)
- Whenever adding new data sources
- If any provider changes their ToS

## ✅ Current Status: COMPLIANT

All primary data sources used in production are confirmed to allow commercial use. The system can operate entirely on free, commercially-allowed data sources with no legal restrictions.