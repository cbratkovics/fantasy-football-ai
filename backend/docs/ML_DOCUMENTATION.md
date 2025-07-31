# Machine Learning Documentation

## Model Overview
Production ML models trained on 31,000+ NFL records from 2019-2024, achieving realistic and reliable predictions for fantasy football.

## Model Performance

### Current Production Models (as of 2025-07-31)
| Position | MAE | Baseline MAE | Improvement | Status |
|----------|-----|--------------|-------------|---------|
| QB | 6.17 | 7.54 | 18.1% | ✅ Production |
| RB | 4.92 | 6.49 | 24.1% | ✅ Production |
| WR | 4.99 | 6.26 | 20.3% | ✅ Production |
| TE | 3.88 | 4.71 | 17.5% | ✅ Production |

## Data Pipeline

### Sources
1. **nfl_data_py** - Historical player stats, play-by-play data
2. **Open-Meteo** - Weather conditions for outdoor games
3. **ESPN API** - Fantasy projections and player news

### Feature Engineering
All features are lagged to prevent data leakage:
- **L1W**: Previous week stats
- **L3W**: 3-week rolling average
- **L5W**: 5-week rolling average
- **Season**: Season-to-date average
- **Efficiency**: Position-specific metrics (completion %, yards/carry, etc.)

### Training Process
1. Temporal split: Train (2019-2022), Validation (2023), Test (2024)
2. Position-specific models for better accuracy
3. Random Forest selected after comparing with XGBoost
4. Proper validation ensuring no future data leakage

## Model Files
Located in `models/production/`:
- `prod_{position}_model_{timestamp}.pkl` - Trained models
- `prod_{position}_scaler_{timestamp}.pkl` - Feature scalers
- `prod_models_metadata_{timestamp}.json` - Model metadata

## Usage Example
```python
import joblib

# Load model
model = joblib.load('models/production/prod_QB_model_20250731_171728.pkl')
scaler = joblib.load('models/production/prod_QB_scaler_20250731_171728.pkl')

# Prepare features (must be lagged)
features = prepare_lagged_features(player_data)
X_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(X_scaled)
```

## Model Validation
- No same-week statistics used
- All features available before game time
- Realistic MAE values (no suspiciously low errors)
- Proper temporal validation
- Regular monitoring for drift in production

## Future Improvements
1. Add injury report integration
2. Include team matchup difficulty
3. Weather impact modeling
4. Ensemble methods for better predictions