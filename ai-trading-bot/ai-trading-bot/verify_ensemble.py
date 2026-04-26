from src.data.preprocess import preprocess_symbol
from src.models.ensemble import EnsembleModel

df = preprocess_symbol('INFY')
print(f'INFY data: {len(df)} rows')

ensemble = EnsembleModel(min_votes=3, backtest_mode=True)
ensemble.load_models()

signal = ensemble.predict(df, 'INFY')
print(f'Direction: {signal.direction}')
print(f'Confidence: {signal.confidence}')
print(f'Votes UP: {signal.votes_up} | DOWN: {signal.votes_down}')
print(f'Regime: {signal.regime}')
print(f'Veto: {signal.veto} ({signal.veto_reason})')
print()
print('Individual votes:')
for name, sig in signal.model_signals.items():
    vote = sig.get('vote', 'N/A')
    conf = sig.get('confidence', 0)
    avail = sig.get('available', False)
    print(f'  {name:>15s}: vote={str(vote):>5s}  conf={conf:.3f}  available={avail}')
