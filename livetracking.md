The Live Tracking Workflow
Step 1: Before the Event (Wednesday/Thursday)
Run predictions for the upcoming card:
cd DelphiAIApp/Modelspython -m ml.predict_card "Strickland vs Hernandez"
This does two things:
Prints the full card predictions (picks, probabilities, methods, rounds)
Automatically saves all 14 predictions to the PredictionTracking table in PostgreSQL with prediction_type = 'live'
Step 2: After the Event (Sunday)
Compare our picks to what actually happened:
python -m ml.update_results "Strickland vs Hernandez"
This:
Finds our stored predictions in the database
Scrapes UFC.com for the actual results (winner, method, round)
Matches each result to our prediction
Marks each as correct/incorrect
Prints a per-event performance report with tier breakdowns and paper trading ROI
Step 3: Check Cumulative Performance (Anytime)
python -m ml.performance_summary
This pulls ALL resolved predictions from the database and shows:
Separate sections for BACKTEST (historical) and LIVE (2026 ongoing)
Accuracy by confidence tier
Calibration check
Cumulative paper trading ROI
A comparison between backtest and live to see if performance is holding
Where We Are Right Now
Backtest (complete): 664 fights across 2023-2025, 64.2% overall accuracy, validated.
Live tracking (just started): 1 event tracked — Strickland vs Hernandez, 8/14 (57.1%). That's our first live data point.
The Problem We Hit Today
The tracking system was built after we originally ran the Strickland card predictions. So when we ran predict_card again today to save them, the predictions were different because:
ELO ratings had already been updated with post-fight results
Injury reports changed after Saturday
Some fighters got different injury penalties
We had to manually fix the database to match the original pre-fight predictions from your terminal history.
Going forward this won't happen — predictions are now saved automatically the moment you run predict_card. The flow is clean: predict before the fight, results get saved, update after the fight, everything matches.
What Happens Over the Next Few Months
After each event, you run update_results. After 3-5 events you'll have ~50-70 live predictions. Then performance_summary becomes meaningful:
Target after 50 live fights:  Overall accuracy:    58-65%  High confidence:     70%+  Divergence from backtest: <5%  Paper trading ROI:   PositiveIf those hold → model is validated for real bettingIf accuracy drops below 55% for 50+ fights → consider retraining
The database keeps everything timestamped, so you can always go back and see exactly what you predicted and when.