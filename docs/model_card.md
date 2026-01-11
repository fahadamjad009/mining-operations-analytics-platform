Model Card - Downtime Risk Model



Model Type

Logistic Regression (baseline)



Objective

Predict the probability that an asset will experience high downtime (>= 120 minutes) on the next day.



Input Features

\- avg\_temp

\- avg\_vib

\- avg\_fuel

\- utilization\_rate

\- downtime\_rate

\- work\_orders

\- high\_sev



Output

downtime\_risk\_probability between 0 and 1



Training Data

\- Synthetic mining telemetry and maintenance logs

\- Optionally NASA C-MAPSS public benchmark dataset



Performance

Baseline ROC AUC approximately 0.61 on synthetic mining dataset.



Limitations

\- Trained on synthetic data

\- Does not capture all real-world failure modes

\- Should not be used for safety-critical decisions without further validation



Ethics and Safety

\- No personal data is used

\- Model is intended to support maintenance planning, not replace human judgement



