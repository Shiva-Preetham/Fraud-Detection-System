# Power BI Measures

Use the gold scored transaction table as the primary fact table.

```DAX
Transactions Today =
COUNTROWS('scored_transactions')

Fraud Flagged =
CALCULATE(
    COUNTROWS('scored_transactions'),
    'scored_transactions'[prediction] = "FRAUD"
)

Average Risk Score =
AVERAGE('scored_transactions'[risk_score])

Blocked Value =
CALCULATE(
    SUM('scored_transactions'[amount]),
    'scored_transactions'[prediction] = "FRAUD"
)

Precision =
DIVIDE([True Positives], [True Positives] + [False Positives])

Recall =
DIVIDE([True Positives], [True Positives] + [False Negatives])

F1 Score =
DIVIDE(2 * [Precision] * [Recall], [Precision] + [Recall])
```
