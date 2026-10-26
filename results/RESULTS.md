| Model | Split | FNC acc | FNC macro F1 | FNC AUC | LIAR acc | LIAR macro F1 | LIAR AUC |
|---|---|---|---|---|---|---|---|
| tfidf_logreg | random | 0.875 | 0.875 | 0.952 | 0.521 | 0.519 | 0.535 |
| roberta-base | random | 0.923 | 0.923 | 0.980 | 0.557 | 0.529 | 0.533 |
| tfidf_logreg | domain | 0.794 | 0.794 | 0.878 | 0.548 | 0.532 | 0.538 |
| roberta-base | domain | 0.836 | 0.835 | 0.913 | 0.578 | 0.523 | 0.554 |
| tfidf_logreg (trained on LIAR) | - | - | - | - | 0.633 | 0.617 | 0.662 |
