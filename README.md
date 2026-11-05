\# Machine Learning-Based Intrusion Detection System Robustness Evaluation



This repository contains the source code for a thesis study evaluating the

robustness of machine learning-based intrusion detection systems (IDS) against

adversarial attacks.



The pipeline evaluates clean performance and adversarial robustness on:



\- NSL-KDD

\- CICIDS2017



It compares:



\- Logistic Regression

\- Random Forest

\- Baseline DNN

\- PGD adversarially trained DNN

\- Defensively distilled DNN



against FGSM, JSMA, and Carlini-Wagner L2 (CW-L2) attacks.



\## Pipeline



Run the scripts in this order:



```text

01\_data\_inventory.py

02\_data\_audit.py

03\_build\_source\_inventory.py

04\_build\_split\_manifests.py

05\_deduplicate\_and\_validate.py

06\_preprocess\_and\_audit.py

07\_train\_baselines.py

08\_adversarial\_training.py

09\_defensive\_distillation.py

10\_fgsm\_evaluation.py

11\_jsma\_evaluation.py

12\_cw\_evaluation.py

13\_consolidate\_results.py

```



Scripts 06–13 form the final reproducible experimental workflow.



\## Leakage prevention



Script 06 excludes annotation fields from the predictor set:



\- `difficulty` in NSL-KDD

\- `Attempted Category` in CICIDS2017



It also fits imputation, categorical encoding, and scaling on the training

partition only. Validation and test data are transformed using the fitted

training objects.



The script audits model-input duplicates and reports cross-partition

collisions.



\## Models and defences



Script 07 trains clean baseline Logistic Regression, Random Forest, and DNN

models.



Script 08 performs PGD-based adversarial training using the clean baseline DNN

as the initial model.



Script 09 performs defensive distillation using the clean baseline DNN as the

teacher.



\## Adversarial evaluation



| Script | Attack | Evaluation population |

|---|---|---|

| 10 | FGSM | Full test set |

| 11 | JSMA | Fixed random sample of 1,000 test records per dataset |

| 12 | CW-L2 | Fixed random sample of 2,000 test records per dataset |



All attacks operate in the standardized model-input feature space.



\## Final results



Script 13 combines the final clean, FGSM, JSMA, and CW-L2 results into:



\- `combined\_thesis\_results.csv`

\- Chapter 4 result tables

\- Figures 4.1–4.6

\- `Thesis\_Final\_Results.xlsx`

\- `provenance.json`

\- `artifact\_hashes.json`



The final results package is stored under:



```text

reports/final\_results/

```



\## Notes



Datasets, pretrained model checkpoints, NumPy arrays, and large generated

report folders are not included in this repository.

