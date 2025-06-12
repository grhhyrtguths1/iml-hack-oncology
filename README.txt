evaluate_part_0.py -
This script evaluates a multi-label classification model by
computing the micro-F1 and macro-F1 scores between the ground truth
(--gold) and predicted (--pred) labels, both in CSV files where each row
is a list of metastasis sites. It first encodes variable-length label
lists into fixed-size multi-hot vectors using a custom Encode_Multi_Hot class,
then compares the two encoded label sets. Finally, it logs the F1 metrics to
assess overall (micro) and per-label (macro) performance.

main.py - This script loads and preprocesses the training and test data,
then trains and evaluates multi-label classification models for metastasis
site prediction. It includes support for a baseline logistic regression model,
an XGBoost-based model, and a neural network model.
We used logistic regression as a baseline, then applied XGBoost to improve performance.
Unfortunately, we didnt have time to integrate the neural network model.
Due to accidentally training on the full train dataset, we avoided further iterations to prevent overfitting.
If we had more time bootstrapping would have been done.

part1.py - This script defines a modular multi-label classification pipeline for predicting
metastasis sites from patient data. It includes three model classes—baseline logistic regression,
XGBoost, and a PyTorch-based neural network—all using a shared multi-hot encoding scheme for
handling variable-length label sets. Each model supports training, prediction,
evaluation (via micro and macro F1 scores), and saving decoded predictions back to
CSV for submission.

preprocessing.py -
