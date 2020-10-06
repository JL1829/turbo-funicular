---
toc: true
layout: post
description: F1 Score vs ROC AUC vs Accuracy vs PR AUC
categories: [Machine Learning]
title: F1 Score vs ROC AUC vs Accuracy vs PR AUC, Which Evaluation Metric Should you Choose?
comments: true
---

# F1 Score vs ROC AUC vs Accuracy vs PR AUC

[pic]

PR AUC and F1 Score are very robust evaluation metrics that work great for many classification problems but from my experience more commonly used metrics are Accuracy and ROC AUC. Are they better? Not really. As with the famous **"AUC vs Accuracy"** discussion: _There are real benefits to using both. the big question is **WHEN**_. 

There are many questions that you may have right now: 

- When `accuracy` is better evaluation than `ROC AUC`?
- What is the `F1 Score` good for?
- What is `PR Curve` and how to actually use it?
- If my problem is highly imbalanced should I use `ROC AUC` or `PR AUC`?

As always it depends, but understanding the trade-offs between different metrics is crucial when it comes to making the correct decision. 

In this blog we will discuss: 
- Talk about some of the most **common** binary classification **metrics** like `F1 Score`, `ROC AUC`, `PR AUC` and `Accuracy`. 
- **Compare them** using an example binary classification problem. 
- Tells **what we should consider** when deciding to **choose one metric over the others** (F1 Score vs ROC AUC)

# Accuracy
It measures how many observations, both positive and negative, were correctly classfied. 
$$Accuracy = \frac{tp + tn}{tp+fp+tn+fn}$$

Here: 
$$tp = True Positive$$
$$tn = True Negative$$
$$fp = False Positive$$
$$fn = False Negative$$

**Example**

Let's say we build a COVID-19 classifier, when it classify a _healthy_ person as **healthy**, then we say this is a `True Negative` result, if this classifier classify this _healthy_ person as COVID-19 patient, we called this result is `False Positive`, similarly, it classify an COVID-19 patient as a COVID-19 patient, we called this result `True Positive`, if it fail to do so, we call this result as `False Negative` 

We **shouldn't use accuracy on imbalanced problems**, it's very easy to get a very high accuracy score by simply classifying all observation as the majority class. 
Why we say so? back to the same COVID-19 problem, if we have a dataset that $99\%$ of the data is COVID-19 patient, and $1\%$ of the data is healthy person, if the classifier we built, just simply "guess" every sample is COVID-19 patient, it get $99\%$ accuracy, but can we say this classifier a **Good** Classifier? 

In Python we can calculate the `accuracy` in the following way: 

```python
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = (tp + tn) / (tp + fp + fn + tn)

# or simply call the accuracy_score api
accuracy_score(y_true, y_pred_class)
```

Since `Accuracy` score is calculated on the predicted classes(not prediction socre) we **need to apply certain threshold** before computing it. The obvious choice is the threshold of $0.5$ but it can be adjust according to the actual problem. 

An example of **How accuracy depends on the threshold** choice: 

[pic]

