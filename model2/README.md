# John's Model - Source Vector Machine

## Why SVM's

1. SVM's are good at doing supervised binary classification (our problem).  
They find a hyperplane which separates the two classes.
2. SVM's work well on high dimensional data (like this dataset), even where the number of features exceeds the number of samples. 

I partitioned the training csv into x and y training and validation for the attempts:

## First Attempt

```text
[[533   0]
 [ 55   0]]
```

When using the default parameters for the SVM, it would only product an output indicating not anomalous.  
Perhaps this is due to an inbalanced dataset? Many more non anomalous vs anomalous.

## Second Attempt: Balanced Dataset

```text
[[271 270]
 [ 27  20]]
```

I under (or over) sampled the data to make anomalous and non-anomalous have the same prevalence.  
The model acted like a sklearn dummy classifier, and gave an output in equal portion to both classes.

## Third Attempt (BEST): Tweaking Model Params

```text
[[519  26]
 [ 41   2]]
accuracy: 0.886054
precision:
[0.92678571 0.07142857]
recall:
[0.95229358 0.04651163]
f1:
[0.93936652 0.05633803]
```

Changed the kernel param to 'linear'. Everything else is default.

## Fourth Attempt: Modifying the model

Tried various param tweaks, nothing else led to improvements.

# Conclusion

SVM's work ok (~90% accuracy) on this dataset.