# Practical Application III: Comparing Classifiers

**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  

### Getting Started

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.


### Understanding the Task

The business objective of this task is to increase the efficiency of directed marketing campaigns for long-term deposit subscriptions at a Portuguese banking institution by developing a predictive model that can accurately identify potential customers who are most likely to subscribe to a term deposit.

This involves analyzing historical data from previous telephone marketing campaigns to understand the key factors that contribute to a successful subscription. The model aims to assist marketing managers in making informed decisions about whom to contact, thereby optimizing the allocation of resources such as human effort, time, and budget. The ultimate goal is to maintain or improve the success rate of these campaigns while reducing the number of contacts made, leading to cost-effective and targeted marketing strategies.

Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models.  Using the default settings for each of the models, fit and score each.  Also, be sure to compare the fit time of each of the models.

### Training Time and Accuracy analysis:

Balance Between Training Time and Accuracy: Logistic Regression and SVM, despite their longer training times (especially SVM), provide the best balance between training and test accuracy. This indicates that they generalize well without overfitting.
Overfitting in Decision Tree: The high training accuracy but lower test accuracy for the Decision Tree model suggests it might be overfitting the training data.
Efficiency of KNN: KNN's short training time is notable, but its slightly lower test accuracy indicates that it might benefit from parameter tuning (like adjusting the number of neighbors).


### Model improvement

# Create a new features for age and job interaction
# List of job category columns

```
job_columns = ['job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 
               'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 
               'job_unemployed', 'job_unknown']


	Model	Train Time	Train Accuracy	Test Accuracy
0	Logistic Regression	0.638171	0.887982	0.886137
1	KNN	0.007312	0.890531	0.873513
2	Decision Tree	0.108112	0.916601	0.862709
3	SVM	28.648827	0.888376	0.886623

```

### Analysis:

```
The Logistic Regression model's performance remained stable with the new features, but the training time slightly increased. This suggests that the new features didn't significantly impact the model's ability to generalize.

The KNN model shows a minor improvement in both training and test accuracy, indicating that the new features might be slightly beneficial for this model.

The Decision Tree model's test accuracy shows a slight improvement. However, it still appears to be overfitting (high training accuracy compared to test accuracy).

The SVM model's training time increased notably with no improvement in accuracy. This might be due to the increased complexity of the data with new features. SVMs are computationally intensive, and more features can lead to longer training times.

```

### Let's work KNN hyperparameters: Best parameters for KNN: {'metric': 'euclidean', 'n_neighbors': 15, 'weights': 'uniform'}

```
Improved Performance: The increase in the number of neighbors (n_neighbors: 15) compared to the default value (usually 5) suggests that the model benefits from considering more neighbors to make a prediction. This can help in smoothing the decision boundaries and potentially reducing overfitting.

Distance Metric: The fact that 'Euclidean' came out as the best metric suggests that the standard Euclidean distance is effective for your data in measuring similarity between instances.

Uniform Weights: The choice of 'uniform' weights indicates that all neighbors contribute equally to the classification of a new point. This approach works well when the density of the data is relatively uniform. If the data has varying densities, sometimes 'distance' weighting might perform better.

```
### Best parameters for Decision Tree: {'max_depth': 10, 'max_features': 'sqrt', 'min_impurity_decrease': 0.1, 'min_samples_leaf': 1, 'min_samples_split': 2}

```
KNN: Improved slightly with hyperparameter tuning. The test accuracy increased modestly, indicating that the adjustments in the number of neighbors and distance metric were beneficial.

Logistic Regression and SVM: Both now show strong performance with a test accuracy of 0.89. This indicates that these models not only provide a good balance between training time and accuracy but are also highly effective with the current feature set. Their performance is notable given that they are achieving the same level of accuracy as the more complex models.

Decision Tree: Shows a significant improvement post-tuning, now achieving comparable performance to Logistic Regression and SVM. This improvement suggests that the hyperparameter tuning effectively addressed overfitting issues, making the Decision Tree a strong contender.

Random Forest: Although it shows slightly lower performance compared to the tuned Decision Tree, Logistic Regression, and SVM, with a test accuracy of 0.87, it remains a robust model. Random Forests are known for their generalization capabilities, which can be advantageous in different or more complex datasets.

The choice between Logistic Regression, SVM, and Decision Tree can now be based more on factors like model interpretability, training time, and how they align with your specific application requirements. All three models are performing similarly in terms of accuracy.

Despite the simplicity of Logistic Regression, it performs on par with more complex models like SVM and Decision Tree. This underscores the effectiveness of feature engineering and the right choice of model for the specific characteristics of this dataset.
```


### The confusion matrix for the Decision Tree model tells the following:
```

True Negative (TN): 7303 observations were correctly predicted as class '0' 
False Positive (FP): 0 observations were incorrectly predicted as class '1' 
True Positive (TP): 0 observations were correctly predicted as class '1'.
False Negative (FN): 935 observations were incorrectly predicted as class '0' when they are actually class '1'.
Here's what we can infer from this confusion matrix:

High Specificity: The model has predicted the negative class (class '0') with high accuracy. It didn't falsely label any negative instances as positive, which is evident from the zero false positives.

Poor Sensitivity: The model failed to correctly predict any of the positive class (class '1'). This is a clear indication of a model that is biased towards the negative class, resulting in a very low sensitivity (recall) for the positive class.

Imbalanced Prediction: The model seems to be biased towards predicting the majority class (presumably class '0'). This could be a result of an imbalanced dataset where the negative class significantly outnumbers the positive class.

Lack of Positive Predictions: The fact that there are zero true positives and zero false positives suggests that the model might have a threshold issue or it is too conservative in predicting class '1'. The model might require threshold tuning or rebalancing of the dataset.

Potential Overfitting: While the accuracy might seem high because the majority class is being predicted almost perfectly, the model is not useful for its inability to predict the minority class. This might be a case of overfitting to the majority class.

Business Impact: If the goal is to identify potential subscribers (class '1'), the model is not serving its purpose. All potential subscribers are being missed (935 false negatives), which could represent a significant loss of opportunity.
```

### The confusion matrix for Logistic Regression shows the following:
```

True Negative (TN): 7285 observations where the model correctly predicted the negative class.
False Positive (FP): 18 observations where the model incorrectly predicted the positive class.
True Positive (TP): 15 observations where the model correctly predicted the positive class.
False Negative (FN): 920 observations where the model incorrectly predicted the negative class.
Here are some key points to interpret from this matrix:

General Performance: The model is predicting the majority of the negative class (class '0') correctly but is struggling with the positive class (class '1'), similar to the Decision Tree model.

Bias Towards Negative Class: There's a strong bias towards predicting the negative class, as indicated by the high number of true negatives compared to true positives.

Few Positive Predictions: The model made very few positive predictions (only 15 true positives), which suggests that it may have a conservative threshold or it may not be picking up the patterns in the data that lead to positive outcomes.

False Negatives: There are a significant number of false negatives, meaning that many of the positive instances were missed by the model. This could be critical if the cost of missing a positive case is high (e.g., a missed opportunity for a term deposit subscription).

Model Sensitivity: The sensitivity (or recall) for the positive class is low, indicating the model's limited ability to detect the positive class instances.

Precision for Positive Class: Although the model has predicted few positives, the precision (the proportion of positive identifications that were actually correct) is low because the false positives are close to the true positives in number.
```

### The confusion matrix for the K-Nearest Neighbors model shows the following:
```

True Negative (TN): 7251 observations where the model correctly predicted the negative class.
False Positive (FP): 52 observations where the model incorrectly predicted the positive class.
True Positive (TP): 37 observations where the model correctly predicted the positive class.
False Negative (FN): 898 observations where the model incorrectly predicted the negative class.
Here's what these results indicate:

Bias Towards Negative Class: Similar to the Decision Tree and Logistic Regression models, the KNN model is also biased towards predicting the majority class, indicated by the high number of true negatives and the high number of false negatives.

Low Sensitivity: The model has a low sensitivity (or recall) for the positive class, as it only correctly identified 37 out of the 935 actual positives (TP + FN).

Moderate Precision: While the model has a higher number of true positives than the previous models, the precision (TP / (TP + FP)) is still moderate due to the 52 false positives.

False Negatives: A significant number of positive instances were missed by the model (898 false negatives), which could be critical depending on the context. For instance, if this is a marketing campaign, these missed opportunities could represent potential revenue loss.

Model Evaluation: This model's performance suggests it might benefit from:

Addressing class imbalance (if present).
Revisiting the chosen value for n_neighbors and considering weight adjustments.
Potentially including more informative features or feature engineering to better capture distinctions between classes.
Comparison to Other Models: When comparing this KNN model to the previously discussed Logistic Regression and Decision Tree models, it is underperforming in terms of correctly identifying the positive class. All models are struggling with the positive class, but the KNN model is not as extreme in its predictions towards the negative class.
```


### The confusion matrix for the SVM model shows the following:
```

True Negative (TN): 7295 observations were correctly predicted as class '0' (e.g., the client did not subscribe to a term deposit).
False Positive (FP): 8 observations were incorrectly predicted as class '1' (e.g., the client subscribed to a term deposit) when they are actually class '0'.
True Positive (TP): 9 observations were correctly predicted as class '1'.
False Negative (FN): 926 observations were incorrectly predicted as class '0' when they are actually class '1'.
Here's what we can deduce from this matrix:

General Performance: Similar to the previous models, the SVM model is highly accurate in predicting the negative class but performs poorly on the positive class.

Bias Towards Negative Class: There's a considerable bias towards predicting the majority class, which is likely class '0', as indicated by the high number of true negatives and false negatives.

Challenges with Positive Class: The model managed to predict a very small number of positive instances correctly (9 true positives), but it also missed many (926 false negatives). This suggests that the SVM model has difficulty identifying the positive class, which is a critical issue if the goal is to predict class '1' accurately.

Low Sensitivity: The sensitivity (or recall) for the positive class is very low, indicating the model's limited ability to detect the positive class instances.

Moderate Precision for Positive Class: The precision (the proportion of positive identifications that were actually correct) is higher for SVM than for the Decision Tree and KNN models, but the low number of true positives still indicates a performance issue.
```

### The confusion matrix for the RandomForest classifier shows the following details:
```

True Negative (TN): 7061 observations where the model correctly predicted the negative class ('0').
False Positive (FP): 242 observations where the model incorrectly predicted the positive class ('1').
True Positive (TP): 84 observations where the model correctly predicted the positive class ('1').
False Negative (FN): 851 observations where the model incorrectly predicted the negative class ('0').

General Performance: The RandomForest model has a high number of true negatives, which indicates good performance on the negative class.

Positive Class Predictions: Unlike the Decision Tree, Logistic Regression, and SVM models, the RandomForest has managed to correctly predict more positive instances (84 true positives). This suggests an improved ability to detect the positive class, which is a step forward.

False Positives and Negatives: There are still a considerable number of false negatives, indicating missed opportunities to identify the positive class. The false positives have increased compared to the other models, which might suggest a slight trade-off between sensitivity and specificity.

Sensitivity and Specificity: The model has achieved better sensitivity (ability to detect true positives) than the previous models but at the cost of more false positives, which reduces its specificity (ability to detect true negatives).

Model Balance: The RandomForest model seems to be more balanced in terms of recognizing both classes compared to the other models, although there's still room for improvement, especially in reducing false negatives.
```


### Conclusions

In this practical application assignment, we aimed to compare the performance of four different classifier models: k-Nearest Neighbors (kNN), Decision Trees, Logistic Regression, and Support Vector Machines (SVM), on a dataset related to the marketing of bank products over the telephone. Our objectives were to understand the dataset, prepare it for modeling, apply these classifiers, and interpret their performance with a focus on their application in a business context.

In conclusion, the analysis revealed that all models performed well on the negative class but had varying degrees of success with the positive class. Adjustments such as hyperparameter tuning and threshold adjustments were made to improve model performance. The RandomForest model, after tuning and threshold adjustments, performed robustly with a balance between precision and recall, as evidenced by the Precision-Recall Curve and an accuracy of 0.89.

Recommendations for next steps include:

Addressing the class imbalance more rigorously, possibly through advanced resampling techniques or by exploring alternative models better suited for imbalanced data.

Further exploration of feature importance to improve model performance, particularly for the positive class.

Continuous evaluation using various metrics and potentially integrating cost-sensitive learning to align the model's predictions with the business impact of different types of classification errors.




















