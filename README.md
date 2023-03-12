# PRISM-Rules

## Background
PRISM is a rules-induction system first proposed by Chendrowska [1][2] and described in Principles of Data Mining [3]. While there are numerous other rule induction systems that are able to, at least for some datasets, construct accurate and interpretable rules, to our knowledge there was no implementation of PRISM in python available, and it can be a useful tool for data mining and for prediction, often producing a clean set of iterpretable rules. This implementation is strictly for data mining and exploratory data analysis; it does not provide the ability to predict on unseen data, though the PRISM algoithm itself supports this and future releases of this package may as well. 

The algorithm works strictly with categorical features, in both the X and y columns. This implementation will, therefore, automatically bin any numeric columns to support the algorithm. By default, three equal-count bins (representing low, medium, and high values for the feature) are used, but this is configurable. 

The algorithm works by creating a set of rules for each class in the target column. The method works on unseen data in a first-rule-to-fire manner, and so all rules are generated and presented in a sensible order. For each value in the target column, the algorithm generates one rule at a time. As each rule is discovered, the rows matching that rule are removed, and the next rule is found to best describe the remaining rows. The rules may have any number of terms. For each value in the target column, we start again with the full dataset, again removing rows as rules are discovered, and generating additional rules to explain the remaining rows for this target class value. 

This implementation enhances the algorithm as described in Principles of Data Mining by outputting statistics related to each rule, as many induced rules can be of minimal significance, or of substantially lower signficance than other rules induced. As well, it allows providing a parameter to specify the minimum support for each rule: the minimum number of rows in the training data for which it applies. This helps reduce noise, though can result in some target classes having few or no rules, potentially not covering all rows for one or more target column values. 

## Comparison to Decision Tree
Decision trees are among the most common interpretable models, quite possibly the most common. When sufficiently small, they can be reasonably interpretable, perhaps as interpretable as any model type, and they can be reasonably accurate for many problems. They do have limitations as interpretable models, which PRISM was designed to address. Decision trees were not specifically designed to be interpretable; it is a convenient property of them that they are as interpretable as they are. They do, however, often grow much larger than is easily comprehensible, for example with repeated sub-trees, each often a distinct variation of the others. The decision paths for indivdiual predictions may include nodes that are irrelevan, or even misleading, to the final predictions. 

The Cendrowska paper provides examples of simple sets of rules that cannot be represented easily by trees. For example, the rules:

- Rule 1: IF a = 1 AND b = 1 THEN class = 1
- Rule 2: IF c = 1 AND d = 1 THEN class = 1

Provides a surpisingly complex tree. In fact, this is a common pattern that results in overly-complex decision trees: "where there are two (underlying) rules with no attribute in common, a situation that is likely to occur frequently in practice"[3]

## Examples

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## Parameters

```
iknn = ikNNClassifier(n_neighbors=15, 
                      method='simple majority', 
                      weight_by_score=True, 
                      max_spaces_tested=500, 
                      num_best_spaces=6)
```
#### Parameters

**n_neighbors**: int

The number of neighbors each rows is compared to. 

**method**: str
            
Must be 'simple majority' or 'use proba'. How the predictions are determined is based on this and 
weight_by_score. If the method is 'simple majority', each kNN used in the predictions simply predicts the 
majority within n_neighbors. If the method is 'use proba', each kNNs prediction is a distribution of
            predictions, weighted by their purity in the range of n_neighbors.  

**weight_by_score**: bool
If True, the kNNs used for each prediction will be weighted by their accuracy on the training set. 

## Installation
The project consists of a single python file, prism_rules.py which may be downloaded and included in any project. 

## Performance
The algorithm is generally able to produce a set of rules in seconds or minutes, but if it is necessary to decrease the execution time of the algorithm, it generally works well on samples of the data; this is safe, as the model is looking for general patterns as opposed to exceptions, and these will be present in any sufficiently large sample. As well the min_coverage, min_prob, and nbins parameters may be set to reduce execution time. Setting min_coverage and min_probs to higher values will encourage early stopping. Setting nbins to lower values (the minimum is two), will result in less categorical values being produced per numeric column, which reduces the number of potential rules to be explored at each step. 

## References
[1] Chendrowska, J. (1987) PRISM: An Algorithm for Inducing Modular Rules. International Journal of Man-Machine Studies, vol 27, pp. 349-370.

[2] Chendrowska, J. (1990) Knowledge Acquisition for Expert Systems: Inducing Modular Rules from Examples. PhD Thesis, The Open University. 

[3] Bramer, M. (2007) Principles of Data Mining, Springer press. 
