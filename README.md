# PRISM-Rules

## Background
PRISM is a rules-induction system first proposed by Chendrowska [[1]](#references) [[2]](#references) and described in Principles of Data Mining [[3]](#references). While there are numerous other rule induction systems that are able to, at least for some datasets, construct accurate and interpretable rules, to our knowledge there was no implementation of PRISM in python available, and it can be a useful tool for data mining and for prediction, often producing a clean set of iterpretable rules. This implementation is strictly for data mining and exploratory data analysis; it does not provide the ability to predict on unseen data, though the PRISM algoithm itself supports this and future releases of this package may as well. 

The algorithm works strictly with categorical features, in both the X and y columns. This implementation will, therefore, automatically bin any numeric columns to support the algorithm. By default, three equal-count bins (representing low, medium, and high values for the feature) are used, but this is configurable. 

The algorithm works by creating a set of rules for each class in the target column. The method works on unseen data in a first-rule-to-fire manner, and so all rules are generated and presented in a sensible order. For each value in the target column, the algorithm generates one rule at a time. As each rule is discovered, the rows matching that rule are removed, and the next rule is found to best describe the remaining rows. The rules may have any number of terms. For each value in the target column, we start again with the full dataset, again removing rows as rules are discovered, and generating additional rules to explain the remaining rows for this target class value. 

This implementation enhances the algorithm as described in Principles of Data Mining by outputting statistics related to each rule, as many induced rules can be of minimal significance, or of substantially lower signficance than other rules induced. As well, it allows providing a parameter to specify the minimum support for each rule: the minimum number of rows in the training data for which it applies. This helps reduce noise, though can result in some target classes having few or no rules, potentially not covering all rows for one or more target column values. 

## Comparison to Decision Tree
Decision trees are among the most common interpretable models, quite possibly the most common. When sufficiently small, they can be reasonably interpretable, perhaps as interpretable as any model type, and they can be reasonably accurate for many problems. They do have limitations as interpretable models, which PRISM was designed to address. Decision trees were not specifically designed to be interpretable; it is a convenient property of them that they are as interpretable as they are. They do, however, often grow much larger than is easily comprehensible, for example with repeated sub-trees, each often a distinct variation of the others. The decision paths for indivdiual predictions may include nodes that are irrelevan, or even misleading, to the final predictions. 

The Cendrowska paper provides examples of simple sets of rules that cannot be represented easily by trees. For example, the rules:

- Rule 1: IF a = 1 AND b = 1 THEN class = 1
- Rule 2: IF c = 1 AND d = 1 THEN class = 1

Provides a surpisingly complex tree. In fact, this is a common pattern that results in overly-complex decision trees: "where there are two (underlying) rules with no attribute in common, a situation that is likely to occur frequently in practice"[[3]](#references)

## Example using the Wine dataset from sklearn

To use the tool, simply create a PrismRules object, and call get_prism_rules():

```python
data = datasets.load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Y'] = data['target']
display(df.head())

prism = PrismRules()
_ = prism.get_prism_rules(df, 'Y')
```

### Results
```
Target: 0
proline = High AND alcohol = High
   Support:  the target has value: '0' for 100.000% of the 39 rows matching the rule 
   Coverage: the rule matches: 39 out of 59 rows for target value: 0. This is:
      66.102% of total rows for target value: 0
      21.910% of total rows in data
proline = High AND alcalinity_of_ash = Low
   Support:  The target has value: '0' for 100.000% of the 10 remaining rows matching the rule 
   Coverage: The rule matches: 10 out of 20 rows remaining for target value: '0'. This is:
      50.000% of remaining rows for target value: '0'
      16.949% of total rows for target value: 0
      5.618% of total rows in data

Target: 1
color_intensity = Low AND alcohol = Low
   Support:  the target has value: '1' for 100.000% of the 46 rows matching the rule 
   Coverage: the rule matches: 46 out of 71 rows for target value: 1. This is:
      64.789% of total rows for target value: 1
      25.843% of total rows in data
color_intensity = Low
   Support:  The target has value: '1' for 78.571% of the 11 remaining rows matching the rule 
   Coverage: The rule matches: 11 out of 25 rows remaining for target value: '1'. This is:
      44.000% of remaining rows for target value: '1'
      15.493% of total rows for target value: 1
      6.180% of total rows in data

Target: 2
flavanoids = Low AND color_intensity = Med
   Support:  the target has value: '2' for 100.000% of the 16 rows matching the rule 
   Coverage: the rule matches: 16 out of 48 rows for target value: 2. This is:
      33.333% of total rows for target value: 2
      8.989% of total rows in data
flavanoids = Low AND alcohol = High
   Support:  The target has value: '2' for 100.000% of the 10 remaining rows matching the rule 
   Coverage: The rule matches: 10 out of 32 rows remaining for target value: '2'. This is:
      31.250% of remaining rows for target value: '2'
      20.833% of total rows for target value: 2
      5.618% of total rows in data
flavanoids = Low AND color_intensity = High AND hue = Low
   Support:  The target has value: '2' for 100.000% of the 21 remaining rows matching the rule 
   Coverage: The rule matches: 21 out of 22 rows remaining for target value: '2'. This is:
      95.455% of remaining rows for target value: '2'
      43.750% of total rows for target value: 2
      11.798% of total rows in data
```

## Example with Numeric Data

In this example, we use sklearn's make_classification() method to create numeric data, which is then binned. 

```python
x, y = make_classification(
    n_samples=1000, 
    n_features=20,    
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2,
    flip_y=0, 
    random_state=0
    )

df = pd.DataFrame(x)
df['Y'] = y

prism = PrismRules()
_ = prism.get_prism_rules(df, 'Y')
```

### Results
The data is binned into low, medium, and high values for each column. The results are a set of rules per target class, each with the 

```
Target: 0
1 = High
   Support:  the target has value: '0' for 100.000% of the 333 rows matching the rule 
   Coverage: the rule matches: 333 out of 500 rows for target value: 0. This is:
      66.600% of total rows for target value: 0
      33.300% of total rows in data
15 = Low AND 4 = Med
   Support:  The target has value: '0' for 100.000% of the 63 remaining rows matching the rule 
   Coverage: The rule matches: 63 out of 167 rows remaining for target value: '0'. This is:
      37.725% of remaining rows for target value: '0'
      12.600% of total rows for target value: 0
      6.300% of total rows in data
4 = High AND 1 = Med
   Support:  The target has value: '0' for 100.000% of the 47 remaining rows matching the rule 
   Coverage: The rule matches: 47 out of 104 rows remaining for target value: '0'. This is:
      45.192% of remaining rows for target value: '0'
      9.400% of total rows for target value: 0
      4.700% of total rows in data
```    

## Methods

```
prism = Prism_rules(min_coverage=10, min_prob=0.75, nbins=3, verbose=0)
```
#### Parameters

**min_coverage**: int 

The minimum number of rows each rule must cover in the training data. This may be adjusted to control the number of rules generated.

**min_prob**: float
            
The minimum support each rule must cover. This is the number of rows that match the current value in the target class for the subset of rows implied by the rule. 

**nbins**: int
The number of bins used to bin numeric columns. This may be adjusted to produce more meaningful rules. 

## verbose: int 
If 0, no output is produced other than the induced rules. If 1, progress indicators are presented as each rule is induced. 

## Performance
The algorithm is generally able to produce a set of rules in seconds or minutes, but if it is necessary to decrease the execution time of the algorithm, it generally works well on samples of the data; this is safe, as the model is looking for general patterns as opposed to exceptions, and these will be present in any sufficiently large sample. As well the min_coverage, min_prob, and nbins parameters may be set to reduce execution time. Setting min_coverage and min_probs to higher values will encourage early stopping. Setting nbins to lower values (the minimum is two), will result in less categorical values being produced per numeric column, which reduces the number of potential rules to be explored at each step. 

## References
[1] Chendrowska, J. (1987) PRISM: An Algorithm for Inducing Modular Rules. International Journal of Man-Machine Studies, vol 27, pp. 349-370.

[2] Chendrowska, J. (1990) Knowledge Acquisition for Expert Systems: Inducing Modular Rules from Examples. PhD Thesis, The Open University. 

[3] Bramer, M. (2007) Principles of Data Mining, Springer press. 
