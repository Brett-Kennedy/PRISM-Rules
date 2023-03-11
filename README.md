# PRISM-Rules

## Background
PRISM is a rules-induction system first proposed by Chendrowska [1][2]. It is described in Principles of Data Mining [3]. While there are numerous other rule induction systems that are able to, at least for some datasets, construct accurate and interpretable rules, to my knowledge there was no implementation of PRISM in python, and it can be a useful tool for data mining and for predition, often producing a clean set of iterpretable rules.

The algorithm works only with categorical features, in both the X and y columns. This implementation will automatically bin any numeric columns to support the algorithm. 

The algorithm works by creating a set of rules for each class in the target column. The method works on unseen data in a first-rule-to-fire manner, and so all rules are generated and ordered in a sensible order. For each value in the target column, the algorithm generates one rule at a time. As each rule is discovered, the rows matching that rule are removed, and the next rule is found to best describe the remaining rows. The rules may have any number of terms. For each value in the target column, we start again with the full dataset. 

This implementation enhances the algorithm as described in Principles of Data Mining by outputting statistics related to each rule, as many induced rules can be of minimal significance, or at least much lower signficance than other rules induced. As well, it allows providing a parameter to specify the minimum support for each rule: the minimum number of rows in the training data for whihc it applies. This helps reduce noise, though can result in some target classes having few or no rules, where there are no rules which can be induced using the PRISM algorithm. 

## Examples


## Usage
The project consists of a single python file, prism_rules.py which may be downloaded and included in any project. 

## References
[1] Chendrowska, J. (1987) PRISM: An Algorithm for Inducing Modular Rules. International Journal of Man-Machine Studies, vol 27, pp. 349-370.

[2] Chendrowska, J. (1990) Knowledge Acquisition for Expert Systems: Inducing Modular Rules from Examples. PhD Thesis, The Open University. 

[3] Bramer, M. (2007) Principles of Data Mining, Springer press. 
