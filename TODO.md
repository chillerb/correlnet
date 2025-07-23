# TODO

- [x] positioner class
- [x] expose submodule definitions directly from correlnet package
- [x] change demo_data generation to also use a correlation matrix
- [] correlation with target variable / node sizes / node color
- [] make plots pretty
- [] check if correction is correct
- [] release on GitHub


## Planning

- want to have a package that provides a simple utility functions to create Correlation Networks
- features:
    - computes pairwise correlations for given variables, supporting different correlation methods (pearson, spearman, kendall)
    - supports significance filtering via alpha threshold to discard insignificant correlations
    - plots correlation networks
        - during plotting, offers different methods to embed the variables in the 2D space
            - t-SNE directly on variables
            - t-SNE on correlation vectors of each variable
            - random, etc.
        - supports coloring variable nodes by group or by correlation with target
- design principles of the package:
    - SOLID
        - Single Responsibility: Each class should only have one responsibility
        - Open/Closed Principle: Each class should be Open to extension, closed for Modifications
        - Liskow Substitution: Replacing object with derived class should not break program
        - Interface Segregation: Use many simple interfaces vs one complicated one
        - Dependency Inversion: High-level modules should not import from low-level modules - both should depend on abstractions (interfaces)
    - DRY: Don't Repeat Yourself

```python
# functional interface for simple usage
corelnet(df, target=y, method="spearman", pos="tsne", var_group="")

# class
cn = CorelNet(df, Correlater, Embedder, var_group_df, target)
```

- if given a variable grouping, the plot should use different marker types / colors for the variable nodes
- if given a target, the plot should color variable nodes accordingly

