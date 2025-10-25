from sklearn.feature_selection import SelectKBest, f_classif

def select_top_features(X, y, k=5):
    """Select top k features based on ANOVA F-score."""
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    print(f"✅ Selected top {k} features.")
    return X_new, selector
