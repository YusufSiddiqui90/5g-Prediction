import json, joblib, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.inspection import permutation_importance

df = pd.read_csv("india_5g_qos_clean.csv")
DROP = ["QoE_Score", "QoS_Class", "SLA_Violation", "Alloc_Mbps"]
X = df.drop(columns=DROP)
cats = list(X.select_dtypes(exclude="number").columns)
levels = {c: sorted(X[c].unique().tolist()) for c in cats}
for c in cats:
    X[c] = pd.Categorical(X[c], categories=levels[c])

Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
    X, df["QoE_Score"], df["QoS_Class"], test_size=0.2, random_state=0, stratify=df["QoS_Class"])

opt = dict(categorical_features="from_dtype", random_state=0)
reg = HistGradientBoostingRegressor(**opt).fit(Xtr, ytr)
clf = HistGradientBoostingClassifier(**opt).fit(Xtr, ctr)

r2 = r2_score(yte, reg.predict(Xte))
acc = accuracy_score(cte, clf.predict(Xte))
f1 = f1_score(cte, clf.predict(Xte), average="macro")
print(f"R2={r2:.3f} acc={acc:.3f} macroF1={f1:.3f}")

pi = permutation_importance(reg, Xte.sample(5000, random_state=0), yte.loc[Xte.sample(5000, random_state=0).index],
                            n_repeats=3, random_state=0)
imp = dict(sorted(zip(X.columns, pi.importances_mean.tolist()), key=lambda t: -t[1]))

ranges = {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in X.select_dtypes("number").columns}
joblib.dump({"reg": reg, "clf": clf, "columns": list(X.columns)}, "model.pkl", compress=3)
json.dump({"levels": levels, "ranges": ranges, "importance": imp,
           "metrics": {"r2": r2, "acc": acc, "f1": f1, "rows": len(df)}}, open("meta.json", "w"), indent=1)
print(imp)
