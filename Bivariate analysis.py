"""
Bivariate Statistical Analysis — Diabetes Dataset
==================================================
Implements all tests from COM725 Week 3 Lecture (and more):

FROM THE LECTURE:
  1.  Chi-Square Test            (Categorical vs Categorical)
  2.  Fisher's Exact Test        (Categorical vs Categorical, small cell counts)
  3.  Independent t-test         (Numerical vs Categorical, 2 groups)
  4.  Paired t-test              (same variable, two time-points / conditions)
  5.  One-Way ANOVA              (Numerical vs Categorical, 3+ groups)
  6.  Tukey HSD Post-Hoc        (after ANOVA)
  7.  Mann-Whitney U             (non-parametric, 2 groups)
  8.  Kruskal-Wallis             (non-parametric, 3+ groups)
  9.  Pearson Correlation        (Numerical vs Numerical)
  10. Spearman Correlation       (Numerical vs Numerical, monotonic)
  11. Simple Linear Regression   (Numerical vs Numerical)

EXTRAS (beyond the lecture):
  12. Point-Biserial Correlation (Binary vs Continuous)
  13. Logistic Regression        (Binary outcome)
  14. Cramér's V                 (effect size for Chi-Square)
  15. Cohen's d                  (effect size for t-test)
  16. Dunn's Test                (post-hoc for Kruskal-Wallis)
  17. Normality Tests (Shapiro-Wilk, D'Agostino K²)
  18. Levene's Test of Equal Variances
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (
    chi2_contingency, fisher_exact, ttest_ind, ttest_rel,
    f_oneway, mannwhitneyu, kruskal, pearsonr, spearmanr,
    shapiro, normaltest, levene
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


df = pd.read_csv("cleaned_data.csv")

# Derived columns
df["smoker"] = (df["smokingyears"] > 0).astype(int)
df["on_insulin"] = df["medication_insulin"]
df["gender_mf"] = df["gender"].map({"M": "Male", "F": "Female"})
df["has_microvascular"] = (df["diabetic_microvascular_complications"] != "0").astype(int)
df["has_macrovascular"] = (df["diabetic_macrovascular_complications"] != "0").astype(int)

# Blood pressure category
def bp_cat(sbp):
    if sbp < 120: return "Normal"
    elif sbp < 130: return "Elevated"
    else: return "High"

df["bp_category"] = df["sysbp"].apply(bp_cat)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: print section header
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def result(label, value):
    print(f"  {label:<45} {value}")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 1: CHI-SQUARE TEST
# Categorical vs Categorical: Gender vs Insulin use
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 1 (Lecture): CHI-SQUARE TEST — Gender vs Insulin Use")

df_mf = df[df["gender"].isin(["M", "F"])].copy()
ct = pd.crosstab(df_mf["gender"], df_mf["medication_insulin"])
chi2, p, dof, expected = chi2_contingency(ct)

print(f"\n  Contingency Table:\n{ct.to_string()}\n")
result("Chi² statistic:", f"{chi2:.4f}")
result("p-value:", f"{p:.4f}")
result("Degrees of freedom:", dof)
result("Interpretation:",
       "SIGNIFICANT (p < 0.05) — Gender and insulin use are associated."
       if p < 0.05 else
       "NOT significant (p ≥ 0.05) — No association between gender and insulin use.")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 2: FISHER'S EXACT TEST
# Categorical vs Categorical (2×2): Smoker vs Amputation
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 2 (Lecture): FISHER'S EXACT TEST — Smoker vs Amputation")

ct2 = pd.crosstab(df["smoker"], df["amputation"])
print(f"\n  Contingency Table:\n{ct2.to_string()}\n")

# Fisher's exact works only on 2x2 tables
if ct2.shape == (2, 2):
    oddsratio, p_fisher = fisher_exact(ct2)
    result("Odds Ratio:", f"{oddsratio:.4f}")
    result("p-value:", f"{p_fisher:.4f}")
    result("Interpretation:",
           "SIGNIFICANT — Smoker status and amputation are associated."
           if p_fisher < 0.05 else
           "NOT significant — No significant association.")
else:
    print("  [Note] Table is not 2×2; using Chi-Square instead.")
    chi2f, pf, _, _ = chi2_contingency(ct2)
    result("Chi² statistic:", f"{chi2f:.4f}")
    result("p-value:", f"{pf:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 3: INDEPENDENT T-TEST
# Numerical vs Categorical (2 groups): BMI by Gender
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 3 (Lecture): INDEPENDENT T-TEST — BMI by Gender (M vs F)")

male_bmi   = df[df["gender"] == "M"]["bmi"].dropna()
female_bmi = df[df["gender"] == "F"]["bmi"].dropna()

t_stat, p_t = ttest_ind(male_bmi, female_bmi, equal_var=False)  # Welch's
result("Male   BMI mean ± SD:", f"{male_bmi.mean():.2f} ± {male_bmi.std():.2f}")
result("Female BMI mean ± SD:", f"{female_bmi.mean():.2f} ± {female_bmi.std():.2f}")
result("t-statistic (Welch's):", f"{t_stat:.4f}")
result("p-value:", f"{p_t:.4f}")
result("Interpretation:",
       "SIGNIFICANT — BMI differs significantly between males and females."
       if p_t < 0.05 else
       "NOT significant — No significant BMI difference between genders.")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 4: PAIRED T-TEST
# Systolic BP vs Diastolic BP (paired within same patient)
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 4 (Lecture): PAIRED T-TEST — Systolic BP vs Diastolic BP")

sysbp = df["sysbp"].dropna()
diabp = df["diabp"].dropna()
min_len = min(len(sysbp), len(diabp))
t_paired, p_paired = ttest_rel(sysbp[:min_len], diabp[:min_len])

result("Mean Systolic BP:", f"{sysbp.mean():.2f}")
result("Mean Diastolic BP:", f"{diabp.mean():.2f}")
result("t-statistic:", f"{t_paired:.4f}")
result("p-value:", f"{p_paired:.6f}")
result("Interpretation:",
       "SIGNIFICANT — Systolic and diastolic BP differ significantly."
       if p_paired < 0.05 else
       "NOT significant — No significant difference.")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 5 + 6: ONE-WAY ANOVA + TUKEY HSD
# Numerical vs Categorical (3 groups): BMI across BP categories
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 5 (Lecture): ONE-WAY ANOVA — BMI across BP Categories")

groups = [g["bmi"].dropna().values for _, g in df.groupby("bp_category")]
f_stat, p_anova = f_oneway(*groups)

result("F-statistic:", f"{f_stat:.4f}")
result("p-value:", f"{p_anova:.4f}")
result("Interpretation:",
       "SIGNIFICANT — At least one BP group has different mean BMI."
       if p_anova < 0.05 else
       "NOT significant — BMI is similar across all BP categories.")

section("TEST 6 (Lecture): TUKEY HSD POST-HOC — BMI across BP Categories")

tukey = pairwise_tukeyhsd(df["bmi"].dropna(),
                          df.loc[df["bmi"].notna(), "bp_category"])
print(f"\n{tukey.summary()}")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 7: MANN-WHITNEY U
# Non-parametric: HbA1c by Gender
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 7 (Lecture): MANN-WHITNEY U — HbA1c by Gender (M vs F)")

male_hba1c   = df[df["gender"] == "M"]["hba1c"].dropna()
female_hba1c = df[df["gender"] == "F"]["hba1c"].dropna()

u_stat, p_mw = mannwhitneyu(male_hba1c, female_hba1c, alternative="two-sided")
result("U-statistic:", f"{u_stat:.1f}")
result("p-value:", f"{p_mw:.4f}")
result("Male   HbA1c median:", f"{male_hba1c.median():.2f}")
result("Female HbA1c median:", f"{female_hba1c.median():.2f}")
result("Interpretation:",
       "SIGNIFICANT — HbA1c medians differ between genders."
       if p_mw < 0.05 else
       "NOT significant — HbA1c medians are similar between genders.")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 8: KRUSKAL-WALLIS
# Non-parametric 3+ groups: HbA1c across BP categories
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 8 (Lecture): KRUSKAL-WALLIS — HbA1c across BP Categories")

groups_kw = [g["hba1c"].dropna().values for _, g in df.groupby("bp_category")]
h_stat, p_kw = kruskal(*groups_kw)

result("H-statistic:", f"{h_stat:.4f}")
result("p-value:", f"{p_kw:.4f}")
result("Interpretation:",
       "SIGNIFICANT — At least one BP group has different HbA1c distribution."
       if p_kw < 0.05 else
       "NOT significant — HbA1c distribution is similar across BP categories.")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 9 + 10: PEARSON & SPEARMAN CORRELATION
# Numerical vs Numerical: Age vs Systolic BP
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 9 (Lecture): PEARSON CORRELATION — Age vs Systolic BP")

r_pearson, p_pearson = pearsonr(df["age"].dropna(), df.loc[df["age"].notna(), "sysbp"])
result("Pearson r:", f"{r_pearson:.4f}")
result("p-value:", f"{p_pearson:.4f}")
result("Interpretation:",
       "SIGNIFICANT linear correlation." if p_pearson < 0.05 else
       "NOT significant — No linear correlation detected.")

section("TEST 10 (Lecture): SPEARMAN CORRELATION — Age vs Systolic BP")

r_spearman, p_spearman = spearmanr(df["age"].dropna(), df.loc[df["age"].notna(), "sysbp"])
result("Spearman ρ:", f"{r_spearman:.4f}")
result("p-value:", f"{p_spearman:.4f}")
result("Interpretation:",
       "SIGNIFICANT monotonic correlation." if p_spearman < 0.05 else
       "NOT significant — No monotonic correlation detected.")

# ─────────────────────────────────────────────────────────────────────────────
# FROM LECTURE — Test 11: SIMPLE LINEAR REGRESSION
# Age → Systolic BP
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 11 (Lecture): SIMPLE LINEAR REGRESSION — Age → Systolic BP")

X = sm.add_constant(df["age"])
y = df["sysbp"]
model = sm.OLS(y, X).fit()
print(f"\n{model.summary().tables[1]}")
result("R-squared:", f"{model.rsquared:.4f}")
result("Slope (age):", f"{model.params['age']:.4f}")
result("Intercept:", f"{model.params['const']:.4f}")
result("p-value (slope):", f"{model.pvalues['age']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 12: NORMALITY TESTS
# Shapiro-Wilk & D'Agostino K²
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 12 (Extra): NORMALITY TESTS — BMI, HbA1c, Systolic BP")

for var in ["bmi", "hba1c", "sysbp"]:
    data = df[var].dropna()
    sw_stat, sw_p = shapiro(data[:50])           # Shapiro-Wilk (best for n<50)
    da_stat, da_p = normaltest(data)             # D'Agostino K²
    print(f"\n  {var.upper()}:")
    result("  Shapiro-Wilk statistic:", f"{sw_stat:.4f}")
    result("  Shapiro-Wilk p-value:", f"{sw_p:.4f}  {'→ NOT normal' if sw_p < 0.05 else '→ Normal'}")
    result("  D'Agostino K² statistic:", f"{da_stat:.4f}")
    result("  D'Agostino K² p-value:", f"{da_p:.4f}  {'→ NOT normal' if da_p < 0.05 else '→ Normal'}")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 13: LEVENE'S TEST OF EQUAL VARIANCES
# Before doing ANOVA, verify homogeneity of variances
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 13 (Extra): LEVENE'S TEST — BMI variance across BP categories")

groups_lev = [g["bmi"].dropna().values for _, g in df.groupby("bp_category")]
lev_stat, lev_p = levene(*groups_lev)

result("Levene statistic:", f"{lev_stat:.4f}")
result("p-value:", f"{lev_p:.4f}")
result("Interpretation:",
       "SIGNIFICANT — Variances are NOT equal (consider Welch's ANOVA)."
       if lev_p < 0.05 else
       "NOT significant — Variances are equal; ANOVA assumption met.")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 14: CRAMÉR'S V (effect size for Chi-Square)
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 14 (Extra): CRAMÉR'S V — Effect size for Gender vs Insulin Use")

n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
result("Cramér's V:", f"{cramers_v:.4f}")
result("Interpretation (rule of thumb):",
       "Small" if cramers_v < 0.1 else ("Medium" if cramers_v < 0.3 else "Large"))

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 15: COHEN'S D (effect size for t-test)
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 15 (Extra): COHEN'S D — Effect size for BMI by Gender")

pooled_std = np.sqrt((male_bmi.std()**2 + female_bmi.std()**2) / 2)
cohens_d = (male_bmi.mean() - female_bmi.mean()) / pooled_std

result("Cohen's d:", f"{cohens_d:.4f}")
result("Interpretation:",
       "Small" if abs(cohens_d) < 0.2 else ("Medium" if abs(cohens_d) < 0.5 else "Large"))

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 16: POINT-BISERIAL CORRELATION
# Binary vs Continuous: Insulin use vs HbA1c
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 16 (Extra): POINT-BISERIAL CORRELATION — Insulin use vs HbA1c")

df_clean = df[["medication_insulin", "hba1c"]].dropna()
r_pb, p_pb = stats.pointbiserialr(df_clean["medication_insulin"], df_clean["hba1c"])

result("r (point-biserial):", f"{r_pb:.4f}")
result("p-value:", f"{p_pb:.4f}")
result("Interpretation:",
       "SIGNIFICANT — Insulin use is significantly correlated with HbA1c."
       if p_pb < 0.05 else
       "NOT significant — No significant correlation.")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 17: LOGISTIC REGRESSION
# Predict microvascular complications from HbA1c + BMI + age
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 17 (Extra): LOGISTIC REGRESSION — Predicting Microvascular Complications")

log_df = df[["hba1c", "bmi", "age", "has_microvascular"]].dropna()
X_log = log_df[["hba1c", "bmi", "age"]]
y_log = log_df["has_microvascular"]

X_sm = sm.add_constant(X_log)
logit_model = sm.Logit(y_log, X_sm).fit(disp=0)
print(f"\n{logit_model.summary().tables[1]}")
print("\n  Interpretation: Coefficients show log-odds change per unit increase.")
print("  Positive coeff → increases odds of microvascular complications.")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — Test 18: DUNN'S POST-HOC for Kruskal-Wallis
# (manual using pairwise Mann-Whitney with Bonferroni)
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 18 (Extra): DUNN'S POST-HOC — HbA1c across BP Categories")

cat_groups = {cat: g["hba1c"].dropna().values
              for cat, g in df.groupby("bp_category")}
cats = list(cat_groups.keys())
n_comps = len(cats) * (len(cats) - 1) // 2  # Bonferroni denominator
print()

for i in range(len(cats)):
    for j in range(i + 1, len(cats)):
        g1, g2 = cats[i], cats[j]
        u, p_raw = mannwhitneyu(cat_groups[g1], cat_groups[g2], alternative="two-sided")
        p_adj = min(p_raw * n_comps, 1.0)  # Bonferroni correction
        sig = "* SIGNIFICANT" if p_adj < 0.05 else "not significant"
        result(f"  {g1} vs {g2}:", f"U={u:.0f}, p_adj={p_adj:.4f} — {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
section("GENERATING VISUALISATIONS...")

fig = plt.figure(figsize=(20, 28))
fig.patch.set_facecolor("#0f172a")
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
AXES_BG    = "#1e293b"
TEXT_COL   = "#e2e8f0"
ACCENT1    = "#38bdf8"
ACCENT2    = "#f472b6"
ACCENT3    = "#34d399"
ACCENT4    = "#fb923c"
PALETTE    = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#a78bfa"]

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(AXES_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)

# ── Plot 1: Chi-Square — Gender vs Insulin (stacked bar)
ax1 = fig.add_subplot(gs[0, 0])
ct_pct = pd.crosstab(df_mf["gender"], df_mf["medication_insulin"], normalize="index") * 100
ct_pct.columns = ["No Insulin", "Insulin"]
ct_pct.plot(kind="bar", stacked=True, ax=ax1, color=[ACCENT1, ACCENT2], edgecolor="#0f172a", width=0.5)
style_ax(ax1, f"Chi-Square: Gender vs Insulin\nχ²={chi2:.2f}, p={p:.3f}", "Gender", "Proportion (%)")
ax1.legend(fontsize=7, facecolor=AXES_BG, labelcolor=TEXT_COL, framealpha=0.7)
ax1.set_xticklabels(["Female", "Male"], rotation=0)

# ── Plot 2: Independent T-Test — BMI by Gender (boxplot)
ax2 = fig.add_subplot(gs[0, 1])
bmi_data = df_mf[["gender", "bmi"]].dropna()
bmi_data["Gender"] = bmi_data["gender"].map({"M": "Male", "F": "Female"})
sns.boxplot(data=bmi_data, x="Gender", y="bmi", ax=ax2,
            palette={"Male": ACCENT1, "Female": ACCENT2},
            linewidth=1.2, flierprops=dict(marker="o", markerfacecolor=ACCENT3, markersize=3))
style_ax(ax2, f"Independent t-Test: BMI by Gender\nt={t_stat:.2f}, p={p_t:.3f}", "Gender", "BMI")

# ── Plot 3: Pearson Correlation — Age vs Systolic BP (scatter)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df["age"], df["sysbp"], alpha=0.5, s=18, color=ACCENT3, edgecolors="none")
m, b = np.polyfit(df["age"].dropna(), df.loc[df["age"].notna(), "sysbp"], 1)
x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
ax3.plot(x_line, m * x_line + b, color=ACCENT4, lw=2, label=f"r={r_pearson:.3f}, p={p_pearson:.3f}")
ax3.legend(fontsize=7, facecolor=AXES_BG, labelcolor=TEXT_COL, framealpha=0.7)
style_ax(ax3, "Pearson Correlation: Age vs Systolic BP", "Age", "Systolic BP")

# ── Plot 4: ANOVA — BMI across BP categories (violin)
ax4 = fig.add_subplot(gs[1, 0])
sns.violinplot(data=df, x="bp_category", y="bmi", ax=ax4,
               palette={"Normal": ACCENT3, "Elevated": ACCENT4, "High": ACCENT2},
               inner="box", linewidth=0.8)
style_ax(ax4, f"ANOVA: BMI across BP Categories\nF={f_stat:.2f}, p={p_anova:.3f}", "BP Category", "BMI")

# ── Plot 5: Mann-Whitney — HbA1c by Gender (violin)
ax5 = fig.add_subplot(gs[1, 1])
df_mf2 = df[df["gender"].isin(["M", "F"])].copy()
df_mf2["Gender"] = df_mf2["gender"].map({"M": "Male", "F": "Female"})
sns.violinplot(data=df_mf2, x="Gender", y="hba1c", ax=ax5,
               palette={"Male": ACCENT1, "Female": ACCENT2},
               inner="box", linewidth=0.8)
style_ax(ax5, f"Mann-Whitney U: HbA1c by Gender\nU={u_stat:.0f}, p={p_mw:.3f}", "Gender", "HbA1c (%)")

# ── Plot 6: Normality — Q-Q plot of HbA1c
ax6 = fig.add_subplot(gs[1, 2])
hba1c_vals = df["hba1c"].dropna()
(osm, osr), (slope, intercept, r) = stats.probplot(hba1c_vals, dist="norm")
ax6.scatter(osm, osr, color=ACCENT1, s=12, alpha=0.7)
ax6.plot(osm, slope * np.array(osm) + intercept, color=ACCENT4, lw=2)
style_ax(ax6, "Q-Q Plot: HbA1c (normality check)", "Theoretical Quantiles", "Sample Quantiles")

# ── Plot 7: Correlation matrix heatmap
ax7 = fig.add_subplot(gs[2, :2])
num_cols = ["age", "bmi", "hba1c", "sysbp", "diabp", "hdl", "ldl", "tri"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=ax7, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, linecolor="#0f172a",
            cbar_kws={"shrink": 0.8})
ax7.set_facecolor(AXES_BG)
ax7.set_title("Correlation Matrix (Pearson)", color=TEXT_COL, fontsize=10, fontweight="bold", pad=8)
ax7.tick_params(colors=TEXT_COL, labelsize=7)
for spine in ax7.spines.values():
    spine.set_edgecolor("#334155")

# ── Plot 8: Logistic Regression — HbA1c vs Microvascular risk
ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(df[df["has_microvascular"] == 0]["hba1c"],
            df[df["has_microvascular"] == 0]["bmi"],
            color=ACCENT1, s=12, alpha=0.6, label="No Micro-complications")
ax8.scatter(df[df["has_microvascular"] == 1]["hba1c"],
            df[df["has_microvascular"] == 1]["bmi"],
            color=ACCENT2, s=12, alpha=0.6, label="Has Micro-complications")
ax8.legend(fontsize=6, facecolor=AXES_BG, labelcolor=TEXT_COL, framealpha=0.7)
style_ax(ax8, "Logistic Regression: HbA1c + BMI\nvs Microvascular Risk",
         "HbA1c (%)", "BMI")

# ── Plot 9: Point-Biserial — Insulin vs HbA1c (strip + box)
ax9 = fig.add_subplot(gs[3, 0])
df_pb = df[["medication_insulin", "hba1c"]].dropna().copy()
df_pb["Insulin"] = df_pb["medication_insulin"].map({0: "No", 1: "Yes"})
sns.boxplot(data=df_pb, x="Insulin", y="hba1c", ax=ax9,
            palette={"No": ACCENT3, "Yes": ACCENT4},
            linewidth=1.0)
sns.stripplot(data=df_pb, x="Insulin", y="hba1c", ax=ax9,
              color="white", alpha=0.3, size=3, jitter=True)
style_ax(ax9, f"Point-Biserial: Insulin vs HbA1c\nr={r_pb:.3f}, p={p_pb:.3f}",
         "On Insulin", "HbA1c (%)")

# ── Plot 10: Distribution comparison of HbA1c by BP category
ax10 = fig.add_subplot(gs[3, 1])
for cat, color in zip(["Normal", "Elevated", "High"], [ACCENT3, ACCENT4, ACCENT2]):
    data_cat = df[df["bp_category"] == cat]["hba1c"].dropna()
    ax10.hist(data_cat, bins=12, alpha=0.5, color=color, label=cat, edgecolor="none")
style_ax(ax10, "HbA1c Distribution by BP Category\n(Kruskal-Wallis context)",
         "HbA1c (%)", "Frequency")
ax10.legend(fontsize=7, facecolor=AXES_BG, labelcolor=TEXT_COL, framealpha=0.7)

# ── Plot 11: Regression residuals plot
ax11 = fig.add_subplot(gs[3, 2])
fitted = model.fittedvalues
residuals = model.resid
ax11.scatter(fitted, residuals, alpha=0.5, s=14, color=ACCENT1, edgecolors="none")
ax11.axhline(0, color=ACCENT4, lw=1.5, linestyle="--")
style_ax(ax11, "Regression Residuals\nAge → Systolic BP",
         "Fitted Values", "Residuals")

# Title
fig.text(0.5, 0.99, "Bivariate Statistical Analysis — Diabetes Dataset",
         ha="center", va="top", color=TEXT_COL, fontsize=15, fontweight="bold")

plt.savefig("/mnt/user-data/outputs/bivariate_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

section("ALL TESTS COMPLETE")
print("\n  Visualisation saved to: bivariate_analysis.png")
print("  Tests performed:")
print("    From lecture : Chi-Square, Fisher's Exact, t-test (independent & paired),")
print("                   ANOVA + Tukey HSD, Mann-Whitney U, Kruskal-Wallis,")
print("                   Pearson Correlation, Spearman Correlation, Simple Regression")
print("    Extras added : Normality (Shapiro-Wilk, D'Agostino K²), Levene's Test,")
print("                   Cramér's V, Cohen's d, Point-Biserial Correlation,")
print("                   Logistic Regression, Dunn's Post-Hoc")
print()