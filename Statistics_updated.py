# ======================== Imports ========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    mean_squared_error,
    r2_score,
    mean_absolute_error)
from math import sqrt

# ======================== Helper Functions ========================

def report_metrics(y_true, y_pred, label="Set"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    reg_test = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    r2 = reg_test.score(y_pred.reshape(-1, 1), y_true)
    print(f"{label} RMSE: {rmse:.3f}")
    print(f"{label} R²: {r2:.3f}")
    return rmse, r2


def plot_regression(y_true, y_pred, r2, label="Set"):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y_pred, y_true, alpha=0.7, color='steelblue')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"{label} Regression Plot")
    plt.ylabel("mPAP (mmHg)")
    plt.xlabel("Regression Value")
    #plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def bland_altman_plot(y_true, y_pred, label="Set"):
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals)
    print(f"{label} SD: {sd_diff:.3f}")

    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(mean_vals, diff_vals, alpha=0.6, color='darkorange')
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f"Mean diff = {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96 * sd_diff, color='red', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96 * sd_diff, color='red', linestyle='--', label='-1.96 SD')
    plt.title(f"{label} Bland–Altman Plot")
    plt.xlabel("Mean of True & Predicted")
    plt.ylabel("Difference (Pred - True)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cross_val(X, y):
    split = [0.33]
    for jj in split:
        r2_values_test = []
        r2_values_train = []
        MAE_values_test = []
        MAE_values_train = []
        RMSE_values_test = []
        RMSE_values_train = []
        true_vals_all = []
        pred_all = []

        num_repeats = int(1 / jj)

        for i in range(num_repeats):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=jj, random_state=i)

            # Train model on selected features
            reg = LinearRegression().fit(X_train, y_train)
            y_pred_test = reg.predict(X_test)
            y_pred_train = reg.predict(X_train)

            # Store predictions
            true_vals_all.extend(y_test)
            pred_all.extend(y_pred_test)

            # Metrics
            r2_values_test.append(r2_score(y_test, y_pred_test))
            r2_values_train.append(r2_score(y_train, y_pred_train))
            MAE_values_test.append(mean_absolute_error(y_test, y_pred_test))
            MAE_values_train.append(mean_absolute_error(y_train, y_pred_train))
            RMSE_values_test.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            RMSE_values_train.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        # Print results for this split
        print(f"=== Test Size: {jj * 100:.0f}% ===")
        print('Train R²:', np.round(r2_values_train, 4))
        print('Test R²:', np.round(r2_values_test, 4))
        print('Train MAE:', np.round(MAE_values_train, 4))
        print('Test MAE:', np.round(MAE_values_test, 4))
        print('Train RMSE:', np.round(RMSE_values_train, 4))
        print('Test RMSE:', np.round(RMSE_values_test, 4))


# ======================== File Paths ========================

# CHANGE HERE WITH YOUR PATH
base_path = r'C:\Users\malak\PycharmProjects\PA_SSA_code'

# Simple encoding modes + mPAP
data_path = f"{base_path}\\all_simple_encoding.xlsx"

# Complex encoding modes + mPAP, divided into the two cohorts (AHC, ASPIRE)
AHC_path = f"{base_path}\\AHC.xlsx"
ASPIRE_path = f"{base_path}\\ASPIRE.xlsx"

# ======================== Load Data ========================
simple_encoding_data = pd.read_excel(data_path)
AHC_data = pd.read_excel(AHC_path)
ASPIRE_data = pd.read_excel(ASPIRE_path)

# ======================== Simple Encoding Regression ========================
print("\n==== Linear Regression: Predicting mPAP, using simple encoding ====")

X = simple_encoding_data.iloc[:, :6]
y = simple_encoding_data['mPAP']


reg = LinearRegression().fit(X, y)
w = reg.coef_
y_pred = reg.predict(X)

# Report metrics
rmse, r2 = report_metrics(y, y_pred, "Simple encoding")

# Plots
plot_regression(y, y_pred, r2, "Simple encoding")
bland_altman_plot(y, y_pred, "Simple encoding")

print("\n==== Cross-validation ====")
cross_val(X, y)

# ======================== Complex Encoding Regression ========================
print("\n==== Linear Regression: Predicting mPAP, using complex encoding ====")

combined_data = pd.concat([AHC_data, ASPIRE_data], ignore_index=True)

X = combined_data.iloc[:, :13]
y = combined_data['mPAP']
target_col = [14, 15]

reg = LinearRegression().fit(X, y)
w = reg.coef_
y_pred = reg.predict(X)

# Report metrics
rmse, r2 = report_metrics(y, y_pred, "Complex encoding")

# Plots
plot_regression(y, y_pred, r2, "Complex encoding")
bland_altman_plot(y, y_pred, "Complex encoding")

print("\n==== Cross-validation ====")
cross_val(X, y)


# ======================== Train/Test Analysis 1 ========================
print("\n===== Linear Regression: Train on AHC, Test on ASPIRE =====")

# Use first 7 components
X_train = AHC_data.iloc[:, :6]
X_test = ASPIRE_data.iloc[:, :6]
y_train = AHC_data.iloc[:, 13]
y_test = ASPIRE_data.iloc[:, 13]


# Fit model
reg = LinearRegression().fit(X_train, y_train)
w = reg.coef_
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Report metrics
rmse_train, r2_train = report_metrics(y_train, y_train_pred, "Train (AHC)")
rmse_test, r2_test = report_metrics(y_test, y_test_pred, "Test (ASPIRE)")


# Plots
plot_regression(y_train, y_train_pred, r2_train, "Train (AHC)")#, highlight_indices)
plot_regression(y_test, y_test_pred, r2_test, "Test (ASPIRE)")
bland_altman_plot(y_train, y_train_pred, "Train (AHC)")
bland_altman_plot(y_test, y_test_pred, "Test (ASPIRE)")



# ======================== LDA Classification ========================
print("\n===== LDA: Train on AHC, Test on ASPIRE =====")

for target in target_col:
    y_train = AHC_data.iloc[:, target]
    y_test = ASPIRE_data.iloc[:, target]
    print(f"\n==== LDA: Classifying {AHC_data.keys()[target]} using AHC vs ASPIRE ====")
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title("LDA Confusion Matrix - ASPIRE as validation cohort")
    plt.show()

    print("LDA Coefficients:\n", lda.coef_)



# ======================== Analysis 2 ========================
print("\n===== Analysis 2a: Train on AHC, Test on ASPIRE, group 1 =====")


ASPIRE_data = pd.read_excel(f"{base_path}\\ASPIRE_G1.xlsx")


# Use first 7 components
X_train = AHC_data.iloc[:, :6]
X_test = ASPIRE_data.iloc[:, :6]
y_train = AHC_data.iloc[:, 13]
y_test = ASPIRE_data.iloc[:, 13]

# Fit model
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


# Report metrics
rmse_train, r2_train = report_metrics(y_train, y_train_pred, "Train (AHC)")
rmse_test, r2_test = report_metrics(y_test, y_test_pred, "Test (ASPIRE G1+controls)")

# Plots
plot_regression(y_train, y_train_pred, r2_train, "Train (AHC)")
plot_regression(y_test, y_test_pred, r2_test, "Test (ASPIRE G1+controls)")
bland_altman_plot(y_train, y_train_pred, "Train (AHC)")
bland_altman_plot(y_test, y_test_pred, "Test (ASPIRE G1+controls)")


print("\n===== Analysis 2b: Train on AHC, Test on ASPIRE, groups 2-4 =====")


ASPIRE_data = pd.read_excel(f"{base_path}\\ASPIRE_control+G2-4.xlsx")


# Use first 7 components
X_train = AHC_data.iloc[:, :6]
X_test = ASPIRE_data.iloc[:, :6]
y_train = AHC_data.iloc[:, 13]
y_test = ASPIRE_data.iloc[:, 13]

# Fit model
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


# Report metrics
rmse_train, r2_train = report_metrics(y_train, y_train_pred, "Train (AHC)")
rmse_test, r2_test = report_metrics(y_test, y_test_pred, "Test (ASPIRE G2-4+controls)")

# Plots
plot_regression(y_train, y_train_pred, r2_train, "Train (AHC)")
plot_regression(y_test, y_test_pred, r2_test, "Test (ASPIRE G2-4+controls)")
bland_altman_plot(y_train, y_train_pred, "Train (AHC)")
bland_altman_plot(y_test, y_test_pred, "Test (ASPIRE G2-4+controls)")

print("\n===== Analysis 2c: Train on AHC, Test on ASPIRE, matched mPAP ranges =====")

AHC_data = pd.read_excel(f"{base_path}\\AHC_matched.xlsx")
ASPIRE_data = pd.read_excel(f"{base_path}\\ASPIRE_G1_matched.xlsx")


# Use first 7 components
X_train = AHC_data.iloc[:, :6]
X_test = ASPIRE_data.iloc[:, :6]
y_train = AHC_data.iloc[:, 13]
y_test = ASPIRE_data.iloc[:, 13]

# Fit model
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


# Report metrics
rmse_train, r2_train = report_metrics(y_train, y_train_pred, "Train (AHC matched mPAP distribution)")
rmse_test, r2_test = report_metrics(y_test, y_test_pred, "Test (ASPIRE matched mPAP distribution)")

# Plots
plot_regression(y_train, y_train_pred, r2_train, "Train (AHC matched mPAP distribution)")
plot_regression(y_test, y_test_pred, r2_test, "Test (ASPIRE matched mPAP distribution)")
bland_altman_plot(y_train, y_train_pred, "Train (AHC matched mPAP distribution)")
bland_altman_plot(y_test, y_test_pred, "Test (ASPIRE matched mPAP distribution)")

# ======================== Analysis 3 ========================
print("\n===== Analysis 3: Combined Dataset with Split & 3-Fold CV =====")

# Combine datasets
X_combined = pd.concat([AHC_data.iloc[:, :7], ASPIRE_data.iloc[:, :7]], axis=0, ignore_index=True)
y_combined = pd.concat([AHC_data.iloc[:, 13], ASPIRE_data.iloc[:, 13]], axis=0, ignore_index=True)

# Split 70/30
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.33, random_state=42)

# Train
reg = LinearRegression().fit(X_train, y_train)
w = reg.coef_
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


# Metrics
rmse_train, r2_train = report_metrics(y_train, y_train_pred, "Train (Combined)")
rmse_test, r2_test = report_metrics(y_test, y_test_pred, "Test (Combined)")

# Plots
plot_regression(y_train, y_train_pred, r2_train, "Train (Combined)")
plot_regression(y_test, y_test_pred, r2_test, "Test (Combined)")
bland_altman_plot(y_train, y_train_pred, "Train (Combined)")
bland_altman_plot(y_test, y_test_pred, "Test (Combined)")

# ======================== 3-Fold Cross Validation ========================
print("\n===== 3-Fold Cross Validation on Combined Data =====")

kf = KFold(n_splits=3, shuffle=True, random_state=42)
r2_scores = []
rmse_scores = []

for i, (train_idx, test_idx) in enumerate(kf.split(X_combined)):
    X_tr, X_te = X_combined.iloc[train_idx], X_combined.iloc[test_idx]
    y_tr, y_te = y_combined.iloc[train_idx], y_combined.iloc[test_idx]

    reg = LinearRegression().fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    rmse = sqrt(mean_squared_error(y_te, y_pred))

    r2_scores.append(r2)
    rmse_scores.append(rmse)

    print(f"Fold {i + 1}: R² = {r2:.3f}, RMSE = {rmse:.3f}")

print(f"\nAverage 3-Fold R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Average 3-Fold RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")


# Normalize regression vector for co-linearity analysis
reg = LinearRegression(fit_intercept=True).fit(X, y)
a = reg.coef_ / np.linalg.norm(reg.coef_)


for target in target_col:
    print(f"\n==== LDA for Target: {combined_data.keys()[target]} ====")

    y = combined_data.iloc[:, target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lda = LDA()
    lda.fit(X_train, y_train)
    b = lda.coef_ / np.linalg.norm(lda.coef_)

    y_pred = lda.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    CI = float(np.dot(a, b.T))

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Co-linearity Index (CI): {CI:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {combined_data.keys()[target]}')
    plt.show()


# ======================== Repeated Evaluation (100 Runs) ========================
print("\n==== Repeated LDA Evaluation (100 Runs) ====")

for target in target_col:
    print(f"\n==== Repeated LDA for Target: {combined_data.keys()[target]} ====")
    y = combined_data.iloc[:, target]

    accuracies, precisions, recalls, f1s = [], [], [], []

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

        if i == 0:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix (Run 1) for {combined_data.keys()[target]}')
            plt.show()

    # Convert to numpy for summary statistics
    accuracies = np.array(accuracies)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    print(f"Average Accuracy : {accuracies.mean():.4f} ± {accuracies.std():.4f}")
    print(f"Average Precision: {precisions.mean():.4f} ± {precisions.std():.4f}")
    print(f"Average Recall   : {recalls.mean():.4f} ± {recalls.std():.4f}")
    print(f"Average F1 Score : {f1s.mean():.4f} ± {f1s.std():.4f}")

