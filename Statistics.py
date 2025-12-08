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
    classification_report
)

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

X = simple_encoding_data.iloc[:, :6]
y_pressure = simple_encoding_data['mPAP']

print("\n==== Linear Regression: Predicting mPAP, using simple encoding ====")

reg = LinearRegression(fit_intercept=True).fit(X, y_pressure)
w = reg.coef_

regression_score = reg.score(X, y_pressure)
print(f"Regression R²: {regression_score:.4f}")

all_scores = reg.intercept_ + X @ w

plt.figure()
plt.scatter(all_scores, y_pressure, color='b')
plt.plot(all_scores, reg.predict(X), color='red', linewidth=2)
plt.xlabel('Regression')
plt.ylabel('mPAP (mmHg)')
plt.axis([-10, 130, -10, 130])
plt.title("Regression Plot - Simple Encoding")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

split_rmse_values = []
split_rmse_values_train = []
split_rmse_std_values = []
split_rmse_std_values_train = []
split_r2_test = []

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
        X_train, X_test, y_train, y_test = train_test_split(X, y_pressure, test_size=jj, random_state=i)

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

    # Compute final R² on all collected predictions
    split_r2_test.append(r2_score(true_vals_all, pred_all))

print('\nFinal R² over all test splits:', split_r2_test)






# ==================== Using AHC as train and AHC as test ========================

X_train = AHC_data.iloc[:, :13]
X_test = ASPIRE_data.iloc[:, :13]

target_col = [14, 15]

# ======================== LDA Classification ========================
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



# ======================== Regression: Pressure Prediction ========================
print("\n==== Linear Regression: Predicting mPAP ====")
y_train_pressure = AHC_data.iloc[:, 13]
y_test_pressure = ASPIRE_data.iloc[:, 13]

reg = LinearRegression(fit_intercept=True).fit(X_train, y_train_pressure)
w = reg.coef_

train_score = reg.score(X_train, y_train_pressure)
print(f"Train R²: {train_score:.4f}")

all_scores = reg.intercept_ + X_train @ w

plt.figure()
plt.scatter(all_scores, y_train_pressure, color='b')
plt.plot(all_scores, reg.predict(X_train), color='red', linewidth=2)
plt.xlabel('Regression')
plt.ylabel('mPAP (mmHg)')
plt.axis([-10, 130, -10, 130])
plt.title("Train (AHC) Regression Plot")
plt.show()

aspire_scores = reg.intercept_ + X_test @ w

plt.figure()
plt.scatter(aspire_scores, y_test_pressure, color='b')
plt.plot(aspire_scores, reg.predict(X_test), color='red', linewidth=2)
plt.xlabel('Regression')
plt.ylabel('mPAP (mmHg)')
plt.axis([-10, 130, -10, 130])
plt.title("Test (ASPIRE) Regression Plot")
plt.show()

reg = LinearRegression().fit(np.array(aspire_scores).reshape(-1, 1), y_test_pressure)
test_score=reg.score((np.array(aspire_scores).reshape(-1, 1)), y_test_pressure)

print(f"Test R²: {test_score:.4f}")



# =============== Combined analyses =========================================

combined_data = pd.concat([AHC_data, ASPIRE_data], ignore_index=True)

X = combined_data.iloc[:, :13]
y_pressure = combined_data['mPAP']
target_col = [14, 15]

# Normalize regression vector for co-linearity analysis
reg = LinearRegression(fit_intercept=True).fit(X, y_pressure)
a = reg.coef_ / np.linalg.norm(reg.coef_)

{AHC_data.keys()[target]}

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

# ======================== Regression: Pressure Prediction ========================
print("\n==== Linear Regression: Predicting mPAP ====")

reg = LinearRegression(fit_intercept=True).fit(X, y_pressure)
w = reg.coef_

regression_score = reg.score(X, y_pressure)
print(f"Regression R²: {regression_score:.4f}")

all_scores = reg.intercept_ + X @ w

plt.figure()
plt.scatter(all_scores, y_pressure, color='b')
plt.plot(all_scores, reg.predict(X), color='red', linewidth=2)
plt.xlabel('Regression')
plt.ylabel('mPAP (mmHg)')
plt.axis([-10, 130, -10, 130])
plt.title("Regression Plot - Combined data")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

split_rmse_values = []
split_rmse_values_train = []
split_rmse_std_values = []
split_rmse_std_values_train = []
split_r2_test = []

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
        X_train, X_test, y_train, y_test = train_test_split(X, y_pressure, test_size=jj, random_state=i)

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

    # Compute final R² on all collected predictions
    split_r2_test.append(r2_score(true_vals_all, pred_all))

print('\nFinal R² over all test splits:', split_r2_test)


