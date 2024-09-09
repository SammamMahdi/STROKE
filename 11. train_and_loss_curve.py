import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd

# Load dataset
df = pd.read_csv("COMBO2.csv")
drop_columns = ['Filename', 'is_stroke_face', 'rightEyeLower1_130', 'rightEyeLower1_25', 'rightEyeLower1_110',
                'rightEyeLower1_24', 'rightEyeLower1_23', 'rightEyeLower1_22', 'rightEyeLower1_26',
                'rightEyeLower1_112', 'rightEyeLower1_243', 'rightEyeUpper0_246', 'rightEyeUpper0_161',
                'rightEyeUpper0_160', 'rightEyeUpper0_159', 'rightEyeUpper0_158', 'rightEyeUpper0_157',
                'rightEyeUpper0_173', 'rightEyeLower0_33', 'rightEyeLower0_7', 'rightEyeLower0_163',
                'rightEyeLower0_144', 'rightEyeLower0_145', 'rightEyeLower0_153', 'rightEyeLower0_154',
                'rightEyeLower0_155', 'rightEyeLower0_133', 'leftEyeLower3_372', 'leftEyeLower3_340',
                'leftEyeLower3_346', 'leftEyeLower3_347', 'leftEyeLower3_348', 'leftEyeLower3_349', 'leftEyeLower3_350',
                'leftEyeLower3_357', 'leftEyeLower3_465', 'rightEyeLower2_226', 'rightEyeLower2_31',
                'rightEyeLower2_228', 'rightEyeLower2_229', 'rightEyeLower2_230', 'rightEyeLower2_231',
                'rightEyeLower2_232', 'rightEyeLower2_233', 'rightEyeLower2_244', 'rightEyeUpper2_113',
                'rightEyeUpper2_225', 'rightEyeUpper2_224', 'rightEyeUpper2_223', 'rightEyeUpper2_222',
                'rightEyeUpper2_221', 'rightEyeUpper2_189', 'leftEyeUpper1_467', 'leftEyeUpper1_260',
                'leftEyeUpper1_259', 'leftEyeUpper1_257', 'leftEyeUpper1_258', 'leftEyeUpper1_286', 'leftEyeUpper1_414',
                'leftEyeLower2_446', 'leftEyeLower2_261', 'leftEyeLower2_448', 'leftEyeLower2_449', 'leftEyeLower2_450',
                'leftEyeLower2_451', 'leftEyeLower2_452', 'leftEyeLower2_453', 'leftEyeLower2_464', 'leftEyeLower1_359',
                'leftEyeLower1_255', 'leftEyeLower1_339', 'leftEyeLower1_254', 'leftEyeLower1_253', 'leftEyeLower1_252',
                'leftEyeLower1_256', 'leftEyeLower1_341', 'leftEyeLower1_463', 'leftEyeUpper2_342', 'leftEyeUpper2_445',
                'leftEyeUpper2_444', 'leftEyeUpper2_443', 'leftEyeUpper2_442', 'leftEyeUpper2_441', 'leftEyeUpper2_413',
                'rightEyebrowLower_35', 'rightEyebrowLower_124', 'rightEyebrowLower_46', 'rightEyebrowLower_53',
                'rightEyebrowLower_52', 'rightEyebrowLower_65', 'leftEyebrowLower_265', 'leftEyebrowLower_353',
                'leftEyebrowLower_276', 'leftEyebrowLower_283', 'leftEyebrowLower_282', 'leftEyebrowLower_295',
                'rightEyeUpper1_247', 'rightEyeUpper1_30', 'rightEyeUpper1_29', 'rightEyeUpper1_27',
                'rightEyeUpper1_28', 'rightEyeUpper1_56', 'rightEyeUpper1_190', 'leftEyeUpper0_466',
                'leftEyeUpper0_388', 'leftEyeUpper0_387', 'leftEyeUpper0_386', 'leftEyeUpper0_385', 'leftEyeUpper0_384',
                'leftEyeUpper0_398', 'noseBottom_2', 'midwayBetweenEyes_168', 'noseRightCorner_98',
                'noseLeftCorner_327']

X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
hyperparameters_RFC = {"n_estimators": 400, "max_depth": 100, "min_samples_split": 10, "min_samples_leaf": 4,
                       "max_features": 0.25, "bootstrap": False, "criterion": "gini"}  # new hyperparameters for top 10

hyperparameters_XGB = {'max_depth': 7, 'min_child_weight': 2, 'learning_rate': 0.2, 'subsample': 0.8,
                       'colsample_bytree': 1.0, 'gamma': 0.3, 'n_estimators': 400, 'use_label_encoder': False,
                       'eval_metric': 'rmse', 'objective': 'binary:logistic'}  # new hyperparameters for top 10

hyperparameters_CB = {'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
                      'colsample_bylevel': 0.917411003148779,
                      'depth': 8, 'grow_policy': 'SymmetricTree', 'iterations': 918, 'l2_leaf_reg': 8,
                      'learning_rate': 0.29287291117375575, 'max_bin': 231, 'min_data_in_leaf': 9, 'od_type': 'Iter',
                      'od_wait': 21, 'one_hot_max_size': 7, 'random_strength': 0.6963042728397884,
                      'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999}
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)


# Function to calculate accuracy and log loss over training "epochs" for tree models
def calculate_metrics_curve(model, X_train, X_test, y_train, y_test, n_splits=10):
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    split_size = len(X_train) // n_splits

    for i in range(1, n_splits + 1):
        X_train_partial = X_train[:split_size * i]
        y_train_partial = y_train[:split_size * i]

        model.fit(X_train_partial, y_train_partial)

        # Predictions and probabilities for both train and test sets
        y_train_pred = model.predict(X_train_partial)
        y_test_pred = model.predict(X_test)

        y_train_proba = model.predict_proba(X_train_partial)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Accuracy scores
        train_acc.append(accuracy_score(y_train_partial, y_train_pred))
        test_acc.append(accuracy_score(y_test, y_test_pred))

        # Log loss scores
        train_loss.append(log_loss(y_train_partial, y_train_proba))
        test_loss.append(log_loss(y_test, y_test_proba))

    return train_acc, test_acc, train_loss, test_loss


# Initialize models
rf_model = RandomForestClassifier(**hyperparameters_RFC, random_state=150)
xgb_model = XGBClassifier(**hyperparameters_XGB)
cb_model = CatBoostClassifier(**hyperparameters_CB)

# Get metrics for Random Forest
rf_train_acc, rf_test_acc, rf_train_loss, rf_test_loss = calculate_metrics_curve(rf_model, X_train, X_test, y_train,
                                                                                 y_test)

# Get metrics for XGBoost
xgb_train_acc, xgb_test_acc, xgb_train_loss, xgb_test_loss = calculate_metrics_curve(xgb_model, X_train, X_test,
                                                                                     y_train, y_test)

# Get metrics for CatBoost
cb_train_acc, cb_test_acc, cb_train_loss, cb_test_loss = calculate_metrics_curve(cb_model, X_train, X_test, y_train,
                                                                                 y_test)

# Plot Training and Validation Curves for Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rf_train_acc, label='RF Train Accuracy', color='blue', linestyle='--')
plt.plot(rf_test_acc, label='RF Test Accuracy', color='blue')

plt.plot(xgb_train_acc, label='XGB Train Accuracy', color='green', linestyle='--')
plt.plot(xgb_test_acc, label='XGB Test Accuracy', color='green')

plt.plot(cb_train_acc, label='CB Train Accuracy', color='red', linestyle='--')
plt.plot(cb_test_acc, label='CB Test Accuracy', color='red')

plt.title('Training and Validation Accuracy')
plt.xlabel('Training Size Split (Simulated Epochs)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Training and Validation Curves for Loss
plt.subplot(1, 2, 2)
plt.plot(rf_train_loss, label='RF Train Loss', color='blue', linestyle='--')
plt.plot(rf_test_loss, label='RF Test Loss', color='blue')

plt.plot(xgb_train_loss, label='XGB Train Loss', color='green', linestyle='--')
plt.plot(xgb_test_loss, label='XGB Test Loss', color='green')

plt.plot(cb_train_loss, label='CB Train Loss', color='red', linestyle='--')
plt.plot(cb_test_loss, label='CB Test Loss', color='red')

plt.title('Training and Validation Loss')
plt.xlabel('Training Size Split (Simulated Epochs)')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
