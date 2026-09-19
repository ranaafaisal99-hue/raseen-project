import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    StackingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical


# إعدادات البيانات
DATA_FILE = "nasa_sample.csv"

BASE_SENSOR_COLUMNS = [
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14"
]

TARGET_COLUMN = "RUL"
GROUP_COLUMN = "engine_id"

RANDOM_STATE = 42


# إنشاء Features مرتبطة بتطور المحرك
def create_features(df):
    df = df.copy()

    df = df.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    feature_columns = [
        "cycle"
    ]

    for sensor in BASE_SENSOR_COLUMNS:

        # القيمة الحالية
        current_column = sensor
        feature_columns.append(current_column)

        # متوسط متحرك
        rolling_mean_column = f"{sensor}_rolling_mean"

        df[rolling_mean_column] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        feature_columns.append(
            rolling_mean_column
        )

        # الانحراف المعياري المتحرك
        rolling_std_column = f"{sensor}_rolling_std"

        df[rolling_std_column] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=2
                ).std()
            )
        )

        feature_columns.append(
            rolling_std_column
        )

        # التغير عن القراءة السابقة
        change_column = f"{sensor}_change"

        df[change_column] = (
            df.groupby("engine_id")[sensor]
            .diff()
        )

        feature_columns.append(
            change_column
        )

        # متوسط التغير خلال آخر 5 قراءات
        change_mean_column = f"{sensor}_change_mean"

        df[change_mean_column] = (
            df.groupby("engine_id")[change_column]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        feature_columns.append(
            change_mean_column
        )

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df, feature_columns


# قراءة وتنظيف البيانات
def load_data():

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        GROUP_COLUMN,
        "cycle",
        TARGET_COLUMN
    ] + BASE_SENSOR_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df, feature_columns = create_features(df)

    return df, feature_columns


# تقسيم البيانات حسب المحرك
def prepare_data():

    df, feature_columns = load_data()

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    groups_train = groups.iloc[train_indices].copy()
    groups_test = groups.iloc[test_indices].copy()

    print("Dataset loaded successfully")

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Number of engines: "
        f"{df[GROUP_COLUMN].nunique()}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print("\nTraining data:")

    print(
        f"Rows: {len(X_train)}"
    )

    print(
        f"Engines: "
        f"{sorted(groups_train.unique().tolist())}"
    )

    print("\nTest data:")

    print(
        f"Rows: {len(X_test)}"
    )

    print(
        f"Engines: "
        f"{sorted(groups_test.unique().tolist())}"
    )

    print("\nFeature examples:")

    print(
        feature_columns[:15]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_columns
    )


# النماذج الأساسية
def create_models():

    models = {

        "Linear Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            )
        ]),

        "Ridge": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge()
            )
        ]),

        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "AdaBoost": AdaBoostRegressor(
            random_state=RANDOM_STATE
        ),

        "Bagging": BaggingRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


# إعدادات Bayesian Optimization
def create_search_spaces():

    search_spaces = {

        "Linear Regression": {
            "model__fit_intercept": Categorical(
                [True, False]
            )
        },

        "Ridge": {
            "model__alpha": Real(
                1e-3,
                100,
                prior="log-uniform"
            )
        },

        "Random Forest": {
            "n_estimators": Integer(
                100,
                300
            ),

            "max_depth": Integer(
                5,
                30
            ),

            "min_samples_split": Integer(
                2,
                10
            ),

            "min_samples_leaf": Integer(
                1,
                5
            ),

            "max_features": Categorical(
                [
                    "sqrt",
                    "log2",
                    1.0
                ]
            )
        },

        "Extra Trees": {
            "n_estimators": Integer(
                100,
                300
            ),

            "max_depth": Integer(
                5,
                30
            ),

            "min_samples_split": Integer(
                2,
                10
            ),

            "min_samples_leaf": Integer(
                1,
                5
            ),

            "max_features": Categorical(
                [
                    "sqrt",
                    "log2",
                    1.0
                ]
            )
        },

        "Gradient Boosting": {
            "n_estimators": Integer(
                100,
                300
            ),

            "learning_rate": Real(
                0.01,
                0.2,
                prior="log-uniform"
            ),

            "max_depth": Integer(
                2,
                8
            ),

            "min_samples_split": Integer(
                2,
                10
            ),

            "min_samples_leaf": Integer(
                1,
                5
            ),

            "subsample": Real(
                0.6,
                1.0
            )
        },

        "Hist Gradient Boosting": {
            "max_iter": Integer(
                100,
                300
            ),

            "learning_rate": Real(
                0.01,
                0.2,
                prior="log-uniform"
            ),

            "max_leaf_nodes": Integer(
                15,
                63
            ),

            "max_depth": Integer(
                3,
                15
            ),

            "min_samples_leaf": Integer(
                10,
                50
            )
        },

        "AdaBoost": {
            "n_estimators": Integer(
                50,
                250
            ),

            "learning_rate": Real(
                0.01,
                1.0,
                prior="log-uniform"
            )
        },

        "Bagging": {
            "n_estimators": Integer(
                50,
                200
            ),

            "max_samples": Real(
                0.5,
                1.0
            ),

            "max_features": Real(
                0.5,
                1.0
            )
        }
    }

    return search_spaces


# تقييم النموذج
def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        mae,
        rmse,
        r2
    )


# تدريب النماذج وتحسينها
def train_models(
    X_train,
    y_train,
    groups_train
):

    models = create_models()

    search_spaces = (
        create_search_spaces()
    )

    cv = GroupKFold(
        n_splits=5
    )

    trained_models = {}

    cv_results = []

    for name, model in models.items():

        print("\n" + "=" * 60)

        print(
            f"Training: {name}"
        )

        print("=" * 60)

        search_space = (
            search_spaces[name]
        )

        optimizer = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=15,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True
        )

        optimizer.fit(
            X_train,
            y_train,
            groups=groups_train
        )

        best_model = (
            optimizer.best_estimator_
        )

        trained_models[name] = (
            best_model
        )

        best_cv_mae = (
            -optimizer.best_score_
        )

        cv_results.append({
            "Model": name,
            "CV MAE": best_cv_mae
        })

        print(
            f"Best CV MAE: "
            f"{best_cv_mae:.4f}"
        )

        print(
            f"Best Parameters: "
            f"{optimizer.best_params_}"
        )

    cv_results_df = pd.DataFrame(
        cv_results
    )

    return (
        trained_models,
        cv_results_df
    )


# إنشاء Stacking Ensemble
def create_stacking_model(
    trained_models
):

    base_models = [

        (
            "extra_trees",
            trained_models[
                "Extra Trees"
            ]
        ),

        (
            "gradient_boosting",
            trained_models[
                "Gradient Boosting"
            ]
        ),

        (
            "hist_gradient_boosting",
            trained_models[
                "Hist Gradient Boosting"
            ]
        ),

        (
            "random_forest",
            trained_models[
                "Random Forest"
            ]
        )
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(
            alpha=1.0
        ),
        cv=5,
        n_jobs=-1
    )

    return stacking_model


# تشغيل الـML بالكامل
def run_ml_pipeline():

    print("=" * 60)

    print(
        "RASEEN ML PIPELINE"
    )

    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_columns
    ) = prepare_data()

    print(
        "\nStarting model training..."
    )

    (
        trained_models,
        cv_results
    ) = train_models(
        X_train,
        y_train,
        groups_train
    )

    print("\n" + "=" * 60)

    print(
        "GROUP CROSS-VALIDATION RESULTS"
    )

    print("=" * 60)

    print(
        cv_results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)

    print(
        "TEST SET EVALUATION"
    )

    print("=" * 60)

    results = []

    for name, model in (
        trained_models.items()
    ):

        (
            mae,
            rmse,
            r2
        ) = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"{name}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )

    print("\n" + "=" * 60)

    print(
        "Training Stacking Ensemble"
    )

    print("=" * 60)

    stacking_model = (
        create_stacking_model(
            trained_models
        )
    )

    stacking_model.fit(
        X_train,
        y_train
    )

    (
        stacking_mae,
        stacking_rmse,
        stacking_r2
    ) = evaluate_model(
        stacking_model,
        X_test,
        y_test
    )

    results.append({
        "Model": "Stacking",
        "MAE": stacking_mae,
        "RMSE": stacking_rmse,
        "R2": stacking_r2
    })

    print(
        f"Stacking: "
        f"MAE={stacking_mae:.4f}, "
        f"RMSE={stacking_rmse:.4f}, "
        f"R2={stacking_r2:.4f}"
    )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 60)

    print(
        "FINAL MODEL COMPARISON"
    )

    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # اختيار أفضل نموذج
    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    if best_model_name == "Stacking":

        best_model = (
            stacking_model
        )

    else:

        best_model = (
            trained_models[
                best_model_name
            ]
        )

    print("\n" + "=" * 60)

    print(
        f"Best Model: "
        f"{best_model_name}"
    )

    print("=" * 60)

    # حفظ النموذج
    joblib.dump(
        best_model,
        "best_rul_model.pkl"
    )

    # حفظ أسماء الـFeatures
    joblib.dump(
        feature_columns,
        "model_features.pkl"
    )

    # حفظ نتائج النماذج
    results_df.to_csv(
        "model_results.csv",
        index=False
    )

    # حفظ نتائج الـCV
    cv_results.to_csv(
        "cv_results.csv",
        index=False
    )

    print("\nFiles saved:")

    print(
        "best_rul_model.pkl"
    )

    print(
        "model_features.pkl"
    )

    print(
        "model_results.csv"
    )

    print(
        "cv_results.csv"
    )

    return (
        best_model,
        best_model_name,
        results_df
    )

# ============================================================
# RUL Prediction for Supervisor / Agent
# ============================================================

def get_dataset_engine_id(engine_id):
    """
    Convert the fleet registry engine ID to the numeric
    engine_id used in nasa_sample.csv.
    """

    engine_mapping = {
        "ENGINE-001": 4,
        "ENGINE-002": 8,
    }

    engine_id_upper = str(engine_id).upper()

    if engine_id_upper in engine_mapping:
        return engine_mapping[engine_id_upper]

    try:
        return int(engine_id)
    except ValueError:
        raise ValueError(
            f"Unknown engine ID: {engine_id}"
        )


def create_prediction_features(history_df):
    """
    Create the same temporal features used during model training.

    The function receives the history of one engine and returns
    the feature row corresponding to the latest reading.
    """

    df = history_df.copy()

    df = df.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    feature_columns = [
        "cycle"
    ]

    for sensor in BASE_SENSOR_COLUMNS:

        # Current sensor value
        feature_columns.append(
            sensor
        )

        # Rolling mean
        rolling_mean_column = (
            f"{sensor}_rolling_mean"
        )

        df[rolling_mean_column] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        feature_columns.append(
            rolling_mean_column
        )

        # Rolling standard deviation
        rolling_std_column = (
            f"{sensor}_rolling_std"
        )

        df[rolling_std_column] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=2
                ).std()
            )
        )

        feature_columns.append(
            rolling_std_column
        )

        # Difference from previous reading
        change_column = (
            f"{sensor}_change"
        )

        df[change_column] = (
            df.groupby("engine_id")[sensor]
            .diff()
        )

        feature_columns.append(
            change_column
        )

        # Rolling mean of change
        change_mean_column = (
            f"{sensor}_change_mean"
        )

        df[change_mean_column] = (
            df.groupby("engine_id")[change_column]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        feature_columns.append(
            change_mean_column
        )

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return df, feature_columns


def predict_rul(engine_id, sensor_values, cycle):
    """
    Predict RUL using the trained Bagging model.

    The current reading is appended to the historical
    readings of the selected engine so that the same
    temporal features used during training can be created.
    """

    model_path = os.path.join(
        BASE_DIR,
        "best_rul_model.pkl"
    )

    features_path = os.path.join(
        BASE_DIR,
        "model_features.pkl"
    )

    data_path = os.path.join(
        BASE_DIR,
        "nasa_sample.csv"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "best_rul_model.pkl was not found."
        )

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            "model_features.pkl was not found."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "nasa_sample.csv was not found."
        )

    model = joblib.load(model_path)

    feature_columns = joblib.load(features_path)

    dataset_engine_id = get_dataset_engine_id(engine_id)

    historical_data = pd.read_csv(data_path)

    historical_data["engine_id"] = pd.to_numeric(
        historical_data["engine_id"],
        errors="coerce"
    )

    historical_data = historical_data[
        historical_data["engine_id"] == dataset_engine_id
    ].copy()

    if historical_data.empty:
        raise ValueError(
            f"No historical data was found for engine "
            f"{engine_id} in nasa_sample.csv."
        )

    historical_data["_is_current"] = False

    current_row = {
        "engine_id": dataset_engine_id,
        "cycle": cycle,
        "_is_current": True
    }

    for sensor_name, sensor_value in zip(
        BASE_SENSOR_COLUMNS,
        sensor_values
    ):
        current_row[sensor_name] = float(sensor_value)

    current_row_df = pd.DataFrame([current_row])

    prediction_history = pd.concat(
        [
            historical_data,
            current_row_df
        ],
        ignore_index=True
    )

    prediction_history = prediction_history.sort_values(
        [
            "engine_id",
            "cycle",
            "_is_current"
        ]
    ).reset_index(drop=True)

    prediction_df, _ = create_prediction_features(
        prediction_history
    )

    current_rows = prediction_df[
        prediction_df["_is_current"] == True
    ]

    if current_rows.empty:
        raise ValueError(
            "The current prediction row could not be identified."
        )

    latest_row = current_rows.iloc[[-1]].copy()

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in latest_row.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing prediction features: "
            + ", ".join(missing_features)
        )

    X = latest_row[
        feature_columns
    ]

    predicted_rul = model.predict(X)[0]

    predicted_rul = max(
        0,
        float(predicted_rul)
    )

    return predicted_rul

def get_health_status(predicted_rul):
    """
    Convert predicted RUL into a maintenance health status.

    The thresholds are based on remaining useful life ranges
    used by the Raseen maintenance workflow.
    """

    if predicted_rul <= 25:
        return "CRITICAL_DEGRADATION"

    if predicted_rul <= 50:
        return "WARNING_DEGRADATION"

    return "NORMAL"

if __name__ == "__main__":

    run_ml_pipeline()