import pandas as pd

COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]
CV_RUNS = 3
# N_COMPONENTS = [2**i for i in range(7)]
RANDOM_STATE = 666
INIT_METHODS = ["k-means++", "random"]
# N_CLUSTERS = [2**i for i in range(7)]


class StrokeParams:
    data_train = "~/Projects/cs-7641-machine-learning/unsupervised_learning/data/stroke_train.csv"
    data_test = (
        "~/Projects/cs-7641-machine-learning/unsupervised_learning/data/stroke_test.csv"
    )
    target_column = "stroke"
    random_state = RANDOM_STATE
    cv_runs = CV_RUNS
    n_components = [2, 4, 6, 8, 10, 12]
    covariance_type = "full"
    n_clusters = [2, 4, 6, 8, 10, 12, 14, 16]
    init_methods = ["k-means++", "random"]
    cat_cols = cat_cols = [
        "gender",
        "age_range",
        "hypertension",
        "heart_disease",
        "ever_married",
        "bmi_range",
        "work_type_govt_job",
        "work_type_never_worked",
        "work_type_private",
        "work_type_self_employed",
        "work_type_children",
        "residence_type_rural",
        "residence_type_urban",
        "smoking_status_formerly_smoked",
        "smoking_status_never_smoked",
        "smoking_status_smokes",
    ]


class FetalHealthParams:
    data_train = "~/Projects/cs-7641-machine-learning/unsupervised_learning/data/fetal_health_train.csv"
    data_test = "~/Projects/cs-7641-machine-learning/unsupervised_learning/data/fetal_health_test.csv"
    target_column = "fetal_health"
    random_state = RANDOM_STATE
    cv_runs = CV_RUNS
    n_components = [2, 4, 6, 8, 12, 16]
    covariance_type = "full"
    n_clusters = [2, 4, 6, 8, 12, 16, 24, 32]
    init_methods = ["k-means++", "random"]
    cat_cols = cat_cols = [
        "histogram_number_of_peaks",
        "histogram_number_of_zeroes",
        "histogram_tendency",
    ]


def load_data(dataset_path, target_column):
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
