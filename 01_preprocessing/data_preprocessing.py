import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'raw_data', 'Bank Marketing.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'processed_data')


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Завантажує вихідний CSV-файл із перевіркою існування."""
    if not os.path.exists(file_path):
        print(f"Помилка: файл не знайдено: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path)
    print(f"[1/4] Дані завантажено: {df.shape[0]} рядків, {df.shape[1]} стовпців.")
    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Виділяє матрицю ознак X та бінарну цільову змінну y (0/1)."""
    # 1 -> 0 (відмова), 2 -> 1 (підписка на депозит)
    y = LabelEncoder().fit_transform(df['Subscription'])

    X = df.drop(columns=['Subscription']).copy()
    # Індикатор попереднього контакту (зберігає семантику pdays != -1)
    X['Previously_Contacted'] = (X['Pdays'] != -1).astype(int)
    return X, y


def split_train_test(
    X: pd.DataFrame, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Виконує стратифікований поділ даних на навчання та тест для запобігання Data Leakage."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[2/4] Стратифікований поділ: Train={X_train.shape[0]}, Test={X_test.shape[0]}.")
    return X_train, X_test, y_train, y_test


def create_column_transformer() -> ColumnTransformer:
    """Створює конвеєр перетворення для числових, порядкових та номінальних ознак."""
    num_features = [
        'Age', 'Balance (euros)', 'Last Contact Day', 'Last Contact Duration',
        'Campaign', 'Pdays', 'Previous', 'Previously_Contacted'
    ]
    ordinal_features = ['Education']
    nominal_features = [
        'Job', 'Marital Status', 'Credit', 'Housing Loan',
        'Personal Loan', 'Contact', 'Last Contact Month', 'Poutcome'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    education_order = [['unknown', 'primary', 'secondary', 'tertiary']]
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_enc', OrdinalEncoder(categories=education_order, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('ord', ordinal_pipeline, ordinal_features),
            ('nom', nominal_pipeline, nominal_features)
        ],
        remainder='drop'
    )


def apply_transformations(
    preprocessor: ColumnTransformer, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Навчає трансформації ВИКЛЮЧНО на Train та перетворює Train і Test у DataFrame."""
    print("[3/4] Виконання трансформацій ознак (RobustScaler + Encoders)...")
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Отримання імен ознак після One-Hot кодування
    num_features = [
        'Age', 'Balance (euros)', 'Last Contact Day', 'Last Contact Duration',
        'Campaign', 'Pdays', 'Previous', 'Previously_Contacted'
    ]
    ordinal_features = ['Education']
    nominal_features = [
        'Job', 'Marital Status', 'Credit', 'Housing Loan',
        'Personal Loan', 'Contact', 'Last Contact Month', 'Poutcome'
    ]
    ohe_step = preprocessor.named_transformers_['nom'].named_steps['ohe']
    nom_encoded_names = list(ohe_step.get_feature_names_out(nominal_features))
    all_feature_names = num_features + ordinal_features + nom_encoded_names

    X_train_df = pd.DataFrame(X_train_trans, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=all_feature_names)
    return X_train_df, X_test_df


def save_processed_datasets(
    output_dir: str, 
    X_train_df: pd.DataFrame, 
    X_test_df: pd.DataFrame, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> None:
    """Зберігає готові 4 CSV-файли у цільову папку."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[4/4] Збереження 4 файлів у {output_dir}...")

    X_train_df.to_csv(os.path.join(output_dir, 'X_train_preprocessed.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_dir, 'X_test_preprocessed.csv'), index=False)
    pd.Series(y_train, name='Subscription').to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.Series(y_test, name='Subscription').to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("Обробка успішно завершена.")


def main():
    df = load_raw_data(INPUT_FILE)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = create_column_transformer()
    X_train_proc, X_test_proc = apply_transformations(preprocessor, X_train, X_test)

    save_processed_datasets(OUTPUT_DIR, X_train_proc, X_test_proc, y_train, y_test)


if __name__ == '__main__':
    main()
