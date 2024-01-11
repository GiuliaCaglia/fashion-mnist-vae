from io import BytesIO
from typing import Tuple

import pandas as pd
from minio import Minio


def get_client() -> Minio:
    return Minio(
        endpoint="minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )


CLIENT = get_client()
TRAIN_KEY = "data/train.parquet.gzip"
TEST_KEY = "data/test.parquet.gzip"


def get_minio_buffer(
    key: str, bucket: str = "mnist-fashion", client: Minio = CLIENT
) -> BytesIO:
    response = client.get_object(bucket, key)
    buffer = BytesIO(response.data)
    return buffer


def read_parquet(
    key: str, bucket: str = "mnist-fashion", client: Minio = CLIENT
) -> pd.DataFrame:
    buffer = get_minio_buffer(key=key, bucket=bucket, client=client)
    return pd.read_parquet(buffer)


def read_csv(
    key: str, bucket: str = "mnist-fashion", client: Minio = CLIENT
) -> pd.DataFrame:
    buffer = get_minio_buffer(key=key, bucket=bucket, client=client)
    return pd.read_csv(buffer)


def to_parquet(
    df: pd.DataFrame, key: str, bucket: str = "mnist-fashion", client: Minio = CLIENT
):
    buffer = BytesIO(df.to_parquet(index=False, compression="gzip"))
    buffer_len = len(buffer.read())
    buffer.seek(0)
    client.put_object(bucket, key, buffer, buffer_len)


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = read_parquet(TRAIN_KEY)
    test = read_parquet(TEST_KEY)
    X_train, y_train = train.iloc[:, 1:].to_numpy(), train.iloc[:, 0].to_numpy()
    X_test, y_test = test.iloc[:, 1:].to_numpy(), test.iloc[:, 0].to_numpy()
    return X_train, y_train, X_test, y_test
