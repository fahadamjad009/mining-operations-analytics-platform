import pandas as pd
from .config import settings

NASA_DIR = settings.data_raw / "nasa_cmapss" / "CMAPSSData"

COLS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_train(dataset: str = "FD001") -> pd.DataFrame:
    path = NASA_DIR / f"train_{dataset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)

    # drop any trailing empty columns
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS

    # Remaining Useful Life (RUL)
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="unit", how="left")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])
    return df

def main():
    settings.data_processed.mkdir(parents=True, exist_ok=True)
    out_path = settings.data_processed / "nasa_fd001_train.parquet"

    df = load_train("FD001")
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print("Units:", df["unit"].nunique())
    print("Max cycle:", df["cycle"].max())
    print("Min/Max RUL:", int(df["rul"].min()), int(df["rul"].max()))

if __name__ == "__main__":
    main()
