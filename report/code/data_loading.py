import pandas as pd

def load_students003(data_path):
    """
    读取 students003 数据，并统一字段名称。
    每条记录包含 time_step、ID、X、Y 四个字段。
    """
    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=["time_step", "ID", "X", "Y"]
    )

    df["time_step"] = df["time_step"].astype(int)
    df["ID"] = df["ID"].astype(int)

    df = df.sort_values(["time_step", "ID"]).reset_index(drop=True)
    return df


def iter_frames(df):
    """
    按 time_step 逐帧返回当前帧中的所有行人点。
    """
    for time_step, frame_df in df.groupby("time_step"):
        yield time_step, frame_df.copy()
