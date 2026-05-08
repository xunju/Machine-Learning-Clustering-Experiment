import os
import numpy as np
import pandas as pd

DATA_PATH = "/root/autodl-tmp/TrajectoryData_students003/students003.txt"
OUT_DIR = "outputs/inspect"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("Stage 1: Inspect students003 dataset")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # 1. 读取数据
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=["time_step", "ID", "X", "Y"]
    )

    print("\n[1] Basic data preview")
    print(df.head(20))

    print("\n[2] Shape and data types")
    print("Shape:", df.shape)
    print(df.dtypes)

    # 2. 基本字段检查
    required_cols = ["time_step", "ID", "X", "Y"]
    print("\n[3] Required columns")
    print("Columns:", df.columns.tolist())
    print("Required columns exist:", all(col in df.columns for col in required_cols))

    # 3. 缺失值检查
    print("\n[4] Missing values")
    print(df.isna().sum())

    # 4. 转换 time_step 和 ID
    df["time_step"] = df["time_step"].astype(int)
    df["ID"] = df["ID"].astype(int)

    # 5. 基本范围统计
    print("\n[5] Value ranges")
    print("time_step min:", df["time_step"].min())
    print("time_step max:", df["time_step"].max())
    print("number of frames:", df["time_step"].nunique())

    print("ID min:", df["ID"].min())
    print("ID max:", df["ID"].max())
    print("number of unique IDs:", df["ID"].nunique())

    print("X min:", df["X"].min())
    print("X max:", df["X"].max())
    print("Y min:", df["Y"].min())
    print("Y max:", df["Y"].max())

    # 6. 时间步间隔检查
    time_steps = np.array(sorted(df["time_step"].unique()))
    intervals = np.diff(time_steps)
    unique_intervals = sorted(set(intervals.tolist()))

    print("\n[6] Time step interval check")
    print("First 10 time steps:", time_steps[:10].tolist())
    print("Last 10 time steps:", time_steps[-10:].tolist())
    print("Unique intervals:", unique_intervals)

    expected_steps = np.arange(time_steps.min(), time_steps.max() + unique_intervals[0], unique_intervals[0])
    missing_steps = sorted(set(expected_steps.tolist()) - set(time_steps.tolist()))

    print("Missing time steps:", len(missing_steps))
    if len(missing_steps) > 0:
        print("First missing time steps:", missing_steps[:20])

    # 7. 每帧人数检查
    frame_counts = df.groupby("time_step").size()

    print("\n[7] Pedestrian count per frame")
    print("min pedestrians per frame:", frame_counts.min())
    print("max pedestrians per frame:", frame_counts.max())
    print("mean pedestrians per frame:", frame_counts.mean())
    print("median pedestrians per frame:", frame_counts.median())
    print("frames with less than 2 pedestrians:", int((frame_counts < 2).sum()))

    print("\nFirst 10 frame counts:")
    print(frame_counts.head(10))

    print("\nLast 10 frame counts:")
    print(frame_counts.tail(10))

    # 8. 同一帧内是否存在重复 ID
    duplicate_records = df.duplicated(subset=["time_step", "ID"]).sum()

    print("\n[8] Duplicate records")
    print("Duplicated (time_step, ID) records:", int(duplicate_records))

    # 9. 坐标异常的简单检查
    q_low = df[["X", "Y"]].quantile(0.01)
    q_high = df[["X", "Y"]].quantile(0.99)

    print("\n[9] Coordinate quantiles")
    print("1% quantile:")
    print(q_low)
    print("99% quantile:")
    print(q_high)

    # 10. 保存结果
    df.to_csv(os.path.join(OUT_DIR, "students003_clean.csv"), index=False)
    df.head(100).to_csv(os.path.join(OUT_DIR, "students003_head100.csv"), index=False)
    frame_counts.to_csv(os.path.join(OUT_DIR, "frame_counts.csv"), header=["num_pedestrians"])

    summary = {
        "num_records": len(df),
        "num_frames": df["time_step"].nunique(),
        "num_unique_ids": df["ID"].nunique(),
        "time_step_min": df["time_step"].min(),
        "time_step_max": df["time_step"].max(),
        "min_pedestrians_per_frame": int(frame_counts.min()),
        "max_pedestrians_per_frame": int(frame_counts.max()),
        "mean_pedestrians_per_frame": float(frame_counts.mean()),
        "frames_with_less_than_2_pedestrians": int((frame_counts < 2).sum()),
        "duplicated_time_id_records": int(duplicate_records),
        "missing_time_steps": int(len(missing_steps)),
        "x_min": float(df["X"].min()),
        "x_max": float(df["X"].max()),
        "y_min": float(df["Y"].min()),
        "y_max": float(df["Y"].max()),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(OUT_DIR, "dataset_summary.csv"), index=False)

    print("\n[10] Saved files")
    print(os.path.join(OUT_DIR, "students003_clean.csv"))
    print(os.path.join(OUT_DIR, "students003_head100.csv"))
    print(os.path.join(OUT_DIR, "frame_counts.csv"))
    print(os.path.join(OUT_DIR, "dataset_summary.csv"))

    print("\nStage 1 inspection finished.")

if __name__ == "__main__":
    main()
