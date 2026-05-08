import numpy as np

def add_motion_features(df, window_size=5, momentum_alpha=0.7):
    """
    计算瞬时速度、滑动窗口速度、指数动量和运动方向特征。
    """
    df = df.copy()
    df = df.sort_values(["ID", "time_step"]).reset_index(drop=True)

    # 瞬时速度
    df["prev_X"] = df.groupby("ID")["X"].shift(1)
    df["prev_Y"] = df.groupby("ID")["Y"].shift(1)
    df["prev_time"] = df.groupby("ID")["time_step"].shift(1)

    dt = df["time_step"] - df["prev_time"]

    df["vx"] = (df["X"] - df["prev_X"]) / dt
    df["vy"] = (df["Y"] - df["prev_Y"]) / dt
    df["vx"] = df["vx"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vy"] = df["vy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 滑动窗口速度
    df["win_X"] = df.groupby("ID")["X"].shift(window_size)
    df["win_Y"] = df.groupby("ID")["Y"].shift(window_size)
    df["win_time"] = df.groupby("ID")["time_step"].shift(window_size)

    dt_win = df["time_step"] - df["win_time"]
    df["window_vx"] = (df["X"] - df["win_X"]) / dt_win
    df["window_vy"] = (df["Y"] - df["win_Y"]) / dt_win
    df["window_vx"] = df["window_vx"].replace([np.inf, -np.inf], np.nan)
    df["window_vy"] = df["window_vy"].replace([np.inf, -np.inf], np.nan)
    df["window_vx"] = df["window_vx"].fillna(df["vx"]).fillna(0.0)
    df["window_vy"] = df["window_vy"].fillna(df["vy"]).fillna(0.0)

    # 指数动量
    momentum_vx, momentum_vy = [], []
    for _, group in df.groupby("ID", sort=False):
        mx_prev, my_prev = 0.0, 0.0
        for _, row in group.iterrows():
            mx = momentum_alpha * mx_prev + (1 - momentum_alpha) * row["vx"]
            my = momentum_alpha * my_prev + (1 - momentum_alpha) * row["vy"]
            momentum_vx.append(mx)
            momentum_vy.append(my)
            mx_prev, my_prev = mx, my

    df["momentum_vx"] = momentum_vx
    df["momentum_vy"] = momentum_vy

    # 运动方向近似朝向
    speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    df["cos_theta"] = df["vx"] / (speed + 1e-8)
    df["sin_theta"] = df["vy"] / (speed + 1e-8)

    static = speed < 1e-8
    df.loc[static, ["cos_theta", "sin_theta"]] = 0.0

    return df
