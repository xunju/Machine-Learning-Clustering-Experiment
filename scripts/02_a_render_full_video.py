import os
import argparse
import subprocess
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_cluster_info(frame_df, cluster_col="cluster_id"):
    """
    提取当前帧中每个聚类组的信息：
    - 成员 ID 集合
    - 中心点坐标
    """
    groups = {}

    for cluster_id, group_df in frame_df.groupby(cluster_col):
        if cluster_id == -1:
            continue

        ids = set(group_df["ID"].astype(int).tolist())
        cx = group_df["X"].mean()
        cy = group_df["Y"].mean()

        groups[int(cluster_id)] = {
            "ids": ids,
            "centroid": (float(cx), float(cy)),
            "size": len(group_df)
        }

    return groups


def jaccard_similarity(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def centroid_similarity(c1, c2, sigma=2.0):
    """
    中心点越接近，相似度越高。
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return math.exp(-dist / sigma)


def align_visual_group_ids(df, match_threshold=0.25):
    """
    对逐帧 cluster_id 进行跨帧对齐，生成 visual_group_id。

    注意：
    这里不是重新聚类，只是为了视频显示时颜色更稳定。
    """
    df = df.copy()
    df["visual_group_id"] = -1

    next_visual_id = 0
    prev_groups = {}
    prev_cluster_to_visual = {}

    time_steps = sorted(df["time_step"].unique().tolist())

    for time_step in time_steps:
        frame_mask = df["time_step"] == time_step
        frame_df = df[frame_mask].copy()

        current_groups = get_cluster_info(frame_df, cluster_col="cluster_id")
        current_cluster_to_visual = {}

        used_prev_visual_ids = set()

        # 为当前帧每个 cluster 找上一帧最匹配的 visual_group_id
        for cur_cluster_id, cur_info in current_groups.items():
            best_score = -1.0
            best_visual_id = None

            for prev_cluster_id, prev_info in prev_groups.items():
                prev_visual_id = prev_cluster_to_visual.get(prev_cluster_id)

                if prev_visual_id is None:
                    continue

                if prev_visual_id in used_prev_visual_ids:
                    continue

                id_score = jaccard_similarity(cur_info["ids"], prev_info["ids"])
                cen_score = centroid_similarity(
                    cur_info["centroid"],
                    prev_info["centroid"],
                    sigma=2.0
                )

                score = 0.7 * id_score + 0.3 * cen_score

                if score > best_score:
                    best_score = score
                    best_visual_id = prev_visual_id

            if best_score >= match_threshold and best_visual_id is not None:
                visual_id = best_visual_id
                used_prev_visual_ids.add(visual_id)
            else:
                visual_id = next_visual_id
                next_visual_id += 1

            current_cluster_to_visual[cur_cluster_id] = visual_id

        # 写回当前帧 visual_group_id
        for cluster_id, visual_id in current_cluster_to_visual.items():
            mask = frame_mask & (df["cluster_id"] == cluster_id)
            df.loc[mask, "visual_group_id"] = visual_id

        # 噪声点保持 -1
        noise_mask = frame_mask & (df["cluster_id"] == -1)
        df.loc[noise_mask, "visual_group_id"] = -1

        prev_groups = current_groups
        prev_cluster_to_visual = current_cluster_to_visual

    return df


def get_color_for_visual_id(visual_id):
    """
    为 visual_group_id 分配稳定颜色。
    使用 tab20 调色板循环。
    """
    if visual_id == -1:
        return "black"

    cmap = plt.get_cmap("tab20")
    return cmap(int(visual_id) % 20)


def render_one_frame(frame_df, method_name, frame_index, time_step, output_path):
    plt.figure(figsize=(10, 8))

    labels = sorted(frame_df["visual_group_id"].unique())

    for visual_id in labels:
        group_df = frame_df[frame_df["visual_group_id"] == visual_id]

        if visual_id == -1:
            plt.scatter(
                group_df["X"],
                group_df["Y"],
                marker="x",
                s=90,
                c="black",
                label="noise"
            )
        else:
            color = get_color_for_visual_id(visual_id)
            plt.scatter(
                group_df["X"],
                group_df["Y"],
                s=60,
                c=[color],
                label=f"group {visual_id}"
            )

        for _, row in group_df.iterrows():
            plt.text(
                row["X"] + 0.03,
                row["Y"] + 0.03,
                str(int(row["ID"])),
                fontsize=7
            )

    plt.title(f"{method_name} | frame_index={frame_index} | time_step={time_step}")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xlim(-1, 16)
    plt.ylim(-1, 15)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-file", type=str, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--match-threshold", type=float, default=0.25)
    args = parser.parse_args()

    if not os.path.exists(args.cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {args.cluster_file}")

    df = pd.read_csv(args.cluster_file)
    df["time_step"] = df["time_step"].astype(int)
    df["ID"] = df["ID"].astype(int)

    print("=" * 80)
    print("Aligning cluster labels for stable visualization")
    print("=" * 80)

    df = align_visual_group_ids(
        df,
        match_threshold=args.match_threshold
    )

    aligned_csv_path = os.path.join(
        args.output_root,
        args.method_name,
        f"{args.method_name}_aligned_visual_labels.csv"
    )
    ensure_dir(os.path.dirname(aligned_csv_path))
    df.to_csv(aligned_csv_path, index=False)

    frame_dir = os.path.join(args.output_root, args.method_name, "frames")
    video_dir = os.path.join(args.output_root, args.method_name, "video")
    ensure_dir(frame_dir)
    ensure_dir(video_dir)

    unique_time_steps = sorted(df["time_step"].unique().tolist())

    print("\n" + "=" * 80)
    print("Rendering all frames with stable colors")
    print("=" * 80)
    print("cluster_file:", args.cluster_file)
    print("method_name:", args.method_name)
    print("num_frames:", len(unique_time_steps))
    print("frame_dir:", frame_dir)
    print("aligned_csv_path:", aligned_csv_path)

    for frame_index, time_step in enumerate(unique_time_steps):
        frame_df = df[df["time_step"] == time_step].copy()
        output_path = os.path.join(frame_dir, f"frame_{frame_index:04d}.png")

        render_one_frame(
            frame_df=frame_df,
            method_name=args.method_name,
            frame_index=frame_index,
            time_step=time_step,
            output_path=output_path
        )

        if frame_index % 50 == 0 or frame_index == len(unique_time_steps) - 1:
            print(f"Rendered frame {frame_index + 1}/{len(unique_time_steps)}")

    video_path = os.path.join(video_dir, f"{args.method_name}.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]

    print("\nGenerating video with ffmpeg...")
    print(" ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)

    print("\nFinished.")
    print("frame_dir:", frame_dir)
    print("video_path:", video_path)
    print("aligned_csv_path:", aligned_csv_path)


if __name__ == "__main__":
    main()
