import os
import subprocess
import matplotlib.pyplot as plt

def render_frame(frame_df, output_path, title):
    """
    绘制单帧聚类结果。
    不同颜色表示不同组，黑色叉号表示噪声点。
    """
    plt.figure(figsize=(10, 8))

    for group_id, group_df in frame_df.groupby("visual_group_id"):
        if group_id == -1:
            plt.scatter(group_df["X"], group_df["Y"],
                        marker="x", s=90, c="black", label="noise")
        else:
            plt.scatter(group_df["X"], group_df["Y"],
                        s=60, label=f"group {group_id}")

        for _, row in group_df.iterrows():
            plt.text(row["X"] + 0.03, row["Y"] + 0.03,
                     str(int(row["ID"])), fontsize=7)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-1, 16)
    plt.ylim(-1, 15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def frames_to_video(frame_dir, video_path, fps=10):
    """
    使用 ffmpeg 将逐帧 PNG 合成为 MP4 视频。
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, check=True)
