import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def draw_interactive(history, row_clues, col_clues):
    rows, cols = history[0].shape

    # 调整画布比例，底部留出更多空间给滑块和说明文字
    fig, ax = plt.subplots(figsize=(12, 11))
    plt.subplots_adjust(bottom=0.2, left=0.15, top=0.8, right=0.95)

    def get_display_mat(step_idx):
        curr = history[step_idx].copy().astype(float)
        disp = np.full((rows, cols, 3), 1.0)
        prev = history[step_idx - 1] if step_idx > 0 else None

        for r in range(rows):
            for c in range(cols):
                val = curr[r, c]
                if val == 0:  # 未知
                    disp[r, c] = [0.94, 0.94, 0.94]
                elif val == 1:  # 填涂
                    if prev is not None and prev[r, c] == 0:
                        disp[r, c] = [1.0, 0.0, 0.0]  # 新确定的填涂：红色
                    else:
                        disp[r, c] = [0.1, 0.1, 0.1]  # 已确定的填涂：深灰黑
                elif val == 2:  # 留白
                    if prev is not None and prev[r, c] == 0:
                        disp[r, c] = [1.0, 0.7, 0.7]  # 新确定的留白：淡红
                    else:
                        disp[r, c] = [1.0, 1.0, 1.0]  # 已确定的留白：白色
        return disp

    img = ax.imshow(get_display_mat(0), aspect='equal', interpolation='nearest')

    # --- 1. 网格线美化系统 ---
    # 设置 1x1 细线 (Minor Ticks)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='#DDDDDD', linestyle='-', linewidth=0.5)

    # 设置 5x5 粗线 (Major Ticks)
    ax.set_xticks(np.arange(-0.5, cols, 5))
    ax.set_yticks(np.arange(-0.5, rows, 5))
    ax.grid(which='major', color='#555555', linestyle='-', linewidth=1.5)

    # 隐藏刻度数字
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which='both', length=0)

    # --- 2. 标注题目 (Clues) ---
    for i, clue in enumerate(row_clues):
        ax.text(-0.8, i, " ".join(map(str, clue)), ha='right', va='center',
                fontsize=min(8, 400 / rows), color='#333', fontweight='bold')
    for i, clue in enumerate(col_clues):
        ax.text(i, -0.8, "\n".join(map(str, clue)), ha='center', va='bottom',
                fontsize=min(8, 400 / cols), color='#333', fontweight='bold')

    # --- 3. 交互组件与动态文字 ---
    # 滑块位置
    ax_slider = plt.axes([0.2, 0.08, 0.4, 0.03])
    slider = Slider(ax_slider, 'Progress ', 0, len(history) - 1, valinit=0, valfmt='%d')

    # 在滑块右侧创建一个用于显示状态的文本对象
    # 使用 fig.text 绝对定位，避免受坐标轴缩放影响
    info_text = fig.text(0.65, 0.075, '', fontsize=11, color='#D32F2F',
                         fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    def update_info(idx):
        info_text.set_text(f"Step: {idx} / {len(history) - 1}\nStatus: Red = Newly Found")

    def update(val):
        idx = int(slider.val)
        img.set_data(get_display_mat(idx))
        update_info(idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update_info(0)  # 初始化文字

    plt.show()