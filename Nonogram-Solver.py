import numpy as np
import functools
from collections import deque
import show  # 导入你的专用可视化模块

columns = [
        [7, 3, 6, 3, 8, 1], [3, 2, 4, 4, 3, 4, 1, 1, 1, 1, 1], [2, 3, 11, 1, 1, 1, 2], [1, 1, 1, 3, 1, 3, 6, 1, 1, 1, 6], [1, 1, 5, 1, 1, 1, 7, 1, 4, 1],
        [3, 2, 9, 7, 3, 2, 1, 5, 2], [2, 1, 4, 4, 6, 3, 1, 2], [2, 2, 5, 12, 4], [2, 3, 3, 6, 3, 4], [3, 3, 1, 1, 6, 2, 4],
        [4, 2, 1, 1, 4, 3, 1, 4], [8, 1, 3, 1, 1, 4, 4], [10, 9, 2, 4, 2], [10, 3, 1, 2, 5, 1, 2, 1, 2], [15, 1, 1, 4, 3, 2],
        [12, 3, 2, 1, 3, 6], [3, 4, 7, 3, 1, 2, 1, 4, 1], [2, 1, 3, 2, 8, 1, 1, 3, 1, 8], [4, 3, 12, 6, 7], [2, 3, 1, 16, 5],
        [9, 1, 2, 3, 4, 6, 3], [7, 1, 2, 5, 3, 6, 4, 1], [9, 1, 1, 5, 2, 1, 8], [1, 1, 4, 2, 4, 2, 7], [3, 3, 3, 3, 2, 4],
        [2, 2, 5, 4, 1], [10, 5, 1, 5], [1, 7, 4, 2, 5, 1, 1, 4, 1], [11, 2, 1, 2, 1, 3, 5, 1], [3, 3, 1, 3, 3, 1, 2, 1],
        [4, 2, 1, 2, 5, 1, 5], [3, 2, 3, 1, 3, 3, 5, 6], [1, 3, 14, 3, 2, 3], [3, 6, 1, 3, 4, 4, 1], [10, 7, 1, 1, 2, 1],
        [2, 1, 1, 6, 7, 3, 2, 2], [8, 1, 1, 5, 9, 1, 2, 1, 4], [7, 1, 4, 1, 6, 3, 10], [3, 1, 1, 3, 6, 9], [4, 1, 5, 3, 1, 1, 9, 1],
        [3, 1, 2, 3, 2, 4, 1, 3, 6], [2, 4, 2, 1, 6, 4, 2], [1, 2, 6, 2], [2, 3, 1, 1, 9, 3, 2, 2, 2], [3, 1, 11, 3, 1, 3],
        [1, 7, 8, 5, 5, 1, 1], [1, 1, 2, 7, 5, 3, 2, 3], [2, 3, 6, 1, 14, 3, 1], [2, 6, 10, 7, 12, 3], [1, 5, 9, 3, 13, 1, 4]
]
rows = [
        [1, 3, 4, 7, 4, 4, 2, 1], [4, 1, 4, 4, 4, 7, 1, 2], [1, 3, 1, 3, 1, 1, 1, 3, 4, 5], [3, 5, 4, 5, 1, 1, 1, 2, 1, 2], [6, 8, 3, 3, 1, 5, 3, 1],
        [2, 2, 8, 4, 3, 1, 3, 1, 7], [1, 6, 5, 3, 6, 1, 2], [2, 7, 10, 3, 2, 2, 1, 2], [2, 8, 14, 1, 1, 3], [1, 8, 4, 8, 1, 1, 7],
        [1, 5, 1, 3, 1, 4], [3, 4, 3, 5], [3, 3, 3, 5, 2], [1, 3, 3, 5, 2, 5, 4], [2, 3, 9, 2, 1, 6, 7],
        [3, 3, 1, 1, 1, 1, 3, 4, 6], [3, 2, 5, 5, 6, 10], [1, 2, 7, 6, 5, 1, 8], [2, 2, 5, 8, 7, 4, 2], [2, 3, 3, 2, 8, 3, 3],
        [2, 1, 3, 2, 2, 1, 1, 3, 2], [2, 3, 2, 3, 1, 1, 2, 2], [1, 4, 1, 4, 3, 3, 1, 2, 2], [1, 4, 13, 1, 1, 1, 1, 7], [5, 8, 5, 10, 2],
        [4, 1, 3, 8, 6, 3, 3, 5], [2, 2, 3, 2, 17, 5], [3, 1, 3, 1, 1, 5, 3, 4], [3, 3, 1, 5, 3, 5], [5, 8, 6, 4],
        [6, 1, 6, 6, 3], [4, 2, 5, 1, 4, 1, 1], [8, 1, 1, 4, 3], [5, 8, 4, 1, 3, 3], [14, 5, 1, 2, 1, 7],
        [3, 19, 1, 2, 1, 2, 6], [2, 3, 3, 4, 3, 5, 5], [1, 4, 2, 4, 3, 1, 1, 1, 5, 3], [4, 1, 2, 7, 9, 4, 3], [1, 1, 4, 3, 1, 3, 3, 2],
        [4, 4, 2, 3, 3, 1, 1, 3, 2], [3, 5, 2, 6, 4], [3, 4, 1, 19, 4], [4, 1, 9, 1, 1, 3, 3, 2], [2, 4, 16, 1, 1, 5],
        [4, 1, 9, 2, 1, 4, 1], [11, 7, 3, 1, 5, 2, 1], [1, 5, 1, 3, 1, 1, 1, 5, 4, 2], [1, 11, 2, 4, 2, 1, 2], [6, 9, 3, 7, 4]
]

# 初始化
size_r, size_c = len(columns), len(rows)
answer = np.zeros((size_c, size_r), dtype=np.uint8)


# --- 核心算法逻辑 ---

@functools.lru_cache(maxsize=200000)
def can_place(line, blocks):
    """使用动态规划检查当前行状态是否能容纳剩余的提示块"""
    if not blocks:
        return 1 not in line

    block = blocks[0]
    first_1 = line.index(1) if 1 in line else -1

    # 搜索范围优化：起始位置不能超过第一个已涂黑的格子
    max_start = len(line) - block
    if first_1 != -1:
        max_start = min(max_start, first_1)

    for i in range(max_start + 1):
        # 检查区域内是否有留白标记
        if 2 in line[i:i + block]:
            continue
        # 检查块后面是否紧跟黑块
        if i + block < len(line) and line[i + block] == 1:
            continue

        # 递归检查后续块（中间需隔开至少1格）
        if can_place(line[i + block + 1:], blocks[1:]):
            return True
    return False


def process_line(line_view, blocks):
    """利用假设检验法（Contradiction）更新单行状态"""
    changed_indices = []
    t_line = list(line_view)
    blocks_tuple = tuple(blocks)

    for i in range(len(line_view)):
        if line_view[i] == 0:
            # 假设填涂
            t_line[i] = 1
            can_1 = can_place(tuple(t_line), blocks_tuple)
            # 假设留白
            t_line[i] = 2
            can_2 = can_place(tuple(t_line), blocks_tuple)

            # 逻辑推导
            if can_1 and not can_2:
                line_view[i] = 1
                t_line[i] = 1
                changed_indices.append(i)
            elif can_2 and not can_1:
                line_view[i] = 2
                t_line[i] = 2
                changed_indices.append(i)
            else:
                t_line[i] = 0  # 无法确定，重置

    return changed_indices


def solve():
    queue = deque()
    in_queue = set()
    history = []  # 存储每一步的状态快照

    def add_to_queue(is_col, idx):
        if (is_col, idx) not in in_queue:
            queue.append((is_col, idx))
            in_queue.add((is_col, idx))

    for r in range(size_c): add_to_queue(0, r)
    for c in range(size_r): add_to_queue(1, c)

    # 初始状态记录
    history.append(answer.copy())

    while queue:
        is_col, idx = queue.popleft()
        in_queue.remove((is_col, idx))

        line_view = answer[idx, :] if is_col == 0 else answer[:, idx]
        blocks = rows[idx] if is_col == 0 else columns[idx]

        changed_indices = process_line(line_view, blocks)

        if changed_indices:
            # 只要有变化，就记录当前快照
            history.append(answer.copy())
            for ci in changed_indices:
                add_to_queue(1 - is_col, ci)

    print(f"✅ 推导完成，共记录 {len(history)} 个步骤。")
    # 调用交互式可视化
    show.draw_interactive(history, rows, columns)


if __name__ == '__main__':
    solve()