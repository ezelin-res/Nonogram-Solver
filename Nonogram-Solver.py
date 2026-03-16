import numpy as np
import functools
from collections import deque
import show  # 导入你的专用可视化模块

columns = [
        [4], [2], [1, 1], [1, 1], [2, 1],
        [2, 1, 2], [1, 4, 1, 1], [4, 5], [3, 5], [3, 5]
    ]
rows = [
        [1, 5], [1, 3], [4], [2], [3, 1],
        [6], [3, 3], [2, 4], [1, 1, 3], [1, 6]
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