"""重叠字幕择优保留。

在不移动原时间轴的前提下，从按输入顺序排列的字幕中选出一个子序列：

1. 任意两条**相邻被保留**条目必须满足
   ``前条结束 + 最小间隔 <= 后条开始``（首尾相接且最小间隔为 0 时允许）；
2. 每条字幕带正整数**优先级**（未设置为 1），在满足间隔约束且包含全部
   “必须保留”条目的可行子序列中，**最大化优先级总和**；
3. 优先级总和相同时，优先取**保留条数更多**的方案；条数也相同时，取保留
   条目**输入序号序列字典序更小**者（第一个差异处取序号更靠前的方案）；
4. “必须保留”条目之间彼此不兼容（按输入顺序相邻的两条必须保留条目满足
   不了间隔约束，中间条目无法弥合）时抛出 :class:`SelectionConflict`，
   并给出所有冲突的相邻必须保留序号对。

动态规划以“最后一条被保留的字幕”为状态：``nodes[i]`` 表示以第 i 条结尾、
且 i 之前的全部必须保留条目都已纳入的最优子序列。转移只从满足间隔约束、
且与 i 之间没有落下任何必须保留条目的状态接续。同分只在（总和、条数）
都相同时沿父指针回溯两条序列做字典序比较，避免为每个状态保存完整序列。
"""
from __future__ import annotations

from dataclasses import dataclass

from .srt import Cue


class SelectionError(ValueError):
    """择优保留参数不合法（序号越界、优先级非正、序号重复等）。"""


@dataclass(frozen=True)
class ConflictPair:
    """一对互不兼容的必须保留条目（均为从 1 开始的输入序号）。"""

    earlier_order: int
    later_order: int
    gap_ms: int  # 后条开始 − 前条结束（毫秒，可能为负）


class SelectionConflict(ValueError):
    """必须保留条目彼此不兼容，无法同时保留。"""

    def __init__(self, pairs: tuple[ConflictPair, ...], min_gap_ms: int):
        self.pairs = pairs
        self.min_gap_ms = min_gap_ms
        desc = "、".join(
            f"{p.earlier_order} 与 {p.later_order}"
            f"（前条结束后实际间隔 {p.gap_ms} ms）"
            for p in pairs
        )
        super().__init__(
            f"必须保留的字幕中有 {len(pairs)} 对相邻条目互不兼容：{desc}；"
            f"按最小间隔 {min_gap_ms} ms 无法同时保留，请调整必须保留序号或最小间隔。"
        )


@dataclass(frozen=True)
class SelectionPlan:
    kept_orders: tuple[int, ...]
    total_priority: int


@dataclass
class _Node:
    """以某条字幕结尾的最优子序列：优先级总和、条数与父状态下标。"""

    total: int
    length: int
    parent: int | None  # 前一条被保留字幕的 0 基下标；虚拟起点为 None


def select_cues(
    cues: list[Cue],
    priorities: list[tuple[int, int]],
    required_orders: list[int],
    min_gap_ms: int,
) -> SelectionPlan:
    """计算满足间隔与必须保留约束的最优保留子序列。

    ``priorities`` 为 ``(输入序号 1 基, 正整数优先级)`` 列表（未列出的
    条目优先级为 1）；``required_orders`` 为必须保留的输入序号；
    ``min_gap_ms`` 为非负最小间隔毫秒数。参数非法时抛出
    :class:`SelectionError`，必须保留条目彼此冲突时抛出
    :class:`SelectionConflict`。
    """
    n = len(cues)
    if min_gap_ms < 0:
        raise SelectionError("最小间隔毫秒数必须是非负整数。")

    weights = [1] * n
    seen_priority: set[int] = set()
    for order, value in priorities:
        if not isinstance(order, int) or not 1 <= order <= n:
            raise SelectionError(
                f"优先级设置中的字幕序号 {order} 超出范围：当前共 {n} 条字幕，"
                "序号应为 1 到该条数之间的正整数。"
            )
        if not isinstance(value, int) or value < 1:
            raise SelectionError(
                f"序号 {order} 的优先级 {value} 不合法：优先级必须是正整数，"
                "未设置时默认为 1。"
            )
        if order in seen_priority:
            raise SelectionError(f"序号 {order} 的优先级被重复设置，请只保留一条。")
        seen_priority.add(order)
        weights[order - 1] = value

    required: set[int] = set()
    for order in required_orders:
        if not isinstance(order, int) or not 1 <= order <= n:
            raise SelectionError(
                f"必须保留的字幕序号 {order} 超出范围：当前共 {n} 条字幕，"
                "序号应为 1 到该条数之间的正整数。"
            )
        if order in required:
            raise SelectionError(f"必须保留的序号 {order} 重复填写，请去重后重试。")
        required.add(order)

    starts = [cue.start for cue in cues]
    ends = [cue.end for cue in cues]

    # 必须保留条目的 0 基位置（升序）；相邻必须保留条目不兼容即整体无解：
    # 二者之间无论选哪些条目，它们在保留序列里仍保持相邻（中间条目无法
    # 同时与两边满足间隔，否则两条必须保留条目自身也已兼容）。
    required_positions = sorted(order - 1 for order in required)
    required_position_set = set(required_positions)
    conflicts = tuple(
        ConflictPair(
            earlier_order=a + 1,
            later_order=b + 1,
            gap_ms=starts[b] - ends[a],
        )
        for a, b in zip(required_positions, required_positions[1:])
        if ends[a] + min_gap_ms > starts[b]
    )
    if conflicts:
        raise SelectionConflict(conflicts, min_gap_ms)

    # pref[k] = 下标 [0, k) 内必须保留条目的数量
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + (1 if i in required_position_set else 0)

    nodes: list[_Node | None] = [None] * n

    def parent_chain(parent: int | None) -> tuple[int, ...]:
        """重建父状态的下标序列（不含当前结尾条目），供同分字典序比较。"""
        chain: list[int] = []
        cur = parent
        while cur is not None:
            chain.append(cur)
            cur = nodes[cur].parent  # type: ignore[union-attr]
        chain.reverse()
        return tuple(chain)

    def prefer(
        cand_total: int,
        cand_length: int,
        cand_parent: int | None,
        best: _Node | None,
    ) -> bool:
        """优先级总和大 → 条数多 → 输入序号序列字典序小。"""
        if best is None:
            return True
        if cand_total != best.total:
            return cand_total > best.total
        if cand_length != best.length:
            return cand_length > best.length
        # 两条序列同长且都以当前条目结尾，比较到当前条目之前的父链即可
        return parent_chain(cand_parent) < parent_chain(best.parent)

    for i in range(n):
        best: _Node | None = None

        # 虚拟起点：i 之前没有任何必须保留条目时，子序列可直接从 i 开始
        if pref[i] == 0:
            best = _Node(total=weights[i], length=1, parent=None)

        for j in range(i):
            node = nodes[j]
            if node is None:
                continue
            if ends[j] + min_gap_ms > starts[i]:
                continue
            # 区间 (j, i) 内落下必须保留条目则不可接续（那些条目必须在链上）
            if pref[i] - pref[j + 1] != 0:
                continue
            cand_total = node.total + weights[i]
            cand_length = node.length + 1
            if prefer(cand_total, cand_length, j, best):
                best = _Node(total=cand_total, length=cand_length, parent=j)

        nodes[i] = best

    # 收尾：i 之后不得还有必须保留条目；没有必须保留条目时全片自由选
    ending: _Node | None = None
    ending_at = -1
    for i in range(n):
        node = nodes[i]
        if node is not None and pref[n] - pref[i + 1] == 0 and prefer(
            node.total, node.length, node.parent, ending
        ):
            ending = node
            ending_at = i

    assert ending is not None, "必须保留约束可行性已由冲突检查保证。"

    kept: list[int] = []
    cur: int | None = ending_at
    while cur is not None:
        kept.append(cur)
        cur = nodes[cur].parent  # type: ignore[union-attr,index]
    kept.reverse()

    return SelectionPlan(
        kept_orders=tuple(i + 1 for i in kept),
        total_priority=ending.total,
    )
