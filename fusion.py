"""fusion.Store() —— 多来源区间增量融合存储（仅依赖 Python 标准库）。

数据模型
========
* 三个固定来源 A > B > C，每个来源各自是一条整数时刻轴上的阶梯函数，
  用随机化 Treap（平衡 BST）维护互不重叠、相邻同值已合并的段集合。
* 融合视图按每个时刻取“仍然有效的最高优先级来源”的值；某来源 v=None
  的改动即撤回该来源在 [a,b) 上的值。融合结果同样以 Treap 维护。
* 日汇总字典 daily_totals[d] = (值×秒数之和, 覆盖秒数)，仅在 apply 时
  对触及的天做旧贡献减回 / 新贡献加入，绝不重建无关日期。

复杂度（每次 apply 的单项）
==========================
只访问与 [a,b) 相交的节点：Treap 分裂合并 O(log N + k)，k 为触及段数；
日汇总最多处理 8 天。不逐秒展开、不扫描无关历史。
"""

from __future__ import annotations

import random

# 来源固定优先级：下标越小优先级越高（A > B > C）
SOURCES = ("A", "B", "C")
_PRIORITY = {s: i for i, s in enumerate(SOURCES)}
SECONDS_PER_DAY = 86400
MAX_BATCH_ITEMS = 20
MAX_ITEM_SPAN_SECONDS = 7 * SECONDS_PER_DAY  # 每项 ≤ 7 天


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def _is_int(x):
    """非布尔整数（bool 是 int 的子类，需显式排除）。"""
    return isinstance(x, int) and not isinstance(x, bool)


def _day_floor(t):
    """时刻 t 所在日号，负数采用向下取整（与 // 一致）。"""
    return t // SECONDS_PER_DAY


# --------------------------------------------------------------------------
# 随机化 Treap：键为段起点 key，附加 length（段长）与 payload。
# 同层段互不重叠且按 key 排序，因此对键轴的 split/merge 即对区间轴操作。
# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("key", "length", "payload", "prio", "left", "right")

    def __init__(self, key, length, payload, prio):
        self.key = key
        self.length = length
        self.payload = payload
        self.prio = prio
        self.left = None
        self.right = None


def _split(root, key):
    """按键分裂：左树 key < key，右树 key >= key。"""
    if root is None:
        return None, None
    if root.key < key:
        left, right = _split(root.right, key)
        root.right = left
        return root, right
    else:
        left, right = _split(root.left, key)
        root.left = right
        return left, root


def _merge(left, right):
    """合并两棵键域不相交的 Treap（左树所有键 < 右树所有键）。"""
    if left is None:
        return right
    if right is None:
        return left
    if left.prio > right.prio:
        left.right = _merge(left.right, right)
        return left
    else:
        right.left = _merge(left, right.left)
        return right


def _rightmost(root):
    while root.right is not None:
        root = root.right
    return root


def _iter_inorder(root):
    """中序遍历，产出 (key, length, payload)。"""
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        yield cur.key, cur.length, cur.payload
        cur = cur.right


# --------------------------------------------------------------------------
# 单个来源层：payload = 该来源在该段上的整数值（不含 None）
# --------------------------------------------------------------------------
class _Layer:
    def __init__(self, rng):
        self._root = None
        self._rng = rng
        self.size = 0  # 当前段数（历史统计用）

    def assign(self, a, b, v):
        """在 [a,b) 上赋值 v（非 None）或撤回（v=None）。

        只分裂出与 [a,b) 相交的节点；全程读取边界信息而不就地修改可能
        被共享的节点，最终在纯列表层面合并相邻同值段并用新节点重建
        受影响子树。跨越 b 的幸存片段可能来自左跨越段（整窗覆盖）或
        中段最右段。
        """
        L, rest = _split(self._root, a)
        M, R = _split(rest, b)

        removed = 0
        right_frag = None
        left_keep = None  # (key, length, payload)，a 左侧保留部分

        # 左侧跨越段（L 最右段，键 < a 且终点 > a）
        if L is not None:
            ln = _rightmost(L)
            end = ln.key + ln.length
            if end > a:
                left_keep = (ln.key, a - ln.key, ln.payload)
                if end > b:  # 整窗覆盖：[b,end) 作为幸存片段
                    right_frag = (b, end - b, ln.payload)

        # 中段全部移除；最右段若越过 b 且尚无幸存片段则剥离
        if M is not None:
            removed = sum(1 for _ in _iter_inorder(M))
            rn = _rightmost(M)
            if rn.key + rn.length > b and right_frag is None:
                right_frag = (b, rn.key + rn.length - b, rn.payload)

        # 汇总最终落位段（左保留 + 窗口新段 + 右幸存），再做同值合并
        final = []
        if left_keep is not None:
            final.append(list(left_keep))
        if v is not None:
            final.append([a, b - a, v])
        if right_frag is not None:
            final.append(list(right_frag))
        final.sort(key=lambda x: x[0])

        merged = []
        for seg in final:
            if merged and merged[-1][0] + merged[-1][1] == seg[0] \
                    and merged[-1][2] == seg[2]:
                merged[-1][1] += seg[1]
            else:
                merged.append(seg)

        # 从 L 摘除已由 left_keep 表达的旧末段（按键分裂，不共享节点）
        left_dropped = 0
        if L is not None:
            llast = _rightmost(L)
            if llast.key + llast.length > a:
                L, _dropped = _split(L, llast.key)
                left_dropped = 1

        left_built = mid_built = right_built = None
        for key, length, payload in merged:
            node = _Node(key, length, payload, self._rng.random())
            if key < a:
                left_built = _merge(left_built, node)
            elif key >= b:
                right_built = _merge(right_built, node)
            else:
                mid_built = _merge(mid_built, node)

        # 记账：移除 = 中段节点 + 被摘除的 L 末段；新增 = 重建节点数
        self.size += len(merged) - (removed + left_dropped)
        new_left = _merge(L, left_built)
        new_right = _merge(right_built, R)
        self._root = _merge(_merge(new_left, mid_built), new_right)

    def range_pieces(self, a, b):
        """返回与 [a,b) 相交的裁剪段 (key, length, payload) 列表。

        Treap 按键分裂：跨越 a 的段在左树（取其最右节点）；中段含键
        ∈ [a,b) 的全部段，其中最右段可能越过 b，统一裁剪到 b。
        查询不改变节点语义。
        """
        L, rest = _split(self._root, a)
        M, R = _split(rest, b)
        out = []
        if L is not None:
            ln = _rightmost(L)
            end = ln.key + ln.length
            if end > a:
                # L 段可能同时越过 a 与 b，裁剪到 [a,b)
                out.append((a, min(end, b) - a, ln.payload))
        if M is not None:
            for key, length, payload in _iter_inorder(M):
                lo = max(key, a)
                hi = min(key + length, b)
                if hi > lo:
                    out.append((lo, hi - lo, payload))
        out.sort(key=lambda x: x[0])
        self._root = _merge(_merge(L, M), R)
        return out


# --------------------------------------------------------------------------
# 融合层：payload = (来源, 值)
# --------------------------------------------------------------------------
class _FusedTree:
    def __init__(self, rng):
        self._root = None
        self._rng = rng
        self.size = 0

    def replace_range(self, a, b, pieces):
        """用窗口内规范新段 pieces 替换融合视图的 [a,b)。

        pieces 只允许覆盖 [a,b)。流程：
        1) 分裂出 L（键<a）、M（键∈[a,b)）、R（键≥b），记录窗口内旧段；
        2) 找出左跨越段（L 末段越过 a）与右幸存片段（越过 b 的部分），
           这两个边界只读取信息、不就地修改共享节点；
        3) 在纯列表层面把边界同值延续吸收进 pieces；
        4) 用全新节点重建受影响子树并一次 merge，杜绝共享引用。
        返回 (old_pieces, inserted, removed)，old_pieces 按起点排序，
        供日汇总差分。
        """
        L, rest = _split(self._root, a)
        M, R = _split(rest, b)

        old_pieces = []
        removed = 0

        # ---- 左边界：L 末段是否越过 a ----
        left_keep = None   # (key, length, payload) 截断后保留在 a 左侧
        left_span = None   # 该段在窗口 [a,b) 内的部分 (lo, length, payload)
        right_frag = None  # (b, length, payload) 越过 b 的幸存片段
        if L is not None:
            ln = _rightmost(L)
            end = ln.key + ln.length
            if end > a:
                left_keep = (ln.key, a - ln.key, ln.payload)
                left_span = (a, min(end, b) - a, ln.payload)
                old_pieces.append(left_span)
                if end > b:  # 整窗覆盖：[b,end) 必须作为幸存片段
                    right_frag = (b, end - b, ln.payload)

        # ---- 中段旧贡献 + 右幸存片段 ----
        if M is not None:
            for key, length, payload in _iter_inorder(M):
                lo = max(key, a)
                hi = min(key + length, b)
                if hi > lo:
                    old_pieces.append((lo, hi - lo, payload))
            rn = _rightmost(M)
            rn_end = rn.key + rn.length
            if rn_end > b and right_frag is None:
                right_frag = (b, rn_end - b, rn.payload)
            removed = sum(1 for _ in _iter_inorder(M))
        old_pieces.sort(key=lambda x: x[0])

        # ---- 纯列表层面构造最终落位的全部段 ----
        final = []
        if left_keep is not None:
            final.append(list(left_keep))
        for key, length, payload in pieces:
            final.append([key, length, payload])
        if right_frag is not None:
            final.append(list(right_frag))
        final.sort(key=lambda x: x[0])

        # 相邻同源同值合并（只可能发生在窗口两侧的至多两个拼接点）
        merged = []
        for seg in final:
            if merged and merged[-1][0] + merged[-1][1] == seg[0] \
                    and merged[-1][2] == seg[2]:
                merged[-1][1] += seg[1]
            else:
                merged.append(seg)

        # ---- 重建：L 末段已由 left_keep 表达，从 L 摘除它 ----
        left_dropped = 0
        if L is not None:
            llast = _rightmost(L)
            if llast.key + llast.length > a:
                L, _dropped = _split(L, llast.key)
                left_dropped = 1

        # merged 按键拆成左部(<a)、窗口部([a,b))、右部(≥b)
        mid_built = None
        right_built = None
        left_built = None
        for key, length, payload in merged:
            node = _Node(key, length, payload, self._rng.random())
            if key < a:
                left_built = _merge(left_built, node)
            elif key >= b:
                right_built = _merge(right_built, node)
            else:
                mid_built = _merge(mid_built, node)

        new_left = _merge(L, left_built)
        new_right = _merge(right_built, R)
        # 记账：移除 = 中段节点 + 被摘除的 L 末段；新增 = 重建节点数
        inserted = len(merged)
        self.size += inserted - (removed + left_dropped)
        self._root = _merge(_merge(new_left, mid_built), new_right)
        return old_pieces, inserted, removed

    def range_pieces(self, a, b):
        """返回与 [a,b) 相交的裁剪段 (key, length, payload) 列表。

        Treap 按键分裂：跨越 a 的段在左树（取其最右节点）；中段含键
        ∈ [a,b) 的全部段，其中最右段可能越过 b，统一裁剪到 b。
        查询不改变节点语义。
        """
        L, rest = _split(self._root, a)
        M, R = _split(rest, b)
        out = []
        if L is not None:
            ln = _rightmost(L)
            end = ln.key + ln.length
            if end > a:
                out.append((a, min(end, b) - a, ln.payload))
        if M is not None:
            for key, length, payload in _iter_inorder(M):
                lo = max(key, a)
                hi = min(key + length, b)
                if hi > lo:
                    out.append((lo, hi - lo, payload))
        out.sort(key=lambda x: x[0])
        self._root = _merge(_merge(L, M), R)
        return out

    def all_pieces(self):
        return list(_iter_inorder(self._root))


# --------------------------------------------------------------------------
# 对外的 Store
# --------------------------------------------------------------------------
class Store:
    def __init__(self):
        rng = random.Random(0x5EED)  # 固定种子，Treap 形态确定、可复现
        self._layers = {s: _Layer(rng) for s in SOURCES}
        self._fused = _FusedTree(rng)
        # d -> [加权和, 覆盖秒数]
        self._daily = {}
        # 最近一次 apply 的局部工作量（演示/观测用，不影响语义）
        self.last_apply = None

    # ---------------- 校验 ----------------
    @staticmethod
    def _validate(changes):
        if not isinstance(changes, (list, tuple)):
            raise ValueError("changes 必须是 (s,a,b,v) 列表")
        if len(changes) > MAX_BATCH_ITEMS:
            raise ValueError("每批最多 %d 项" % MAX_BATCH_ITEMS)
        norm = []
        for ch in changes:
            if not isinstance(ch, (list, tuple)) or len(ch) != 4:
                raise ValueError("每项必须是 (s,a,b,v) 四元组")
            s, a, b, v = ch
            if s not in _PRIORITY:
                raise ValueError("来源必须是 A/B/C，收到 %r" % (s,))
            if not (_is_int(a) and _is_int(b)):
                raise ValueError("a、b 必须为非布尔整数")
            if a >= b:
                raise ValueError("必须满足 a < b")
            if b - a > MAX_ITEM_SPAN_SECONDS:
                raise ValueError("单项跨度不得超过 7 天")
            if v is not None and not _is_int(v):
                raise ValueError("v 必须为 None 或非布尔整数")
            norm.append((s, a, b, v))
        return norm

    # ---------------- 批次入口 ----------------
    def apply(self, changes):
        norm = self._validate(changes)  # 全部校验通过才动手，保证整批原子

        touched = set()
        total_old = total_new = 0
        for s, a, b, v in norm:
            # 1) 更新来源层（只动 [a,b) 局部）
            self._layers[s].assign(a, b, v)

            # 2) 收集三层与 [a,b) 相交的段，事件扫描重算该窗口融合结果
            new_pieces = self._fuse(a, b)

            # 3) 融合树局部替换，拿回替换前旧段
            old_pieces, ins, rem = self._fused.replace_range(a, b, new_pieces)
            total_old += sum(x[1] for x in old_pieces)
            total_new += sum(x[1] for x in new_pieces)

            # 4) 日汇总：窗口内逐秒融合值的差分。
            # 旧段在边界处可能与窗口外同值段连续，直接整段加减会把窗口外
            # 部分重复计入，因此按新旧裁剪段做坐标归并扫描，仅对变化量调账。
            self._adjust_daily_diff(a, b, old_pieces, new_pieces)

            d0 = _day_floor(a)
            d1 = _day_floor(b - 1)
            for d in range(d0, d1 + 1):
                touched.add(d)

        self.last_apply = {
            "items": len(norm),
            "fused_segments": self._fused.size,
            "touched_days": len(touched),
            "replaced_old_seconds": total_old,
            "inserted_new_seconds": total_new,
        }

        # 5) 更正触及各日的最终汇总（无覆盖天给出 (0,0)）
        return {d: tuple(self._daily.get(d, [0, 0])) for d in sorted(touched)}

    # ---------------- 查询 ----------------
    def segments(self, a, b):
        if not (_is_int(a) and _is_int(b)) or a >= b:
            raise ValueError("segments 需要整数 a < b")
        raw = self._fused.range_pieces(a, b)
        # 裁剪后相邻同源同值可能在窗口边界处出现，做一次线性合并
        out = []
        for lo, length, (s, v) in raw:
            if out and out[-1][0] == s and out[-1][3] == v and \
                    out[-1][2] == lo:
                prev = out[-1]
                out[-1] = (s, prev[1], prev[2] + length, v)
            else:
                out.append((s, lo, lo + length, v))
        return out

    def daily(self, d):
        if not _is_int(d):
            raise ValueError("日号必须为非布尔整数")
        return tuple(self._daily.get(d, [0, 0]))

    # ---------------- 内部 ----------------
    def _fuse(self, a, b):
        """事件扫描重算 [a,b) 内的融合段，相邻同源同值即时合并。

        事件按坐标归组，同坐标内按 (来源优先级, 先生效后失效) 处理，
        保证同一来源“旧段失效 + 新段生效”在该坐标内顺序完成，且其他
        来源的生效先于失效，active 不会被瞬时清空而让低来源越权。
        """
        events = []  # (坐标, 来源优先级, 类型, 来源, 值)；0=生效 1=失效
        for s in SOURCES:
            for lo, length, v in self._layers[s].range_pieces(a, b):
                events.append((lo, _PRIORITY[s], 0, s, v))
                events.append((min(lo + length, b), _PRIORITY[s], 1, s, v))
        events.sort()

        active = {}
        out = []

        def emit(lo, hi, s, v):
            if hi <= lo:
                return
            if out and out[-1][2] == (s, v) and out[-1][0] + out[-1][1] == lo:
                out[-1][1] += hi - lo
            else:
                out.append([lo, hi - lo, (s, v)])

        def top_source():
            for cand in SOURCES:  # A>B>C，首个有效即最高来源
                if cand in active:
                    return cand, active[cand]
            return None

        prev = a
        i = 0
        while i < len(events):
            coord = events[i][0]
            # 发射 [prev, coord)：用该坐标事件生效“之前”的 active
            if coord > prev:
                top = top_source()
                if top is not None:  # 缺失即空缺，不产生融合段
                    emit(prev, coord, top[0], top[1])
            # 同坐标全部事件原子应用，active 的瞬时中间态不外显
            while i < len(events) and events[i][0] == coord:
                _c, _prio, kind, s, v = events[i]
                if kind == 0:
                    active[s] = v
                elif active.get(s) == v:
                    # 仅当该来源当前值仍是此失效事件对应的旧值才移除；
                    # 同坐标的新生效已接管时不得清空
                    active.pop(s, None)
                i += 1
            prev = coord
        if prev < b:
            top = top_source()
            if top is not None:
                emit(prev, b, top[0], top[1])
        return out

    def _adjust_daily(self, lo, hi, value, sign):
        """把 [lo,hi) 段对各天汇总的贡献按 sign(+1/-1/0) 增量更新。

        sign=+1/-1 时同时调整加权和与覆盖秒数；sign=0 时 value 直接作为
        “每秒加权增量”（覆盖秒数不变）。循环以 hi 为真实终止条件，
        hi 可能早于当天结束，绝不能按整天外推。
        """
        d = _day_floor(lo)
        cursor = lo
        while cursor < hi:
            nxt = min(hi, (d + 1) * SECONDS_PER_DAY)
            secs = nxt - cursor
            agg = self._daily.get(d)
            if agg is None:
                agg = [0, 0]
                self._daily[d] = agg
            if sign == 0:
                agg[0] += value * secs
            else:
                agg[0] += sign * value * secs
                agg[1] += sign * secs
                if agg[1] == 0:  # 该天已无覆盖，移除存储
                    del self._daily[d]
            cursor = nxt
            d += 1

    def _adjust_daily_diff(self, a, b, old_pieces, new_pieces):
        """按窗口内逐秒融合值的新旧差分校正日汇总。

        把新旧裁剪段的端点归并为一组子区间，每个子区间内新旧值恒定，
        仅当值发生变化时调账（空缺视为“无覆盖”）。调账仍按天切分，
        因此只触及 [a,b) 跨越的至多 8 天，不逐秒展开。
        """
        coords = {a, b}
        for lo, length, _p in old_pieces:
            coords.add(lo)
            coords.add(lo + length)
        for lo, length, _p in new_pieces:
            coords.add(lo)
            coords.add(lo + length)
        coords = sorted(coords)

        oi = ni = 0
        old_val = new_val = None  # None 表示空缺

        for idx in range(len(coords) - 1):
            lo, hi = coords[idx], coords[idx + 1]
            if hi <= lo:
                continue
            # 推进到覆盖 lo 的旧段（段集合互不重叠且按起点排序）
            while oi < len(old_pieces) and \
                    old_pieces[oi][0] + old_pieces[oi][1] <= lo:
                oi += 1
            ov = old_pieces[oi][2][1] if oi < len(old_pieces) \
                and old_pieces[oi][0] <= lo else None
            # 同理推进新段
            while ni < len(new_pieces) and \
                    new_pieces[ni][0] + new_pieces[ni][1] <= lo:
                ni += 1
            nv = new_pieces[ni][2][1] if ni < len(new_pieces) \
                and new_pieces[ni][0] <= lo else None

            if ov is None and nv is None:
                continue  # 空缺 -> 空缺
            if ov is None:
                # 空缺 -> 有值：覆盖与加权同时增加
                self._adjust_daily(lo, hi, nv, +1)
            elif nv is None:
                # 有值 -> 空缺：覆盖与加权同时减回
                self._adjust_daily(lo, hi, ov, -1)
            elif ov != nv:
                # 有值 -> 另一个值：覆盖秒数不变，仅校正加权和
                self._adjust_daily(lo, hi, nv - ov, 0)

    # 演示/测试辅助
    def fused_segment_count(self):
        return self._fused.size

    def layer_segment_count(self, s):
        return self._layers[s].size
