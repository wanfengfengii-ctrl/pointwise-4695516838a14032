"""verify.py —— 小域逐秒对照核验 fusion.Store。

朴素参考模型：直接维护每个来源的逐秒字典，融合按 A>B>C 逐秒取值，
所有查询逐秒重算，与 Store 的增量结果做完全对照。
标准库实现：python3 verify.py
"""

import random

import fusion

SOURCES = ("A", "B", "C")
DAY = 86400


class NaiveStore:
    """逐秒朴素模型，只用于小域核验。"""

    def __init__(self):
        self.src = {s: {} for s in SOURCES}  # 时刻 -> 值

    def apply(self, changes):
        for s, a, b, v in changes:
            for t in range(a, b):
                if v is None:
                    self.src[s].pop(t, None)
                else:
                    self.src[s][t] = v

    def value_at(self, t):
        for s in SOURCES:  # A>B>C
            if t in self.src[s]:
                return s, self.src[s][t]
        return None

    def segments(self, a, b):
        out = []
        for t in range(a, b):
            cur = self.value_at(t)
            if cur is None:
                out.append(None)
            else:
                s, v = cur
                if out and out[-1] is not None:
                    ps, pa, pb, pv = out[-1]
                    if ps == s and pv == v and pb == t:
                        out[-1] = (ps, pa, pb + 1, pv)
                        continue
                out.append((s, t, t + 1, v))
        return [x for x in out if x is not None]

    def daily(self, d):
        weighted = covered = 0
        for t in range(d * DAY, (d + 1) * DAY):
            cur = self.value_at(t)
            if cur is not None:
                covered += 1
                weighted += cur[1]
        return weighted, covered


def pick_change(rng, lo, hi):
    s = rng.choice(SOURCES)
    a = rng.randint(lo, hi - 2)
    b = rng.randint(a + 1, min(hi, a + rng.randint(1, 3 * DAY)))
    v = None if rng.random() < 0.3 else rng.randint(-5, 5)
    return (s, a, b, v)


def check_spec_examples():
    st = fusion.Store()
    r1 = st.apply([("C", 0, 10, 2), ("A", 3, 7, 5)])
    assert r1 == {0: (32, 10)}, r1
    assert st.segments(0, 10) == [("C", 0, 3, 2), ("A", 3, 7, 5), ("C", 7, 10, 2)]
    assert st.daily(0) == (32, 10)
    r2 = st.apply([("A", 4, 6, None)])
    assert r2 == {0: (26, 10)}, r2
    assert st.segments(0, 10) == [("C", 0, 3, 2), ("A", 3, 4, 5),
                                  ("C", 4, 6, 2), ("A", 6, 7, 5), ("C", 7, 10, 2)]
    assert st.daily(0) == (26, 10)
    # 全部撤回 -> 空
    r3 = st.apply([("A", 0, DAY, None), ("C", 0, DAY, None)])
    assert r3 == {0: (0, 0)}, r3
    assert st.segments(0, DAY) == []
    print("[PASS] 题目示例与全部撤回收口")


def check_random_domain(lo, hi, label, rounds=80, seed=1234):
    rng = random.Random(seed)
    st = fusion.Store()
    nv = NaiveStore()
    for rnd in range(rounds):
        n = rng.randint(1, 6)
        changes = [pick_change(rng, lo, hi) for _ in range(n)]
        got = st.apply(changes)
        nv.apply(changes)

        # 整窗 segments 直接对照（含相邻合并、裁剪语义）
        assert st.segments(lo, hi) == nv.segments(lo, hi), (label, rnd)

        # 单次线性逐秒扫描对照（Store 侧沿其段列表推进，不重复查树）
        segs = st.segments(lo, hi)
        idx = 0
        for t in range(lo, hi):
            while idx < len(segs) and segs[idx][2] <= t:
                idx += 1
            if idx < len(segs) and segs[idx][1] <= t < segs[idx][2]:
                gv = (segs[idx][0], segs[idx][3])
            else:
                gv = None
            assert gv == nv.value_at(t), (label, rnd, t, gv, nv.value_at(t))

        # 随机子窗口裁剪对照
        for _ in range(3):
            a = rng.randint(lo, hi - 2)
            b = rng.randint(a + 1, hi)
            assert st.segments(a, b) == nv.segments(a, b), (label, rnd, a, b)

        # daily 对照覆盖域内所有天
        d0 = fusion._day_floor(lo)
        d1 = fusion._day_floor(hi - 1)
        for d in range(d0, d1 + 1):
            assert st.daily(d) == nv.daily(d), (label, d, st.daily(d), nv.daily(d))

        # apply 返回字典 == 各触及天最终汇总
        for d, agg in got.items():
            assert agg == nv.daily(d), (label, "return", d, agg, nv.daily(d))
    print(f"[PASS] 随机逐秒对照（{label}，{rounds} 批，域宽 {hi - lo} 秒）")


def check_zero_and_negative():
    st = fusion.Store()
    # 零值仍计覆盖：(-10,0) 内值为 0
    r = st.apply([("B", -10, 0, 0)])
    assert st.daily(-1) == (0, 10), st.daily(-1)
    assert r == {-1: (0, 10)}, r
    # 负日边界：[-86400, 86400) 跨 -1、0 两天
    st2 = fusion.Store()
    st2.apply([("C", -DAY, DAY, 3)])
    assert st2.daily(-1) == (3 * DAY, DAY)
    assert st2.daily(0) == (3 * DAY, DAY)
    assert st2.daily(1) == (0, 0)
    # 窗口裁剪与相邻合并
    assert st2.segments(0, 100) == [("C", 0, 100, 3)]
    print("[PASS] 零值计覆盖、负日与窗口裁剪")


def check_invalid_atomic():
    bad_batches = [
        [("X", 0, 1, 1)],                                  # 非法来源
        [("A", 5, 5, 1)],                                  # a==b
        [("A", 0, 1, True)],                               # 布尔值
        [("A", True, 2, 1)],                               # 布尔端点
        [("A", 0.0, 1, 1)],                                # 浮点端点
        [("A", 0, 1, "x")],                                # 非法 v
        [("A", 0, 7 * DAY + 1, 1)],                        # 超 7 天
        [("A", 0, 1)] * (fusion.MAX_BATCH_ITEMS + 1),      # 超 20 项
        [("A", 2, 1, 1)],                                  # a>b
        [("B", 0, 10, 1), ("Z", 0, 5, 2)],                 # 批内第二项非法
        [("B", 0, 10, 1), ("B", 0, 8 * DAY, 2)],           # 批内第二项跨度非法
        "not-a-list",
        [("A", 0, 1)],                                     # 元组长度不对
    ]
    for bad in bad_batches:
        st = fusion.Store()
        st.apply([("C", 0, 100, 7)])  # 预置数据
        snapshot = st.segments(-DAY, DAY)
        daily_snap = {d: st.daily(d) for d in range(-2, 3)}
        try:
            st.apply(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("应拒绝非法批: %r" % (bad,))
        # 整批不变：融合段与日汇总与执行前完全一致
        assert st.segments(-DAY, DAY) == snapshot
        assert all(st.daily(d) == daily_snap[d] for d in range(-2, 3))
        assert st.fused_segment_count() == 1

    # 空批合法且无副作用
    st = fusion.Store()
    assert st.apply([]) == {}
    assert st.daily(0) == (0, 0)
    print("[PASS] 非法批抛 ValueError 且整批原子不变（%d 组）" % len(bad_batches))


def check_batch_ordering():
    """同批依次解释：同源后项覆盖前项；撤回与赋值交错。"""
    st = fusion.Store()
    r = st.apply([
        ("A", 0, 100, 1),
        ("A", 10, 20, None),   # 撤回中间
        ("A", 15, 25, 9),      # 再赋值，与前段不相邻
        ("B", 0, 100, 2),      # 低来源补空缺
    ])
    expect = [("A", 0, 10, 1), ("B", 10, 15, 2), ("A", 15, 25, 9),
              ("A", 25, 100, 1)]
    got = st.segments(0, 100)
    assert got == expect, got
    # 日汇总核对：加权 = 1*10 + 2*5 + 9*10 + 1*75 = 185，覆盖 100
    assert st.daily(0) == (185, 100), st.daily(0)
    assert r == {0: (185, 100)}, r
    print("[PASS] 批内依次解释与优先级融合")


def check_boundary_merges():
    """跨窗口边界的同源同值合并与撤回后低层显露的专项场景。"""
    st = fusion.Store()
    st.apply([("C", 0, 100, 2)])
    st.apply([("A", 10, 90, 5)])
    # 撤回 A 后 C 应完整复原为单段（相邻合并）
    r = st.apply([("A", 10, 90, None)])
    assert st.segments(0, 100) == [("C", 0, 100, 2)], st.segments(0, 100)
    assert r == {0: (200, 100)}, r
    # 跨日撤回，触及多天
    st2 = fusion.Store()
    st2.apply([("A", DAY - 2, DAY + 2, 4)])
    r2 = st2.apply([("A", DAY - 1, DAY + 1, None)])
    assert set(r2.keys()) == {0, 1}, r2
    assert st2.daily(0) == (4, 1) and st2.daily(1) == (4, 1)
    print("[PASS] 边界合并与跨日撤回")


if __name__ == "__main__":
    check_spec_examples()
    check_random_domain(0, 2 * DAY, "正域")
    check_random_domain(-2 * DAY, DAY, "含负域", seed=4321)
    check_zero_and_negative()
    check_invalid_atomic()
    check_batch_ordering()
    check_boundary_merges()
    print("ALL VERIFY PASS")
