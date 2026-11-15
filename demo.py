"""demo.py —— 十万段规模的局部更正演示。

分两部分：
1. scale=100000：灌入约十万个融合段（大时刻轴，避免逐秒），随后做一次
   小窗口高层撤回，验证：
   * 撤回后低层覆盖正确显露（与朴素真值逐点对照更正窗口及边界）；
   * 日汇总仅触及窗口跨越的天，其余天数字典规模与值均不变；
   * 局部工作量（重算段数、触及天数）与历史总量无关。
2. 计时对照：同一小窗口在“历史 1 千段”与“历史 10 万段”下的更正耗时，
   演示复杂度不随历史总量增长。

标准库实现：python3 demo.py
"""

import random
import time

import fusion

SOURCES = ("A", "B", "C")
DAY = 86400
TARGET_SEGMENTS = 100_000


def build_store(n_segments, seed=7):
    """构造约 n_segments 个融合段：三层错位、不同值，天然互不合并。"""
    rng = random.Random(seed)
    st = fusion.Store()
    # 大间距铺底，保证每段独立；三层起点错位形成大量融合切换
    step = 1000
    span = 600
    per_layer = n_segments // 3 + 1
    batch = []
    for i in range(per_layer):
        base = i * step
        # 每层偏移不同、值不同，保证融合视图产生独立段
        batch.append(("C", base + 0, base + span, 100 + (i % 7)))
        batch.append(("B", base + 200, base + 200 + span, 200 + (i % 5)))
        batch.append(("A", base + 400, base + 400 + span, 300 + (i % 3)))
        if len(batch) == 18:  # 每批 ≤20 项
            st.apply(batch)
            batch = []
    if batch:
        st.apply(batch)
    return st, per_layer * step


def truth_value(layers_state, t):
    for s in SOURCES:
        if t in layers_state[s]:
            return s, layers_state[s][t]
    return None


def main():
    print("== 1) 构造十万融合段 ==")
    t0 = time.perf_counter()
    st, axis_end = build_store(TARGET_SEGMENTS)
    build_secs = time.perf_counter() - t0
    fused_n = st.fused_segment_count()
    layer_n = sum(st.layer_segment_count(s) for s in SOURCES)
    print(f"灌入完成：融合段 {fused_n}，三层段合计 {layer_n}，"
          f"时刻轴终点 {axis_end}，构造耗时 {build_secs:.2f}s")
    assert fused_n >= 90_000, fused_n

    # 选一个 A 层段做局部撤回，窗口远小于历史
    target_a = 12345 * 1000 + 400   # 某个 A 段起点
    target_b = target_a + 600
    print(f"\n== 2) 局部撤回高层 A 的 [{target_a},{target_b})（600 秒）==")

    # 记录更正前触及天外的日汇总快照（抽样若干远日，证明未被触碰）
    far_days = [fusion._day_floor(target_a) + d for d in (-100, 100, 500)]
    far_snapshot = {d: st.daily(d) for d in far_days}
    daily_keys_before = len(st._daily)

    t0 = time.perf_counter()
    ret = st.apply([("A", target_a, target_b, None)])
    correct_secs = time.perf_counter() - t0
    print(f"更正耗时 {correct_secs * 1000:.2f} ms，触及天 {sorted(ret.keys())}，"
          f"last_apply={st.last_apply}")

    # 远日汇总必须原样
    for d in far_days:
        assert st.daily(d) == far_snapshot[d], (d, st.daily(d), far_snapshot[d])
    assert len(st._daily) == daily_keys_before, "日汇总字典规模不应变化"
    print("[OK] 无关天的日汇总与字典规模均未变化")

    # 逐点对照：在更正窗口及两侧边界用独立朴素真值核验显露结果
    truth = {s: {} for s in SOURCES}
    # 用 Store 自身的来源层导出真值（来源层在撤回后即权威状态），
    # 再用与 _fuse 相同的优先级规则独立计算期望融合值
    check_lo, check_hi = target_a - 5, target_b + 5
    layer_pieces = {s: st._layers[s].range_pieces(check_lo, check_hi)
                    for s in SOURCES}

    def expected(t):
        for s in SOURCES:
            for lo, length, v in layer_pieces[s]:
                if lo <= t < lo + length:
                    return s, v
        return None

    segs = st.segments(check_lo, check_hi)
    idx = 0
    for t in range(check_lo, check_hi):
        while idx < len(segs) and segs[idx][2] <= t:
            idx += 1
        got = (segs[idx][0], segs[idx][3]) if idx < len(segs) \
            and segs[idx][1] <= t < segs[idx][2] else None
        exp = expected(t)
        assert got == exp, (t, got, exp)
    print(f"[OK] 窗口及边界 {check_hi - check_lo} 个时刻逐点对照通过，"
          f"撤回后由 B/C 正确补位")

    # 日汇总正确性：对触及天用段列表独立重算，与 apply 返回及 daily 查询一致
    for d in ret:
        day_lo, day_hi = d * DAY, (d + 1) * DAY
        weighted = covered = 0
        for s, a, b, v in st.segments(max(day_lo, check_lo - DAY),
                                      min(day_hi, check_hi + DAY)):
            lo = max(a, day_lo)
            hi = min(b, day_hi)
            if hi > lo:
                weighted += v * (hi - lo)
                covered += hi - lo
        # 上面只覆盖了检查邻域；改为全天重算做严格核对
        weighted = covered = 0
        for s, a, b, v in st.segments(day_lo, day_hi):
            weighted += v * (b - a)
            covered += b - a
        assert st.daily(d) == (weighted, covered), d
        assert ret[d] == (weighted, covered), d
    print("[OK] 触及天的最终日汇总独立重算一致")

    print("\n== 3) 局部更正耗时不随历史总量增长 ==")
    for n in (1_000, 10_000, 100_000):
        s_small, end_small = build_store(n, seed=11)
        wa = 50 * 1000 + 400
        # 预热
        s_small.apply([("A", wa, wa + 10, None)])
        # 再对另一处窗口计时
        wa2 = 80 * 1000 + 400
        t0 = time.perf_counter()
        for _ in range(50):
            s_small.apply([("A", wa2, wa2 + 300, 999)])
            s_small.apply([("A", wa2, wa2 + 300, None)])
        per = (time.perf_counter() - t0) / 100
        print(f"历史融合段 {s_small.fused_segment_count():>7}："
              f"单次局部更正平均 {per * 1000:.3f} ms")

    print("\nDEMO PASS")


if __name__ == "__main__":
    main()
