# -*- coding: utf-8 -*-
"""
趋势波段策略 — 缩量回踩信号检测

核心买点逻辑：
- 缩量回踩 MA5（不破5日线 + 换手率>5%）
- 回踩 MA10（次优，需确认）
- 缩量贴 MA5（信号3，观察信号：同 setup 但当日收涨未回踩，不是买点，等回踩再接）
"""

import logging
from dataclasses import dataclass, fields
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 板块取不到时的显式占位符。
# 禁止用空串或 None：空串渲染出空单元格（视觉上像列错位），None 会渲染成字面量 "None"。
UNKNOWN_SECTOR = "未知板块"

# 信号类型白名单。新增买点形态时必须在此登记，并同步补 report._build_action_guide 的操作指引。
KNOWN_SIGNAL_TYPES = ("pullback_ma5", "pullback_ma10", "near_ma5")


class SignalFieldError(ValueError):
    """信号对象字段缺失或非法时抛出。

    设计取向：宁可让任务失败，也不让 0 / "" / None 静默流进渲染层——
    那只会产出一份"看起来正常、其实评分全为 0 或列全错位"的报告。
    """


@dataclass
class TechnicalSignal:
    """技术信号数据类。

    字段分两层：
    - **必填（无默认值）**：买点判定直接产生的事实，漏传即 TypeError，构造阶段就炸。
    - **可选（有默认值）**：板块、环境调节结果等外部补充信息。

    关于 effective_score：
        它必须是 None（尚未调节）或 apply_regime() 算出的值，**不允许默认 0**。
        渲染层一律经 effective 属性读取，未调节时直接抛异常——
        避免"漏跑调节步骤 → 静默变 0 → 所有信号被判进'暂不关注'"。

    Attributes:
        sector: 所属板块名称，取不到时为 UNKNOWN_SECTOR
        position_gain: 本轮起点涨幅（%），距近 60 日最低收盘的涨幅（与 veto V7 同口径）。
            开仓档位收紧（基础/半收紧/收紧）的判定输入，见 src/trend/entry_tier.py
        entry_tier: 市场状态对应的开仓档位（None = 不启用档位，禁止开仓）
        effective_score: 经市场环境档位登记后的有效评分（None = 未登记）
        regime_note: 档位说明（如"收紧档"）
    """

    # ── 必填（漏传即 TypeError）──
    code: str
    name: str
    signal_type: str  # 'pullback_ma5' / 'pullback_ma10'，见 KNOWN_SIGNAL_TYPES
    score: int  # 技术评分 0-100
    current_price: float
    ma5: float
    ma10: float
    ma20: float
    bias_ma5: float  # 乖离率(%)
    volume_ratio: float  # 量比
    turnover_rate: float  # 换手率(%)
    pct_change: float  # 当日涨跌幅(%)
    description: str  # 信号描述

    # ── 可选（有默认值）──
    sector: str = UNKNOWN_SECTOR  # 所属板块名称
    position_gain: float = 0.0  # 本轮起点涨幅（%），档位收紧判定用
    entry_tier: Optional[str] = None  # 开仓档位（None = 不启用档位）
    effective_score: Optional[int] = None  # 登记档位后的有效评分
    regime_note: str = ""  # 档位说明

    # 允许为 None 的可选字段（其余字段 None 一律视为脏数据）
    _NONE_ALLOWED = ("effective_score", "entry_tier")

    def __post_init__(self) -> None:
        """构造即校验：禁止 None、校验取值域，把错误挡在渲染层之前。"""
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.name not in self._NONE_ALLOWED:
                raise SignalFieldError(
                    f"{self.code or '?'}: 字段 {f.name} 为 None（禁止 None 流入渲染层）"
                )

        if not self.code or not self.name:
            raise SignalFieldError(f"信号缺少 code/name: code={self.code!r} name={self.name!r}")
        if self.signal_type not in KNOWN_SIGNAL_TYPES:
            raise SignalFieldError(
                f"{self.name}({self.code}): 未知 signal_type={self.signal_type!r}，"
                f"须为 {KNOWN_SIGNAL_TYPES}（新增买点形态须登记并补操作指引）"
            )
        if not 0 <= self.score <= 100:
            raise SignalFieldError(f"{self.name}({self.code}): score={self.score} 超出 0-100")
        for fname in ("current_price", "ma5", "ma10", "ma20"):
            value = getattr(self, fname)
            if value <= 0:
                raise SignalFieldError(f"{self.name}({self.code}): {fname}={value} 非正数")

    def apply_regime(self, tier: Optional[str], note: str = "") -> None:
        """登记市场状态对应的开仓档位，写入有效评分（渲染前必须且只需调用一次）。

        2026-09 改版：删除「市场状态 → 评分系数」（×1.0/×0.85/×0.8/×0.5），
        有效评分 = 技术评分。系数会把整批信号一起压低，弱的没筛掉、强的被拖进
        "暂不关注"，且"降分"与"收紧选股"混在一起，事后无法归因。
        环境的影响改为落到具体规则：
        - 选股收紧（位置/资金）→ 由 entry_tier.screen_by_tier 过滤；
        - 仓位限制（亏损限额 ÷ 止损距离）→ 由报告层按档位计算。
        """
        self.entry_tier = tier
        self.regime_note = note
        self.effective_score = self.score

    @property
    def effective(self) -> int:
        """渲染层读取有效评分的唯一入口：未调节即抛异常。"""
        if self.effective_score is None:
            raise SignalFieldError(
                f"{self.name}({self.code}): effective_score 未计算（未调用 apply_regime），拒绝渲染"
            )
        return self.effective_score

    @property
    def sector_known(self) -> bool:
        """板块是否取到（False 时依赖板块的判定不可用）。"""
        return self.sector != UNKNOWN_SECTOR


def _position_penalty(pos: float) -> float:
    """位置扣分：距近 60 日低点涨幅越高扣分越重（与 veto V7 同一口径）。

    映射（线性 0.4 分/百分点）：≤25% 扣 0，30% 扣 2，40% 扣 6，
    50% 扣 10，60% 扣 14，70% 扣 18，≥75% 扣满 20 分。
    ≥80% 由 veto V7 在前置层否决，不会到达此处。
    """
    if pos <= 25:
        return 0.0
    if pos >= 75:
        return 20.0
    return (pos - 25) * 0.4


def _compute_signal_score(signal_type: str, metrics: dict) -> int:
    """
    根据多维指标动态计算信号评分，替代硬编码的 90/65。

    Args:
        signal_type: 'pullback_ma5' / 'near_ma5' / 'pullback_ma10'
        metrics: 包含以下键的字典：
            bias_ma5, volume_ratio, pct_change, turnover_rate,
            close_position, lower_shadow, recently_above_ma5_count,
            position_gain（距近 60 日低点涨幅%，可选）

    Returns:
        0-100 的评分
    """
    bias = metrics.get('bias_ma5', 0)
    vr = metrics.get('volume_ratio', 1.0)
    pct = metrics.get('pct_change', 0)
    tr = metrics.get('turnover_rate', 0)
    cp = metrics.get('close_position', 0.5)
    ls = metrics.get('lower_shadow', 0.1)
    above_count = metrics.get('recently_above_ma5_count', 3)
    pos = metrics.get('position_gain', 0.0)

    if signal_type in ('pullback_ma5', 'near_ma5'):
        # near_ma5（缩量贴MA5）与 pullback_ma5 是同一组 setup，共用同一套权重；
        # 但封顶 79 分：≥80 会进报告「优先关注（适合尾盘介入）」档，
        # 与 near_ma5"不是买点、等回踩"的剧本矛盾。
        # --- bias_ma5 (0~25分): 越贴近MA5越好 ---
        if 0.0 <= bias <= 2.0:
            bias_score = 25 - abs(bias - 0.5) * 10  # 峰值为+0.5%
        elif -1.0 <= bias < 0.0:
            bias_score = 20 - abs(bias) * 5  # 负乖离也可接受
        else:
            bias_score = max(0, 10 - abs(bias - 2.0) * 5)

        # --- volume_ratio (0~20分): 适度缩量最好 ---
        if 0.5 <= vr <= 0.9:
            vol_score = 20 - abs(vr - 0.7) * 25  # 峰值为0.7
        elif 0.4 <= vr < 0.5:
            vol_score = 12
        elif 0.9 < vr <= 1.05:
            vol_score = 10 - (vr - 0.9) * 66
        else:
            vol_score = max(0, 5)

        # --- pct_change (0~15分): 接近零最好 ---
        if -2.0 <= pct <= 1.0:
            pct_score = 15 - abs(pct) * 3
        elif -4.0 <= pct < -2.0 or 1.0 < pct <= 4.0:
            pct_score = 10 - abs(pct - 1.0) * 2
        else:
            pct_score = max(0, 5)

        # --- turnover_rate (0~15分): 4~7%最佳 ---
        if 4.0 <= tr <= 7.0:
            tr_score = 15
        elif 3.0 <= tr < 4.0 or 7.0 < tr <= 10.0:
            tr_score = 10
        else:
            tr_score = max(0, 5)

        # --- intraday_support (0~15分) ---
        if cp > 0.6 and ls > 0.2:
            id_score = 15
        elif cp > 0.4 and ls > 0.1:
            id_score = 10
        else:
            id_score = 5

        # --- recently_above_ma5 (0~10分) ---
        above_score = min(10, above_count * 2)

        score = bias_score + vol_score + pct_score + tr_score + id_score + above_score
        score -= _position_penalty(pos)
        score_cap = 79 if signal_type == 'near_ma5' else 95
        return min(score_cap, max(50, score))

    elif signal_type == 'pullback_ma10':
        # MA10回踩评分：更低基础，保守权重
        # bias_ma5 (0~15分)
        if -3.0 <= bias <= 0:
            bias_score = 15 - abs(bias + 1.0) * 3
        else:
            bias_score = max(0, 8 - abs(bias) * 2)

        # volume_ratio (0~20分)
        if 0.4 <= vr <= 0.9:
            vol_score = 20 - abs(vr - 0.6) * 20
        else:
            vol_score = max(0, 8)

        # pct_change (0~15分)
        if -4.0 <= pct <= 1.0:
            pct_score = 15 - abs(pct + 1.0) * 2
        else:
            pct_score = max(0, 5)

        # turnover_rate (0~20分)
        if 3.0 <= tr <= 8.0:
            tr_score = 20 - abs(tr - 5.0) * 3
        else:
            tr_score = max(0, 8)

        # intraday_support (0~15分)
        id_score = 10 if cp > 0.4 and ls > 0.1 else 5

        # touches_ma10 (0~15分)
        touches = metrics.get('touches_ma10', False)
        touch_score = 15 if touches else 0

        score = bias_score + vol_score + pct_score + tr_score + id_score + touch_score
        score -= _position_penalty(pos)
        return min(85, max(30, score))

    return 50


def detect_pullback_signals(code: str, name: str, df: pd.DataFrame) -> List[TechnicalSignal]:
    """
    检测缩量回踩 MA5 / MA10 趋势波段信号。

    Args:
        code: 股票代码
        name: 股票名称
        df: 已计算 MA5/MA10/MA20 的日线 DataFrame

    Returns:
        匹配规则的 TechnicalSignal 列表
    """
    signals: List[TechnicalSignal] = []

    if df is None:
        logger.warning(f"⚠️ {name}({code}): 数据为空，跳过分析")
        return signals

    if len(df) < 20:
        logger.warning(f"⚠️ {name}({code}): 数据不足20条(仅{len(df)}条)，可能影响分析准确性")

    df = df.sort_values('date').reset_index(drop=True)

    # 取最新数据
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    current_price = latest['close']
    ma5 = latest['ma5']
    ma10 = latest['ma10']
    ma20 = latest['ma20']
    _vr = latest.get('volume_ratio')
    volume_ratio = float(_vr) if _vr is not None and pd.notna(_vr) else 1.0

    if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
        return signals

    # 计算乖离率
    bias_ma5 = (current_price - ma5) / ma5 * 100 if ma5 > 0 else 0
    bias_ma20 = (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0

    # 计算当日涨跌幅（用于条件过滤 + 描述）
    pct_change = 0.0
    if prev is not None and prev['close'] > 0:
        pct_change = (current_price - prev['close']) / prev['close'] * 100

    # 当日涨跌幅是否温和（排除涨停/准涨停和崩盘式下跌）
    is_moderate_change = -5.0 < pct_change < 7.0

    # 计算日内承接：通过上下影线和收盘位置判断资金承接意愿
    intraday_range = latest['high'] - latest['low']
    if intraday_range > 0:
        close_position = (latest['close'] - latest['low']) / intraday_range
        upper_shadow_ratio = (latest['high'] - max(latest['open'], latest['close'])) / intraday_range
        lower_shadow_ratio = (min(latest['open'], latest['close']) - latest['low']) / intraday_range
        has_intraday_support = (
            close_position > 0.4
            and lower_shadow_ratio > 0.1
        )
    else:
        close_position = 0.5
        has_intraday_support = True

    # === 策略检查 ===

    # 1. 均线多头排列：5日线 > 10日线 > 20日线
    is_bullish_alignment = ma5 > ma10 > ma20

    # 2. 不破5日线（或盘中破但尾盘收回）
    holds_ma5 = current_price >= ma5 * 0.995  # 允许微破

    # 3. 选股池条件：换手率 > 3%（保证活跃度，缩量日允许适当降低）
    #    换手率缺失（数据源未提供）时视为「未知」：跳过该子条件放行并发告警，
    #    避免「默认0 → 永远挡掉所有信号」的静默黑屏（见 base._BACKFILL_COLUMNS）。
    _tr = latest.get('turnover_rate')
    turnover_unknown = _tr is None or pd.isna(_tr)
    turnover = 0.0 if turnover_unknown else float(_tr)
    meets_liquidity = True if turnover_unknown else turnover > 3.0
    if turnover_unknown:
        logger.warning(f"  {name}({code}): 换手率缺失，流动性子条件跳过（不因此挡单）")

    # 4. 非情绪过热检查：排除短期涨幅过大、偏离5日线过远、波动剧烈的标的
    is_euphoric = False
    recent_3d_gain = recent_5d_gain = recent_max_amplitude = 0.0
    if len(df) >= 4:
        recent_3d_gain = (df.iloc[-1]['close'] - df.iloc[-4]['close']) / df.iloc[-4]['close'] * 100
        recent_5d_gain = (df.iloc[-1]['close'] - df.iloc[-6]['close']) / df.iloc[-6]['close'] * 100 if len(df) >= 6 else 0
        recent_max_bias_ma5 = max(
            (df.iloc[i]['close'] - df.iloc[i]['ma5']) / df.iloc[i]['ma5'] * 100
            for i in range(-min(5, len(df)), 0)
            if pd.notna(df.iloc[i]['ma5']) and df.iloc[i]['ma5'] > 0
        ) if len(df) >= 2 else 0
        # 近3日最大振幅（高波动 = 博弈激烈，不适合低吸）
        recent_max_amplitude = max(
            (df.iloc[i]['high'] - df.iloc[i]['low']) / df.iloc[i-1]['close'] * 100
            for i in range(-3, 0)
            if df.iloc[i-1]['close'] > 0
        )
        if recent_3d_gain >= 18 or recent_5d_gain >= 30 or recent_max_bias_ma5 >= 12 or recent_max_amplitude >= 15:
            is_euphoric = True
        # 单日暴涨（涨停/准涨停）视为情绪过热
        if pct_change > 7.0:
            is_euphoric = True
    # MA20 乖离过大：收盘偏离20日线过远，追高风险高
    if bias_ma20 > 15.0:
        is_euphoric = True

    # 4.4 位置维度：距近60日最低收盘涨幅（与 veto V7 统一口径）
    #     涨幅≥80% → 信号失效（追高风险极高），<80% 时作为评分扣分维度（见 _position_penalty）
    low_60 = float(df['close'].tail(60).min())
    position_gain = (current_price - low_60) / low_60 * 100 if low_60 > 0 else 0.0
    is_overextended = position_gain >= 80.0

    # 说明：原「妖股拦截」（近5日涨停≥3次）已迁入 veto_rules 的 V6（近20日涨跌停≥3天），
    # 口径更宽且覆盖原条件（5日涨停3次必然满足20日涨跌停3天），此处不再重复判定。

    # 5. 量能检查：当日成交量 < 5日均量 * 1.1（不允许爆量，而非必须地量）
    current_volume = latest['volume']
    volume_ma5 = df['volume'].rolling(5).mean().iloc[-1] if len(df) >= 5 else 0
    no_volume_blowoff = volume_ma5 > 0 and current_volume < volume_ma5 * 1.1

    # 6. 第一次回踩MA5检查：过去5天至少3天收盘在MA5之上
    above_ma5_count = (
        sum(1 for i in range(-6, -1) if df.iloc[i]['close'] > df.iloc[i]['ma5'])
        if len(df) >= 6 else 0
    )
    recently_above_ma5 = above_ma5_count >= 3

    # 7. MA10 回踩确认：盘中最低价接近MA10且收盘守住
    touches_ma10 = latest['low'] <= ma10 * 1.01 and current_price >= ma10

    # 8. 回踩形态判定：当日收跌，或盘中最低价触及 MA5 附近。
    #    容差 1.005 与 holds_ma5 的 0.995 对称（一侧允许微破、一侧允许贴线）。
    #    信号1（缩量回踩MA5）与信号3（缩量贴MA5）共用同一组 setup 闸门，仅以此形态分岔：
    #    真回踩 → 买点信号；收涨未回踩 → 观察信号（不是买点，等回踩再接）。
    is_pullback_shape = pct_change < 0 or latest['low'] <= ma5 * 1.005

    # 只有均线多头排列的股票才有分析意义
    if not is_bullish_alignment:
        logger.debug(f"  {name}({code}) ✗ 均线非多头排列: ma5={ma5:.2f} ma10={ma10:.2f} ma20={ma20:.2f}")
        return signals

    # === 诊断日志：逐条件输出 ===
    _cond_fails = []
    if not holds_ma5:
        _cond_fails.append(f"未守住MA5(price={current_price:.2f} vs ma5*0.995={ma5*0.995:.2f})")
    if not no_volume_blowoff:
        _cond_fails.append(f"放量(vol={current_volume:.0f} vs vol_ma5*1.1={volume_ma5*1.1:.0f})")
    if not (-1.5 < bias_ma5 < 3.5):
        _cond_fails.append(f"乖离率超范围(bias_ma5={bias_ma5:+.2f}%)")
    if not has_intraday_support:
        _cond_fails.append(f"日内承接弱(cp={close_position:.2f} us={upper_shadow_ratio:.2f} ls={lower_shadow_ratio:.2f})")
    if not meets_liquidity:
        _cond_fails.append(f"换手率不足(turnover={turnover:.2f}%)")
    if is_euphoric:
        _cond_fails.append(f"情绪过热(3d={recent_3d_gain:.1f}% 5d={recent_5d_gain:.1f}% 振幅={recent_max_amplitude:.1f}% 涨跌={pct_change:+.2f}% MA20乖离={bias_ma20:+.1f}%)")
    if is_overextended:
        _cond_fails.append(f"距60日低点涨幅过大({position_gain:+.1f}%≥80%)")
    if not recently_above_ma5:
        _cond_fails.append("5天内<3天站在MA5之上")
    if not is_moderate_change:
        _cond_fails.append(f"涨跌幅异常(pct_change={pct_change:+.2f}%)")

    if _cond_fails:
        logger.debug(f"  {name}({code}) 信号1不满足 | {'; '.join(_cond_fails)}")
    else:
        logger.info(f"  {name}({code}) ✅ 信号1/3 setup 条件满足（按当日形态分岔为回踩/贴线）")

    # 信号1: 缩量回踩 MA5 — 最佳买点（当日真回踩：收跌或盘中触及 MA5）
    # 信号3: 缩量贴 MA5 — 同一 setup 的非回踩形态（收涨未回踩），是观察信号而非买点
    # 共用条件：多头排列 + 守住MA5 + 缩量 + 小实体/小跌 + 换手达标 + 非加速 + 涨跌幅温和
    meets_ma5_setup = (holds_ma5 and no_volume_blowoff and -1.5 < bias_ma5 < 3.5
                       and has_intraday_support and meets_liquidity
                       and not is_euphoric and not is_overextended
                       and recently_above_ma5 and is_moderate_change)

    if meets_ma5_setup:
        signal_type = 'pullback_ma5' if is_pullback_shape else 'near_ma5'
        if is_pullback_shape:
            signal_desc = f"缩量回踩MA5（量比{volume_ratio:.2f}）不破5日线，涨跌{pct_change:+.2f}%"
        else:
            signal_desc = f"缩量贴MA5（量比{volume_ratio:.2f}）收涨未回踩，涨跌{pct_change:+.2f}%"
        if turnover > 8.0 and 2.0 <= pct_change <= 5.0:
            signal_desc += " ⭐换手活跃具龙头特征"
        elif turnover > 10.0:
            signal_desc += f" 换手{turnover:.1f}%分歧剧烈"
        elif turnover >= 7.0:
            signal_desc += f" 换手{turnover:.1f}%偏热"
        elif turnover >= 4.0:
            signal_desc += f" 换手{turnover:.1f}%正常"

        # --- 一阳三阴形态标注（借鉴 one_yang_three_yin 策略）---
        if len(df) >= 4:
            anchor = df.iloc[-4]
            anchor_pct = (anchor['close'] - anchor['open']) / anchor['open'] * 100 if anchor['open'] > 0 else 0
            pullback_days = df.iloc[-3:]
            if anchor_pct > 3.0 and all(row['close'] < row['open'] for _, row in pullback_days.iterrows()):
                # 量确认：后三天量递减，且平均量 < 阳线量
                pullback_volumes = df.iloc[-3:]['volume']
                if (pullback_volumes.is_monotonic_decreasing
                        and pullback_volumes.mean() < anchor['volume']):
                    signal_desc += " 📐一阳三阴形态"

        # 计算动态评分（near_ma5 与 pullback_ma5 同权重，评分函数内对 near_ma5 封顶 79）
        s1_metrics = {
            'bias_ma5': bias_ma5,
            'volume_ratio': volume_ratio,
            'pct_change': pct_change,
            'turnover_rate': turnover,
            'close_position': close_position,
            'lower_shadow': lower_shadow_ratio,
            'recently_above_ma5_count': above_ma5_count if len(df) >= 6 else 3,
            'position_gain': position_gain,
        }
        dynamic_score = _compute_signal_score(signal_type, s1_metrics)

        signals.append(TechnicalSignal(
            code=code,
            name=name,
            signal_type=signal_type,
            score=int(dynamic_score),
            current_price=current_price,
            ma5=ma5,
            ma10=ma10,
            ma20=ma20,
            bias_ma5=bias_ma5,
            volume_ratio=volume_ratio,
            turnover_rate=turnover,
            pct_change=pct_change,
            description=signal_desc,
            position_gain=position_gain,
        ))

    # 信号2: 缩量回踩 MA10（次优买点 — 回踩较深，需确认支撑）
    # 策略强调"不破5日线"，回踩MA10说明分歧较大，评分降低
    elif (no_volume_blowoff
          and touches_ma10  # 盘中回踩MA10且收盘守住
          and has_intraday_support
          and meets_liquidity
          and not is_euphoric
          and not is_overextended
          and not holds_ma5):  # 确实跌破了MA5
        # 计算动态评分替代硬编码65分
        s2_metrics = {
            'bias_ma5': bias_ma5,
            'volume_ratio': volume_ratio,
            'pct_change': pct_change,
            'turnover_rate': turnover,
            'close_position': close_position,
            'lower_shadow': lower_shadow_ratio,
            'touches_ma10': touches_ma10,
            'position_gain': position_gain,
        }
        dynamic_score_s2 = _compute_signal_score('pullback_ma10', s2_metrics)

        signals.append(TechnicalSignal(
            code=code,
            name=name,
            signal_type='pullback_ma10',
            score=int(dynamic_score_s2),
            current_price=current_price,
            ma5=ma5,
            ma10=ma10,
            ma20=ma20,
            bias_ma5=bias_ma5,
            volume_ratio=volume_ratio,
            turnover_rate=turnover,
            pct_change=pct_change,
            description=f"回踩MA10（回踩较深），缩量（量比{volume_ratio:.2f}），涨跌{pct_change:+.2f}%，需次日弱转强确认",
            position_gain=position_gain,
        ))

    else:
        # 信号2也未触发，输出信号2专属失败原因
        _s2_fails = []
        if holds_ma5:
            _s2_fails.append("仍守住MA5(需跌破才触发回踩MA10)")
        if not no_volume_blowoff:
            _s2_fails.append(f"放量(vol={current_volume:.0f} vs vol_ma5*1.1={volume_ma5*1.1:.0f})")
        if not touches_ma10:
            _s2_fails.append(f"未触及MA10(low={latest['low']:.2f} vs ma10*1.01={ma10*1.01:.2f})")
        if not has_intraday_support:
            _s2_fails.append(f"日内承接弱(cp={close_position:.2f} us={upper_shadow_ratio:.2f} ls={lower_shadow_ratio:.2f})")
        if not meets_liquidity:
            _s2_fails.append(f"换手率不足(turnover={turnover:.2f}%)")
        if is_euphoric:
            _s2_fails.append(f"情绪过热(MA20乖离={bias_ma20:+.1f}%)")
        if is_overextended:
            _s2_fails.append(f"距60日低点涨幅过大({position_gain:+.1f}%)")
        logger.debug(f"  {name}({code}) 信号2不满足 | {'; '.join(_s2_fails)}")

    # 注意：不放量突破信号 — 策略明确规定"不做加速追高"

    return signals
