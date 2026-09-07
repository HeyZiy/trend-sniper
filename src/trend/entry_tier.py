# -*- coding: utf-8 -*-
"""
===================================
趋势策略 — 开仓规则档位（基础 / 半收紧 / 收紧）
===================================

对应 strategy/trend_strategy.md「市场状态分级与响应动作」→「开仓规则档位」。

设计要点（2026-09 改版）：
- 删除原「市场状态 → 评分系数」（×1.0 / ×0.85 / ×0.8 / ×0.5）。
  系数会把整批信号一起压低，弱的没被筛掉、强的却被拖进"暂不关注"，
  且"降分"与"收紧选股"混为一谈，事后无法归因。
- 改为按市场状态决定**启用哪一档开仓规则**，环境调整全部落到具体规则上：
  - **选股收紧**：位置（本轮起点涨幅上限，口径与 veto V7 一致）
  - **资金确认**：收紧档额外要求近 5 日主力净流入为正
  - **仓位限制**：单笔亏损限额（元），仓位上限 = 亏损限额 ÷ 止损距离

三档对应关系：trending_up→基础档、weak_up→半收紧档、sideways→收紧档；
trending_down / chaos 不启用档位（禁止开仓）。
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 档位标识 ──
TIER_BASE = "base"     # 基础档：trending_up
TIER_HALF = "half"     # 半收紧档：weak_up
TIER_TIGHT = "tight"   # 收紧档：sideways

# 市场状态 → 开仓档位；None = 该状态不启用开仓档位（禁止开仓）
REGIME_TIER = {
    "trending_up": TIER_BASE,
    "weak_up": TIER_HALF,
    "sideways": TIER_TIGHT,
    "trending_down": None,
    "chaos": None,
}

# 止损距离下限（%）：价格紧贴 MA10 时止损距离趋近 0，直接相除会算出天量仓位
MIN_STOP_LOSS_PCT = 3.0


@dataclass(frozen=True)
class TierRule:
    """一档开仓规则。

    Attributes:
        tier: 档位标识
        label: 中文名（报告文案用）
        position_gain_max: 本轮起点涨幅上限（%），None = 位置不收紧
        require_money_inflow: 是否要求近 5 日主力净流入为正
        loss_budget: 单笔亏损限额（元）
    """

    tier: str
    label: str
    position_gain_max: Optional[float]
    require_money_inflow: bool
    loss_budget: float

    def describe(self) -> str:
        """人类可读的规则摘要（报告展示用）。"""
        if self.position_gain_max is None:
            parts = ["位置不收紧"]
        else:
            parts = [f"位置<{self.position_gain_max:.0f}%"]
        if self.require_money_inflow:
            parts.append("近5日主力净流入为正")
        return f"{self.label}：{'；'.join(parts)}；单笔亏损限额{self.loss_budget:.0f}元"


# 档位规则表（半收紧档的 65% / 150 元为初始拟值，需积累样本后回测校准）
TIER_RULES = {
    TIER_BASE: TierRule(TIER_BASE, "基础档", None, False, 200.0),
    TIER_HALF: TierRule(TIER_HALF, "半收紧档", 65.0, False, 150.0),
    TIER_TIGHT: TierRule(TIER_TIGHT, "收紧档", 50.0, True, 100.0),
}

# 档位标识 → 中文名（无档位时的兜底文案由调用方处理）
TIER_LABELS = {t: r.label for t, r in TIER_RULES.items()}


def resolve_tier(regime: str) -> Optional[str]:
    """市场状态 → 开仓档位。

    Returns:
        档位标识；None = 该状态不启用开仓档位（trending_down / chaos 禁止开仓）
    """
    return REGIME_TIER.get(regime)


def tier_rule(tier: Optional[str]) -> Optional[TierRule]:
    """取档位规则；tier 为 None 或未知时返回 None。"""
    return TIER_RULES.get(tier) if tier else None


def check_position(tier: Optional[str], position_gain: float) -> Optional[str]:
    """档位的位置收紧检查（本地计算，无外部 I/O）。

    Args:
        tier: 开仓档位
        position_gain: 本轮起点涨幅（%），即距近 60 日最低收盘的涨幅（与 V7 同口径）

    Returns:
        None = 通过；否则返回拦截原因
    """
    rule = tier_rule(tier)
    if rule is None or rule.position_gain_max is None:
        return None
    if position_gain >= rule.position_gain_max:
        return (
            f"{rule.label}位置收紧：距60日低点涨幅{position_gain:.1f}% "
            f"≥ {rule.position_gain_max:.0f}%"
        )
    return None


def check_money_flow(tier: Optional[str], net_inflow: Optional[float]) -> Optional[str]:
    """档位的资金确认检查（收紧档要求近 5 日主力净流入为正）。

    数据缺失（net_inflow 为 None）一律 fail-open 放行并记 warning，
    与负面清单外部数据类（V1/V5）同约定——避免数据源抖动导致系统静默黑屏。

    Returns:
        None = 通过；否则返回拦截原因
    """
    rule = tier_rule(tier)
    if rule is None or not rule.require_money_inflow:
        return None
    if net_inflow is None:
        logger.warning(f"  {rule.label}资金确认：主力净流入数据缺失，fail-open 放行")
        return None
    if net_inflow <= 0:
        return (
            f"{rule.label}资金确认：近5日主力净流入{net_inflow / 1e8:.2f}亿 ≤ 0"
        )
    return None


def position_size(tier: Optional[str], stop_loss_pct: float) -> float:
    """按档位亏损限额与止损距离算仓位上限（元）。

    仓位计算：亏损限额 ÷ 止损距离。例：止损 5%，收紧档 100 ÷ 5% = 2000 元。

    Args:
        tier: 开仓档位（None 时返回 0）
        stop_loss_pct: 止损距离（%），不足 MIN_STOP_LOSS_PCT 时按下限计
    """
    rule = tier_rule(tier)
    if rule is None:
        return 0.0
    try:
        pct = max(float(stop_loss_pct), MIN_STOP_LOSS_PCT)
    except (TypeError, ValueError):
        pct = MIN_STOP_LOSS_PCT
    return rule.loss_budget / pct * 100


def screen_by_tier(
    signals,
    tier: Optional[str],
    net_inflow_fn: Optional[Callable[[str, str], Optional[float]]] = None,
) -> Tuple[List, List[Tuple[str, str, str]]]:
    """按开仓档位的收紧规则过滤已产出信号的候选。

    资金确认依赖妙想 API 且有日调用限额，只对**通过位置收紧**的候选惰性调用
    net_inflow_fn；未传该回调时跳过资金确认并记 warning（不静默放行）。

    Args:
        signals: TechnicalSignal 列表（须含 position_gain）
        tier: 开仓档位；None 表示当前状态禁止开仓（全部拦截）
        net_inflow_fn: (code, name) → 近5日主力净流入（元），None = 取不到

    Returns:
        (passed, blocked)：通过的信号列表；blocked 为 (code, name, reason) 列表
    """
    if tier is None:
        blocked = [(s.code, s.name, "当前市场状态不启用开仓档位（禁止开仓）")
                   for s in signals]
        if blocked:
            logger.info(f"档位：当前状态禁止开仓，{len(blocked)} 个信号仅作观察")
        return [], blocked

    rule = tier_rule(tier)
    passed: List = []
    blocked: List[Tuple[str, str, str]] = []

    for s in signals:
        reason = check_position(tier, getattr(s, "position_gain", 0.0) or 0.0)
        if reason is None and rule.require_money_inflow:
            if net_inflow_fn is None:
                logger.warning(
                    f"  {s.name}({s.code}): {rule.label}资金确认未提供取数回调，跳过该项"
                )
            else:
                reason = check_money_flow(tier, net_inflow_fn(s.code, s.name))
        if reason:
            blocked.append((s.code, s.name, reason))
            logger.info(f"🚫 档位收紧拦截 {s.name}({s.code}): {reason}")
        else:
            passed.append(s)

    if blocked:
        logger.info(f"档位{rule.label}：通过{len(passed)}个，拦截{len(blocked)}个")
    return passed, blocked
