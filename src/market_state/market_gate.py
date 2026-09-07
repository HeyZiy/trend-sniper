# -*- coding: utf-8 -*-
"""
===================================
市场环境开仓门控模块
===================================

职责：
1. fetch_gate_inputs(): 【入口层调用】取门控所需外部数据（指数日线/成交额/涨跌停）
2. check_market_gate(gate_inputs): 纯判定——4 项门控 + 硬拦截，判断是否允许开仓
3. _check_hard_intercept(): 硬拦截层，检查极端环境
4. _detect_regime(): 根据均线状态判断市场结构（5 级）

门控条件（≥2 项通过才开仓）：
① 上证收盘 > MA20
②a 两市成交额 ≥ 1.5 万亿（DataFetcherManager 全市场统计）
②b 成交量 > 近 20 日均量（上证指数日线成交量同比）
③ 涨停 ≥ 30 且涨停 > 跌停 × 1.5

硬拦截（触发任一即锁仓）：
- 成交额冰点：两市成交额 < 1.5 万亿 且连续 ≥ 3 天（同②a数据源）
- 千股跌停：跌停 ≥ 50 且跌停 > 涨停 × 3
- 指数暴跌：上证单日跌幅 > 3%
- 成交量骤降：当日成交量 < 近 20 日均量 × 0.5
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# 状态判定的命中路径文案（与 _detect_regime 的判定顺序一一对应，供报告诊断用）
REGIME_PATHS = {
    "trending_down": "① 均线空头排列(MA5<MA10<MA20) + 收盘<MA10",
    "trending_up":   "② 均线多头排列(MA5>MA10>MA20) + 收盘>MA10 + 门控≥2",
    "sideways":      "③ 收盘紧贴MA20（偏离<1.5%）",
    "weak_up":       "④ 收盘>MA20 + 门控≥2（非标准多头排列）",
    "chaos":         "⑤ 以上均不满足（或数据不足）",
}

SIDEWAYS_BIAS = 0.015  # sideways 判定阈值：收盘偏离 MA20 < 1.5%

# 可开仓的市场状态（对应 strategy/trend_strategy.md「市场状态分级与响应动作」）：
# trending_up/weak_up/sideways 分别启用基础/半收紧/收紧档开仓规则（见 src/trend/entry_tier.py）；
# trending_down（均线空头）与 chaos（方向不明）不启用档位，无论门控通过几项都禁止开新仓。
REGIME_CAN_OPEN = ("trending_up", "weak_up", "sideways")


def _n(value: Optional[float]) -> Optional[float]:
    """float 统一 2 位小数（报告口径，避免各处精度不一）。"""
    return None if value is None else round(float(value), 2)


def _alignment_text(ma5: float, ma10: float, ma20: float) -> str:
    """均线排列描述，如 'MA5>MA10>MA20（多头排列）'。"""
    order = sorted((("MA5", ma5), ("MA10", ma10), ("MA20", ma20)),
                   key=lambda kv: kv[1], reverse=True)
    seq = ">".join(name for name, _ in order)
    if seq == "MA5>MA10>MA20":
        label = "（多头排列）"
    elif seq == "MA20>MA10>MA5":
        label = "（空头排列）"
    else:
        label = "（交织，无明确排列）"
    return f"{seq}{label}"


@dataclass
class RegimeDiagnosis:
    """市场状态判定的诊断信息（回答"为什么判成这个状态"）。

    Attributes:
        regime: 判定结果
        ma5/ma10/ma20/close: 判定所依据的均线与收盘（均为 2 位小数）
        bias_ma20: 收盘偏离 MA20 的百分比（正=在 MA20 上方）
        alignment: 均线排列描述
        path: 命中的判定路径文案
        note: 补充说明（如数据不足、门控项数）
    """
    regime: str
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    close: Optional[float] = None
    bias_ma20: Optional[float] = None
    alignment: str = "数据不足，无法判定"
    path: str = ""
    note: str = ""

    @property
    def available(self) -> bool:
        return self.ma20 is not None and self.close is not None

    def describe(self) -> str:
        """人类可读的三行诊断（供日志与报告共用）。"""
        if not self.available:
            return f"状态={self.regime}｜均线数据不足，无法给出排列与偏离（{self.note}）"
        return (
            f"状态={self.regime}｜{self.alignment}｜"
            f"收盘{self.close} 偏离MA20 {self.bias_ma20:+.2f}%｜命中：{self.path}"
        )

# ── 简单持久化：硬拦截中"成交额冰点连续天数"的记录文件 ──
import os
import json

def _ice_days_path():
    from pathlib import Path
    return Path(__file__).parent.parent.parent / "data" / "market_gate_ice_days.json"


def _load_ice_days() -> int:
    try:
        path = _ice_days_path()
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            return int(data.get("ice_days", 0))
    except Exception:
        pass
    return 0


def _save_ice_days(days: int):
    try:
        path = _ice_days_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ice_days": days, "updated": date.today().isoformat()}), encoding='utf-8')
    except Exception:
        pass


def diagnose_regime(index_df, met_count: int) -> RegimeDiagnosis:
    """根据指数均线状态判断市场结构，并给出可解释的诊断信息。

    判定优先级：trending_down > trending_up > sideways > weak_up > chaos

    Returns:
        RegimeDiagnosis：含 regime、MA5/MA10/MA20 排列、偏离 MA20 百分比、命中路径
        trending_up   — 均线多头排列 + 收盘在 MA10 上方 + 门控通过
        trending_down — 均线空头排列 + 收盘在 MA10 下方
        sideways      — 收盘紧贴 MA20（偏离 < 1.5%）
        weak_up       — 收盘在 MA20 上方 + 门控通过，但非明确多头
        chaos         — 数据不足或无法判断
    """
    try:
        if index_df is None or len(index_df) < 20:
            return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"],
                                   note="指数日线缺失或不足20条")
        ma5 = index_df['close'].rolling(5).mean().iloc[-1]
        ma10 = index_df['close'].rolling(10).mean().iloc[-1]
        ma20 = index_df['close'].rolling(20).mean().iloc[-1]
        close = index_df['close'].iloc[-1]
        if any(pd.isna(x) for x in [ma5, ma10, ma20]):
            return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"],
                                   note="MA5/MA10/MA20 存在 NaN")

        ma5, ma10, ma20, close = float(ma5), float(ma10), float(ma20), float(close)
        bias_ma20 = (close - ma20) / ma20 * 100 if ma20 > 0 else 0.0
        base = dict(
            ma5=_n(ma5), ma10=_n(ma10), ma20=_n(ma20), close=_n(close),
            bias_ma20=_n(bias_ma20), alignment=_alignment_text(ma5, ma10, ma20),
        )

        # ① trending_down — 均线空头，最高优先级
        if ma5 < ma10 < ma20 and close < ma10:
            return RegimeDiagnosis("trending_down", path=REGIME_PATHS["trending_down"],
                                   note=f"收盘{_n(close)} < MA10 {_n(ma10)}", **base)

        # ② trending_up — 均线多头 + 门控通过
        if ma5 > ma10 > ma20 and close > ma10 and met_count >= 2:
            return RegimeDiagnosis("trending_up", path=REGIME_PATHS["trending_up"],
                                   note=f"门控通过{met_count}项", **base)

        # ③ sideways — 紧贴 MA20 震荡
        if abs(close - ma20) / ma20 < SIDEWAYS_BIAS:
            return RegimeDiagnosis("sideways", path=REGIME_PATHS["sideways"],
                                   note=f"偏离MA20 {_n(bias_ma20):+.2f}%", **base)

        # ④ weak_up — 在 MA20 上方 + 门控通过，但不是标准多头排列
        if close > ma20 and met_count >= 2:
            return RegimeDiagnosis("weak_up", path=REGIME_PATHS["weak_up"],
                                   note=f"门控通过{met_count}项，但均线非标准多头", **base)

        # ⑤ chaos — 其余情况
        return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"],
                               note=f"门控通过{met_count}项，未满足以上任一结构", **base)
    except Exception as e:
        logger.warning(f"市场状态判定失败，降级为 chaos：{e}")
    return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"], note="判定异常，降级处理")


def _detect_regime(index_df, met_count: int) -> str:
    """同 diagnose_regime，但只返回状态字符串（内部判定用）。"""
    return diagnose_regime(index_df, met_count).regime


def _check_hard_intercept(
    index_df,
    total_amount_yi: float,
    limit_up: int,
    limit_down: int,
) -> Tuple[bool, str]:
    """检查硬拦截条件

    Returns:
        (is_intercepted, reason)
    """
    if index_df is None or len(index_df) < 20:
        return False, ""

    close = index_df['close'].iloc[-1]
    prev_close = index_df['close'].iloc[-2] if len(index_df) >= 2 else close
    idx_pct = (close - prev_close) / prev_close * 100 if prev_close else 0

    # 拦截 1：成交量骤降（当日 < 近20日均量 × 0.5）
    if 'volume' in index_df.columns:
        latest_vol = index_df['volume'].iloc[-1]
        avg_vol = index_df['volume'].iloc[-20:].mean()
        if pd.notna(avg_vol) and avg_vol > 0 and latest_vol < avg_vol * 0.5:
            return True, f"🔴 硬拦截-成交量骤降：当日沪市成交量{latest_vol/1e8:.1f}亿股 < 20日均量{avg_vol/1e8:.1f}亿股×0.5"

    # 拦截 2：成交额冰点（两市 < 1.5 万亿 且连续 ≥ 3 天）
    ice_days = _load_ice_days()
    if total_amount_yi > 0 and total_amount_yi < 15000:
        ice_days += 1
        _save_ice_days(ice_days)
        if ice_days >= 3:
            return True, f"🔴 硬拦截-成交额冰点：两市成交额{total_amount_yi:.0f}亿 < 1.5万亿，已连续{ice_days}天"
    else:
        if ice_days > 0:
            _save_ice_days(0)

    # 拦截 3：指数暴跌（上证单日跌幅 > 3%）
    if idx_pct < -3:
        return True, f"🔴 硬拦截-指数暴跌：上证单日跌幅{idx_pct:.1f}%"

    # 拦截 4：千股跌停（跌停 ≥ 50 且跌停 > 涨停 × 3）
    if limit_down >= 50 and limit_down > limit_up * 3:
        return True, f"🔴 硬拦截-千股跌停：跌停{limit_down}家 ≥ 50 且 > 涨停{limit_up}家×3"

    return False, ""


def fetch_gate_inputs(fm) -> Dict[str, Any]:
    """
    【入口层调用】获取门控所需的全部外部数据，供 check_market_gate 纯判定使用。

    数据需求：
    - 上证指数日线（data_provider.bars.get_index_daily；①②b 及硬拦截/均线判定用）
    - 两市成交额（fm.get_market_stats；②a 及成交额冰点硬拦截用）
    - 涨跌停家数（akshare 涨跌停池；③ 及千股跌停硬拦截用）

    单项获取失败不影响其他项（判定层对缺失数据逐项降级）。
    """
    inputs: Dict[str, Any] = {
        "index_df": None,
        "total_amount_yi": 0.0,
        "limit_up": 0,
        "limit_down": 0,
        "zt_fetched": False,
    }

    # akshare 导入一次即可；导入失败时指数日线与涨跌停两项一起降级（成交额项不依赖它）
    try:
        import akshare as ak
    except Exception:
        ak = None

    if ak is not None:
        try:
            from data_provider.bars import get_index_daily
            index_df = get_index_daily("sh000001")
            if index_df is not None and len(index_df) >= 20:
                inputs["index_df"] = index_df
        except Exception as e:
            logger.warning(f"获取上证指数日线失败: {e}")

        try:
            today_str = date.today().strftime("%Y%m%d")
            zt_df = ak.stock_zt_pool_em(date=today_str)
            if zt_df is not None and not zt_df.empty:
                inputs["limit_up"] = len(zt_df)
                dt_df = ak.stock_zt_pool_dtgc_em(date=today_str)
                inputs["limit_down"] = len(dt_df) if dt_df is not None else 0
                inputs["zt_fetched"] = True
        except Exception as e:
            logger.warning(f"获取涨跌停数据失败: {e}")

    try:
        market_stats = fm.get_market_stats() if fm is not None else None
        if market_stats:
            inputs["total_amount_yi"] = market_stats.get('total_amount', 0) or 0.0
    except Exception as e:
        logger.warning(f"获取两市成交额失败: {e}")

    return inputs


def check_market_gate(gate_inputs: Dict[str, Any]) -> Tuple[bool, Dict[str, bool], str, str, bool]:
    """
    市场环境门控 + 硬拦截（纯判定；外部数据由入口层 fetch_gate_inputs 取好传入）

    先执行硬拦截，再执行 4 项门控检查。

    Args:
        gate_inputs: fetch_gate_inputs() 的返回

    Returns:
        can_trade           — 是否允许开仓
        conditions_dict     — 各门控项通过情况
        summary_str         — 人类可读的检查摘要
        regime              — 市场状态：trending_up | trending_down | sideways | weak_up | chaos
        hard_intercept      — 是否触发了硬拦截
    """
    conditions: Dict[str, bool] = {
        "上证收盘>MA20": False,
        "两市成交额≥1.5万亿": False,
        "成交量>近20日均量": False,
        "涨停≥30且>跌停×1.5": False,
    }
    met_count = 0
    details: List[str] = []
    index_df = gate_inputs.get("index_df")
    total_amount_yi = gate_inputs.get("total_amount_yi", 0.0)
    limit_up = gate_inputs.get("limit_up", 0)
    limit_down = gate_inputs.get("limit_down", 0)

    try:
        if index_df is not None and len(index_df) >= 20:
            index_df = index_df.copy()
            index_df['ma20'] = index_df['close'].rolling(window=20).mean()
            latest = index_df.iloc[-1]
            idx_close = latest['close']
            idx_ma20 = latest['ma20']

            # ① 上证 > MA20
            if pd.notna(idx_ma20) and idx_close > idx_ma20:
                conditions["上证收盘>MA20"] = True
                met_count += 1
                details.append(f"✅ 上证{idx_close:.0f} > MA20{idx_ma20:.0f}")
            else:
                details.append(f"❌ 上证{idx_close:.0f} ≤ MA20{idx_ma20:.0f}")

            # ②a 两市成交额 ≥ 1.5 万亿（数据来自 fetch_gate_inputs）
            if total_amount_yi >= 15000:
                conditions["两市成交额≥1.5万亿"] = True
                met_count += 1
                details.append(f"✅ 两市成交额{total_amount_yi:.0f}亿 ≥ 1.5万亿")
            elif total_amount_yi > 0:
                details.append(f"❌ 两市成交额{total_amount_yi:.0f}亿 < 1.5万亿")
            else:
                details.append("⚠️ 获取两市成交额失败，跳过此项")

            # ②b 成交量 > 近20日均量（用成交量同比比较，单位一致无需转换）
            if 'volume' in index_df.columns:
                latest_vol = latest.get('volume', 0)
                avg_vol = index_df['volume'].iloc[-20:].mean()
                if pd.notna(avg_vol) and avg_vol > 0 and latest_vol > avg_vol:
                    conditions["成交量>近20日均量"] = True
                    met_count += 1
                    details.append(f"✅ 沪市成交量{latest_vol/1e8:.1f}亿股 > 20日均量{avg_vol/1e8:.1f}亿股")
                else:
                    details.append(f"❌ 沪市成交量{latest_vol/1e8:.1f}亿股 ≤ 20日均量{avg_vol/1e8:.1f}亿股")

        # ── ③ 涨停 ≥ 30 且涨停 > 跌停 × 1.5（数据来自 fetch_gate_inputs）──
        if gate_inputs.get("zt_fetched"):
            up_ok = limit_up >= 30
            ratio_ok = limit_up > limit_down * 1.5
            if up_ok and ratio_ok:
                conditions["涨停≥30且>跌停×1.5"] = True
                met_count += 1
                details.append(f"✅ 涨停{limit_up}家(≥30) > 跌停{limit_down}家")
            elif not up_ok:
                details.append(f"❌ 涨停{limit_up}家 < 30（数量不足）")
            else:
                details.append(f"❌ 涨停{limit_up}家 ≤ 跌停{limit_down}家×1.5（比值不足）")
        else:
            details.append("⚠️ 获取涨跌停数据失败，跳过此项")

    except Exception as e:
        logger.error(f"市场门控检查失败: {e}")
        return True, conditions, "市场环境检查失败（默认放行）", "chaos", False

    # ── 硬拦截检查 ──
    is_hard, hard_reason = _check_hard_intercept(
        index_df, total_amount_yi, limit_up, limit_down
    )
    if is_hard:
        logger.warning(hard_reason)
        summary = f"🔴 **硬拦截触发**\n{hard_reason}\n\n" + "\n".join(details)
        return False, conditions, summary, "chaos", True

    # ── 门控判定 ──
    # 先判定市场状态（均线结构），再结合门控项数决定是否开仓
    regime = _detect_regime(index_df, met_count)

    if regime == "trending_down":
        # 均线空头时无论门控通过多少项都不开仓：
        # 空头结构下"高成交+高情绪"组合是下跌中继/放量出货的典型特征，不是反转信号。
        # 趋势策略坚持"底部偏右进场"，等收盘重回 MA20（weak_up/trending_up）再参与。
        can_trade = False
        details.append("📉 均线空头排列，无论门控通过多少项均禁止开仓（等收盘重回MA20）")
    elif regime not in REGIME_CAN_OPEN:
        # chaos（方向不明，多为收盘在 MA20 下方但尚未形成空头排列）：
        # 不启用开仓档位，禁止开新仓；持仓卖出照常按正常版输出。
        can_trade = False
        details.append(f"🌪️ 状态{regime}：不启用开仓档位，禁止开新仓（等方向明确）")
    else:
        can_trade = met_count >= 2

    env_icon = "✅ 允许开仓" if can_trade else "❌ 建议空仓"
    summary = (
        f"市场环境检查：满足{met_count}/4项条件 → {env_icon}\n"
        + "\n".join(details)
    )

    if can_trade:
        logger.info(f"✅ 市场环境满足开仓条件（{met_count}/4，状态: {regime}）")
    else:
        logger.warning(f"⛔ 市场环境不满足开仓条件（仅{met_count}/4，状态: {regime}）")

    return can_trade, conditions, summary, regime, False
