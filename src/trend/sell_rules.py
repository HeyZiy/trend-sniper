# -*- coding: utf-8 -*-
"""
===================================
趋势策略 — 持仓卖出信号检测
===================================

实现 strategy/trend_strategy.md「卖出」设计稿（正常版 + trending_down 收紧版）。

第一卖点（减仓50%，满足其一）：
  - 放量跌破5日线（量比≥2 且 收盘<MA5）
  - 高位长阴吞没（前阳被当日阴线吞没 + 跌幅≥3% + 高位）
  - 从阶段高点回撤≥5%（阶段高点=近20日最高收盘，不含当日）
  - 板块明显走弱（所属行业板块当日跌幅≤-2%，数据缺失跳过）
第二卖点（全仓清仓，满足其一）：
  - 连续2日收盘跌破10日线
  - 放量跌破10日线（量比≥2 且 收盘<MA10）
  - 主线明显退潮（近似：板块当日跌幅≤-3%）
  - 个股跌破关键平台（近20日最低收盘，不含当日）
止盈保护（建议减半）：盈利≥15% 且 放量滞涨（量比≥2 且 涨幅<2% 或 收盘位于日内下半部）

trending_down 收紧版：MA10 破位 1 天 / 回撤≥3% / 量比≥1.5 / 板块走弱阈值 -1%。
硬拦截：全部清仓。
sideways/chaos：用**正常版**并照常输出卖出信号——「自然退出」指不收紧阈值，
不是不执行，正常版的清仓规则在 sideways/chaos 下照常生效。

持仓事实来源：妙想模拟仓（用户手动同步持仓）。

执行方式：由尾盘任务 trend_sell.py 在每交易日 14:45 后读取本模块信号，
自动下模拟仓市价单（reduce_half / clear / 硬拦截全清 全部自动执行）。
本模块只负责判定，不碰下单；下单与股数收敛见 trend_sell.py:execute_sell()。

量比口径：当日成交量 ÷ 前 5 日均量（不含当日）。

板块降级约定：
- 板块取不到 → 降级为「未知板块」（UNKNOWN_SECTOR），不用空串/None；
- 板块行情缺失 → sector_pct 为 None；
- 两者任一成立时板块类规则（板块走弱 / 主线退潮）跳过，并由
  HoldingRow.sector_skipped 把"判不了"这一事实带到报告层显式标注，
  避免用户把"无卖出信号"误读成"板块没走弱"。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.mx.position_utils import position_profit_pct
from src.trend.signal_detector import UNKNOWN_SECTOR, SignalFieldError

logger = logging.getLogger(__name__)

# 卖出动作白名单（渲染层文案与判定分支共用）
VALID_ACTIONS = ("reduce_half", "clear")

# ── 正常版阈值 ──
VOL_RATIO_NORMAL = 2.0        # 放量：当日量/前5日均量 ≥ 2.0
VOL_RATIO_TIGHT = 1.5         # trending_down 收紧版量比
DRAWDOWN_NORMAL = 5.0         # 阶段高点回撤减仓阈值（%）
DRAWDOWN_TIGHT = 3.0          # trending_down 收紧版回撤
SECTOR_WEAK_NORMAL = -2.0     # 板块明显走弱：当日跌幅 ≤ -2%
SECTOR_WEAK_TIGHT = -1.0      # trending_down 收紧版
SECTOR_TIDE_OUT = -3.0        # 主线明显退潮（近似）：当日跌幅 ≤ -3%
LONG_YIN_PCT = -3.0           # 长阴：当日跌幅 ≤ -3%
HIGH_5D_GAIN = 10.0           # 高位判定：近5日累计涨幅 ≥ 10%
HIGH_NEAR_PEAK = 0.97         # 高位判定：收盘 ≥ 近20日高点 × 0.97
PEAK_WINDOW = 20              # 阶段高点/关键平台回看窗口（交易日，不含当日）
TP_PROFIT_PCT = 15.0          # 止盈保护盈利线（%）
TP_STALL_GAIN = 2.0           # 放量滞涨：当日涨幅 < 2%
TP_CLOSE_POS = 0.4            # 放量滞涨：收盘位于日内区间下 40%

# 减半后剩余仓位的操作提醒（用户主动执行，不做阶段跟踪）
REDUCE_NOTE = "减仓后剩余仓位：止损线上移至10日线，收盘跌破即清仓"


@dataclass
class SellSignal:
    """持仓卖出信号（只出建议，不执行交易）。

    构造即校验：code/name/action/reasons 为必填事实，缺失或非法直接抛异常，
    不允许带着空壳对象流进渲染层。

    Attributes:
        sector: 所属板块名称，取不到时为 UNKNOWN_SECTOR
        sector_pct: 板块当日涨跌幅（None=无法判断，板块类规则跳过）
    """
    # ── 必填（漏传即 TypeError）──
    code: str
    name: str
    action: str                      # 'reduce_half' | 'clear'
    reasons: List[str]
    current_price: float
    cost_price: float
    profit_pct: float
    count: int                       # 当前持仓股数
    suggest_shares: int              # 建议卖出股数（100 整数倍）
    # ── 可选（有默认值）──
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    entry_date: str = ""             # 买入日期（历史委托推导，可能为空）
    pct_change: float = 0.0
    vol_ratio: float = 0.0
    stage_high: float = 0.0          # 阶段高点（近20日最高收盘）
    platform_low: float = 0.0        # 关键平台（近20日最低收盘）
    sector: str = UNKNOWN_SECTOR     # 所属板块名称
    sector_pct: Optional[float] = None   # 板块当日涨跌幅
    note: str = ""                   # 附加提醒

    def __post_init__(self) -> None:
        if not self.code or not self.name:
            raise SignalFieldError(f"卖出信号缺少 code/name: {self.code!r}/{self.name!r}")
        if self.action not in VALID_ACTIONS:
            raise SignalFieldError(
                f"{self.name}({self.code}): 未知卖出动作 {self.action!r}，须为 {VALID_ACTIONS}"
            )
        if not self.reasons:
            # 没有触发原因的卖出信号对用户毫无意义，多半是规则漏填
            raise SignalFieldError(f"{self.name}({self.code}): 卖出信号缺少触发原因")
        if self.current_price <= 0:
            raise SignalFieldError(f"{self.name}({self.code}): current_price={self.current_price} 非正数")
        if self.count < 0 or self.suggest_shares < 0:
            raise SignalFieldError(
                f"{self.name}({self.code}): 股数为负（count={self.count} suggest={self.suggest_shares}）"
            )


@dataclass
class HoldingRow:
    """持仓行 —— 日报「持仓卖出信号」板块的渲染输入（一只持仓一行）。

    替代原 (position, signal, sector, sector_pct) 元组：元组按下标取值，
    加字段时容易取错位置且无校验。

    Attributes:
        position: 妙想持仓接口返回的标准化 dict
        signal: 卖出信号，None = 无信号（继续持有）
        sector: 所属板块名称，取不到时为 UNKNOWN_SECTOR
        sector_pct: 板块当日涨跌幅（None=无法判断）
        sector_skipped: 板块类规则是否已跳过（板块未知 或 板块行情缺失）。
            用于区分「板块没走弱」与「板块走弱判不了」——
            用户在报告里看到的"无卖出信号"，含义完全不同。
    """
    position: dict
    signal: Optional[SellSignal] = None
    sector: str = UNKNOWN_SECTOR
    sector_pct: Optional[float] = None
    sector_skipped: bool = False

    @property
    def code(self) -> str:
        return str(self.position.get("code", "") or "")

    @property
    def name(self) -> str:
        return str(self.position.get("name", "") or self.code)

    @property
    def profit_pct(self) -> float:
        return position_profit_pct(self.position)


def _compute_metrics(df: pd.DataFrame) -> Optional[dict]:
    """从日线计算卖出规则所需指标。df 需已按日期排序并含 MA5/MA10/MA20。"""
    if df is None or len(df) < 6:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest["close"])
    ma5 = float(latest["ma5"]) if pd.notna(latest.get("ma5")) else 0.0
    ma10 = float(latest["ma10"]) if pd.notna(latest.get("ma10")) else 0.0
    ma20 = float(latest["ma20"]) if pd.notna(latest.get("ma20")) else 0.0
    if ma5 <= 0 or ma10 <= 0:
        return None

    prev_close = float(prev["close"]) if pd.notna(prev.get("close")) else 0.0
    pct_change = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

    vol = float(latest.get("volume", 0) or 0)
    vol_ma5 = float(df["volume"].iloc[-6:-1].mean()) if "volume" in df.columns else 0.0
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 0.0

    window = df.iloc[-PEAK_WINDOW - 1:-1]
    stage_high = float(window["close"].max()) if not window.empty else close
    platform_low = float(window["close"].min()) if not window.empty else close

    gain5 = 0.0
    if len(df) >= 7:
        base = float(df.iloc[-6]["close"])
        if base > 0:
            gain5 = (close - base) / base * 100

    high = float(latest.get("high", close) or close)
    low = float(latest.get("low", close) or close)
    close_position = (close - low) / (high - low) if high > low else 0.5

    return {
        "close": close,
        "open": float(latest.get("open", close) or close),
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "prev_close": prev_close,
        "prev_open": float(prev["open"]) if pd.notna(prev.get("open")) else prev_close,
        "prev_ma10": float(prev["ma10"]) if pd.notna(prev.get("ma10")) else 0.0,
        "pct_change": pct_change,
        "vol_ratio": vol_ratio,
        "stage_high": stage_high,
        "platform_low": platform_low,
        "gain5": gain5,
        "close_position": close_position,
    }


def _check_rules(metrics: dict, profit_pct: float, regime: str,
                 sector_pct: Optional[float], hard_intercept: bool) -> Tuple[str, List[str], str]:
    """逐条检查卖出规则。

    Returns:
        (action, reasons, note)
        action: 'clear' | 'reduce_half' | ''（空=无信号）
    """
    if hard_intercept:
        return "clear", ["市场门控硬拦截，全部清仓"], ""

    # 只有 trending_down 收紧；sideways/weak_up/chaos 一律走正常版并照常输出信号
    tight = regime == "trending_down"
    vol_thr = VOL_RATIO_TIGHT if tight else VOL_RATIO_NORMAL
    dd_thr = DRAWDOWN_TIGHT if tight else DRAWDOWN_NORMAL
    sector_thr = SECTOR_WEAK_TIGHT if tight else SECTOR_WEAK_NORMAL

    close = metrics["close"]
    ma5, ma10 = metrics["ma5"], metrics["ma10"]
    vol_ratio = metrics["vol_ratio"]
    pct_change = metrics["pct_change"]
    tag = "（收紧版）" if tight else ""

    clear_reasons: List[str] = []
    reduce_reasons: List[str] = []

    # ── 第二卖点（全仓清仓）──

    # 连续收盘跌破 MA10（收紧版缩为 1 天）
    below_ma10_today = close < ma10
    below_ma10_yesterday = (
        metrics["prev_close"] < metrics["prev_ma10"] if metrics["prev_ma10"] > 0 else False
    )
    if tight and below_ma10_today:
        clear_reasons.append(f"收盘跌破MA10{tag}（今{close:.2f}<MA10 {ma10:.2f}）")
    elif not tight and below_ma10_today and below_ma10_yesterday:
        clear_reasons.append(
            f"连续2日收盘跌破MA10（昨{metrics['prev_close']:.2f}，今{close:.2f}<{ma10:.2f}）"
        )

    # 放量跌破 MA10
    if below_ma10_today and vol_ratio >= vol_thr:
        clear_reasons.append(f"放量跌破MA10{tag}（量比{vol_ratio:.1f}≥{vol_thr}，收盘{close:.2f}<{ma10:.2f}）")

    # 主线明显退潮（近似：板块当日跌幅 ≤ -3%）
    if sector_pct is not None and sector_pct <= SECTOR_TIDE_OUT:
        clear_reasons.append(f"主线明显退潮（板块当日{sector_pct:+.1f}% ≤ {SECTOR_TIDE_OUT}%）")

    # 跌破关键平台（近20日最低收盘）
    if close < metrics["platform_low"]:
        clear_reasons.append(f"跌破关键平台（收盘{close:.2f} < 20日平台{metrics['platform_low']:.2f}）")

    # ── 第一卖点（减仓50%）──

    # 放量跌破 MA5
    if close < ma5 and vol_ratio >= vol_thr:
        reduce_reasons.append(f"放量跌破5日线{tag}（量比{vol_ratio:.1f}≥{vol_thr}，收盘{close:.2f}<MA5 {ma5:.2f}）")

    # 高位长阴吞没
    is_yin = close < metrics["open"]
    prev_yang = metrics["prev_close"] > metrics["prev_open"]
    engulfing = metrics["open"] >= metrics["prev_close"] and close <= metrics["prev_open"]
    is_high = metrics["gain5"] >= HIGH_5D_GAIN or close >= metrics["stage_high"] * HIGH_NEAR_PEAK
    if is_yin and prev_yang and engulfing and pct_change <= LONG_YIN_PCT and is_high:
        reduce_reasons.append(
            f"高位长阴吞没（跌幅{pct_change:+.1f}%，近5日{metrics['gain5']:+.1f}%）"
        )

    # 阶段高点回撤
    if metrics["stage_high"] > 0:
        drawdown = (close - metrics["stage_high"]) / metrics["stage_high"] * 100
        if drawdown <= -dd_thr:
            reduce_reasons.append(
                f"阶段高点回撤{drawdown:.1f}% ≥ {dd_thr}%{tag}（高点{metrics['stage_high']:.2f}→{close:.2f}）"
            )

    # 板块明显走弱
    if sector_pct is not None and sector_pct <= sector_thr:
        reduce_reasons.append(f"板块明显走弱{tag}（当日{sector_pct:+.1f}% ≤ {sector_thr}%）")

    # ── 止盈保护（建议减半，余仓按趋势持有）──
    stall = pct_change < TP_STALL_GAIN or metrics["close_position"] < TP_CLOSE_POS
    if profit_pct >= TP_PROFIT_PCT and vol_ratio >= vol_thr and stall:
        reduce_reasons.append(
            f"止盈保护：盈利{profit_pct:.1f}%≥{TP_PROFIT_PCT}% 且放量滞涨"
            f"（量比{vol_ratio:.1f}，涨幅{pct_change:+.1f}%）"
        )

    if clear_reasons:
        return "clear", clear_reasons, ""
    if reduce_reasons:
        return "reduce_half", reduce_reasons, REDUCE_NOTE
    return "", [], ""


def detect_sell_signals(code: str, name: str, df: pd.DataFrame, position: dict,
                        regime: str, hard_intercept: bool,
                        sector: str = UNKNOWN_SECTOR,
                        sector_pct: Optional[float] = None,
                        entry_date: str = "") -> Optional[SellSignal]:
    """检测单只持仓的卖出信号。

    Args:
        code/name: 股票代码与名称
        df: 已排序并计算 MA5/MA10/MA20 的日线 DataFrame
        position: 妙想持仓接口返回的标准化 dict（含 count/cost_price/profit_pct）
        regime: 市场状态（trending_up/weak_up/sideways/trending_down/chaos）
        hard_intercept: 市场门控硬拦截是否触发
        sector: 所属板块名称（取不到时为 UNKNOWN_SECTOR）
        sector_pct: 板块当日涨跌幅（None=无法判断，板块类规则跳过）
        entry_date: 买入日期（历史委托推导，可为空）

    Returns:
        SellSignal 或 None（无信号）。suggest_shares 已按可卖量收敛并取整到 100 整数倍，
        可能因可用不足而为 0（调用方按不足一手处理，不直接下单）。
    """
    metrics = _compute_metrics(df)
    if metrics is None:
        return None

    profit_pct = position_profit_pct(position)
    action, reasons, note = _check_rules(metrics, profit_pct, regime, sector_pct, hard_intercept)
    if not action:
        return None

    count = int(position.get("count", 0) or 0)
    if action == "reduce_half":
        half = (count // 200) * 100
        suggest = half if half >= 100 else count  # 不足一手半仓时建议直接清仓
    else:
        suggest = count

    # 收敛到可卖量：avail_count 可能小于总持仓（T+1 或已挂单），缺省回退总持仓。
    # 并向下取整到 100 整数倍 —— 妙想对非整手委托直接拒单。
    avail = int(position.get("avail_count", 0) or 0) or count
    suggest = (min(suggest, avail) // 100) * 100

    return SellSignal(
        code=code,
        name=name,
        action=action,
        reasons=reasons,
        current_price=metrics["close"],
        ma5=metrics["ma5"],
        ma10=metrics["ma10"],
        ma20=metrics["ma20"],
        count=count,
        suggest_shares=suggest,
        cost_price=float(position.get("cost_price", 0) or 0),
        profit_pct=profit_pct,
        entry_date=entry_date,
        pct_change=metrics["pct_change"],
        vol_ratio=metrics["vol_ratio"],
        stage_high=metrics["stage_high"],
        platform_low=metrics["platform_low"],
        sector=sector,
        sector_pct=sector_pct,
        note=note,
    )


def _df_to_pct_map(df, name_col: str, pct_col: str) -> Dict[str, float]:
    """把板块行情 DataFrame 转成 {板块名: 涨跌幅%} 字典。"""
    if df is None or df.empty or name_col not in df.columns or pct_col not in df.columns:
        return {}
    result: Dict[str, float] = {}
    for _, row in df.iterrows():
        pct = pd.to_numeric(row[pct_col], errors="coerce")
        if pd.notna(pct):
            result[str(row[name_col])] = float(pct)
    return result


def fetch_sector_pct_map() -> Dict[str, float]:
    """拉取全量行业板块当日涨跌幅 {板块名: 涨跌幅%}，失败返回空字典。

    多源回退：akshare 东财全量板块 → akshare 新浪 → efinance（东财实时），
    单一数据源不稳定不影响整体；全部失败时板块类卖出规则跳过。
    """
    from data_provider.fetchers.akshare_fetcher import AkshareFetcher
    from data_provider.fetchers.efinance_fetcher import EfinanceFetcher

    try:
        result = AkshareFetcher().get_sector_pct_map()
        if result:
            return result
    except Exception:
        logger.warning("akshare 板块行情获取失败，尝试 efinance", exc_info=True)

    try:
        ef = EfinanceFetcher()
        df = ef.get_sector_quotes()
        if df is None or df.empty:
            return {}
        name_col = "股票名称" if "股票名称" in df.columns else "name"
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
        return _df_to_pct_map(df, name_col, pct_col)
    except Exception:
        logger.warning("板块行情获取失败，板块类卖出规则跳过", exc_info=True)
        return {}


def match_sector_pct(board_name: str, pct_map: Dict[str, float]) -> Optional[float]:
    """按名称匹配板块当日涨跌幅；精确匹配失败时做包含匹配。

    板块未知或行情缺失时返回 None = 板块类规则判不了（不是"板块没走弱"）。
    """
    if not board_name or board_name == UNKNOWN_SECTOR or not pct_map:
        return None
    if board_name in pct_map:
        return pct_map[board_name]
    for k, v in pct_map.items():
        if board_name in k or k in board_name:
            return v
    return None
