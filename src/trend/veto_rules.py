# -*- coding: utf-8 -*-
"""
趋势波段策略 — 负面清单（硬否决层）

定位：在"剔除规则（removal_rules，趋势破坏）"之后、"信号检测（signal_detector，
买点 + 质量门）"之前的一道准入闸门。任一规则触发即否决：不进信号池、不看评分。

与相邻层的职责分工：
- removal_rules：持仓/观察池的"趋势破坏"剔除（跌破10日线、放量长阴等），
  关注"已经持有的票要不要扔"。
- veto_rules（本模块）：极端风险/情绪过热标的的"准入否决"，
  关注"再便宜也不能买"。
- signal_detector：买点形态与信号质量门（is_euphoric / is_overextended）。

规则清单（9 条，任一触发即否决）：
    V1 [外部] 近 20 日发布过股票交易异常波动 / 风险提示公告        → skip
    V2 [行情] 近 60 日累计涨幅 > 100%                              → remove
    V3 [行情] 近 20 日换手率均值 > 12%                             → skip
    V4 [行情] 近 20 日出现 ≥2 次单日跌幅 > 7%                      → skip
    V5 [外部] 近 5 日主力资金净流出 > 流通市值 1%                   → skip
    V6 [行情] 近 20 日涨停或跌停天数 ≥ 3                           → remove
    V7 [行情] 距近 60 日最低收盘价的涨幅 > 80%                      → remove
    V8 [行情] 反弹逼近前高（曾深跌≥15% 且当前距 60 日前高 < 10%）   → skip
    V9 [行情] 近 20 日日收益率标准差 > 5%（波动率过大）             → skip

动作语义：
- remove：极端过热类，从妙想自选池剔除（与 removal_rules 处置一致）。
- skip  ：暂时性风险类（公告、资金流、换手、大跌、压力位），只跳过当日信号，
          保留在自选池，后续可能重新合格。

设计约定：
- 行情类规则（V2/V3/V4/V6/V7/V8/V9）纯本地计算，无额外 I/O，可对全池逐股执行。
- 外部数据规则（V1/V5）依赖妙想 API 且有日调用限额，只对"已产出信号"的候选
  惰性执行；数据缺失或解析失败一律 fail-open（记 warning 放行），避免外部
  数据源抖动导致整个系统静默黑屏。
- 换手率列缺失时跳过依赖换手的规则并发 warning（与 removal_rules 同一约定）。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# === 动作类型 ===
ACTION_SKIP = "skip"      # 仅跳过当日信号，保留自选池
ACTION_REMOVE = "remove"  # 从自选池剔除

ACTION_LABELS = {
    ACTION_SKIP: "跳过信号",
    ACTION_REMOVE: "剔除自选池",
}

# === 规则阈值（集中配置，改这里即改策略）===
ANNOUNCEMENT_LOOKBACK_DAYS = 20   # V1 公告回溯天数
ANNOUNCEMENT_KEYWORDS = ("异常波动", "风险提示", "交易风险", "停牌核查")

GAIN_60D_MAX = 100.0              # V2 近60日累计涨幅上限(%)
TURNOVER_20D_MAX = 12.0           # V3 近20日换手率均值上限(%)
BIG_DROP_PCT = -7.0               # V4 单日跌幅阈值(%)
BIG_DROP_MIN_COUNT = 2            # V4 触发所需次数
FUND_FLOW_DAYS = 5                # V5 主力资金统计天数
FUND_FLOW_OUTFLOW_PCT = 1.0       # V5 净流出占流通市值比例阈值(%)
FUND_FLOW_SANITY_PCT = 50.0       # V5 单位合理性上限：净流出超流通市值50%视为单位存疑，放行
LIMIT_MOVE_DAYS = 3               # V6 涨跌停天数阈值
LIMIT_MOVE_LOOKBACK = 20          # V6 回溯天数
FROM_60D_LOW_MAX = 80.0           # V7 距60日最低收盘涨幅上限(%)
NEAR_HIGH_PCT = 10.0              # V8 距60日前高的距离阈值(%)
PULLBACK_DEPTH_RATIO = 0.85       # V8 判定"曾深跌"的回撤比例（最低收盘 ≤ 最高收盘×0.85）
HIGH_NOT_RECENT_BARS = 3          # V8 前高须形成于≥3个交易日前（排除正在创新高的主升股）
VOLATILITY_20D_MAX = 5.0           # V9 近20日日收益率标准差上限(%)，博弈激烈、波动过大

# 计算 60 日规则所需的最少交易日数（今日 + 60 个交易日前的基准）
BARS_FOR_60D = 61


@dataclass
class VetoResult:
    """否决结果。

    Attributes:
        vetoed: 是否被否决
        reasons: 触发的规则描述列表
        action: 'skip'（跳过当日信号）或 'remove'（剔除自选池）；
                多条同时触发时取最重动作（remove > skip）
    """

    vetoed: bool = False
    reasons: List[str] = field(default_factory=list)
    action: str = ACTION_SKIP

    def add(self, reason: str, action: str = ACTION_SKIP) -> None:
        """记录一条触发的规则（remove 优先级高于 skip）。"""
        self.vetoed = True
        self.reasons.append(reason)
        if action == ACTION_REMOVE:
            self.action = ACTION_REMOVE


# ==================== 行情类规则（V2/V3/V4/V6/V7/V8/V9）====================

def _limit_move_threshold(code: str) -> float:
    """涨跌停判定阈值(%)。

    创业板/科创板涨跌幅限制 20%，其余 10%；取略低于实际限制的值以覆盖
    9.9%/19.9% 这类四舍五入的封板情况。ST（5%）与北交所（30%）未接入，
    选股池默认已排除。
    """
    if code.startswith(("300", "301", "688", "689")):
        return 19.5
    return 9.5


def check_market_veto(code: str, name: str, df: Optional[pd.DataFrame]) -> VetoResult:
    """负面清单 — 行情类规则（V2/V3/V4/V6/V7/V8/V9），纯本地计算，无外部 I/O。

    Args:
        code: 股票代码
        name: 股票名称（仅用于日志）
        df: 日线 DataFrame，需含 close / turnover_rate(可选) / date，按日期升序

    Returns:
        VetoResult；未触发时 vetoed=False
    """
    result = VetoResult()
    if df is None or len(df) < 2:
        return result

    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    tag = f"{name}({code})"
    closes = df['close'].astype(float)
    pct = closes.pct_change() * 100  # 日涨跌幅(%)，首行为 NaN

    # --- V3 近20日换手率均值 > 12%（换手率缺失时跳过，与 removal_rules 同约定）---
    if 'turnover_rate' in df.columns and n >= 20:
        tr_20 = pd.to_numeric(df['turnover_rate'].iloc[-20:], errors='coerce')
        tr_mean = tr_20.mean()
        if pd.notna(tr_mean) and tr_mean > TURNOVER_20D_MAX:
            result.add(
                f"V3 近20日换手均值{tr_mean:.1f}%>{TURNOVER_20D_MAX}%",
                ACTION_SKIP,
            )
    elif 'turnover_rate' not in df.columns:
        logger.warning(f"  {tag}: 换手率列缺失，负面清单 V3（高换手）跳过")

    # --- V4 近20日 ≥2 次单日跌幅 > 7% ---
    if n >= 20:
        big_drops = int((pct.iloc[-20:] <= BIG_DROP_PCT).sum())
        if big_drops >= BIG_DROP_MIN_COUNT:
            result.add(
                f"V4 近20日{big_drops}次单日跌幅>7%",
                ACTION_SKIP,
            )

    # --- V6 近20日涨停或跌停天数 ≥ 3 ---
    if n >= LIMIT_MOVE_LOOKBACK + 1:
        threshold = _limit_move_threshold(code)
        limit_days = int((pct.iloc[-LIMIT_MOVE_LOOKBACK:].abs() >= threshold).sum())
        if limit_days >= LIMIT_MOVE_DAYS:
            result.add(
                f"V6 近{LIMIT_MOVE_LOOKBACK}日涨跌停{limit_days}天≥{LIMIT_MOVE_DAYS}天",
                ACTION_REMOVE,
            )

    # --- V9 近20日日收益率标准差 > 5%（波动率过大，博弈激烈，不适合低吸）---
    if n >= 20:
        vol_std = float(pct.iloc[-20:].std())
        if pd.notna(vol_std) and vol_std > VOLATILITY_20D_MAX:
            result.add(
                f"V9 近20日日收益率标准差{vol_std:.1f}%>{VOLATILITY_20D_MAX}%",
                ACTION_SKIP,
            )

    # --- 60 日规则：需要至少 61 个交易日 ---
    if n < BARS_FOR_60D:
        logger.warning(
            f"  {tag}: 数据仅{n}条(<{BARS_FOR_60D})，60日类否决规则(V2/V7/V8)跳过"
        )
        return result

    window = closes.iloc[-60:]
    last_close = float(closes.iloc[-1])
    high_60 = float(window.max())
    low_60 = float(window.min())

    # --- V2 近60日累计涨幅 > 100% ---
    base_close = float(closes.iloc[-BARS_FOR_60D])
    if base_close > 0:
        gain_60d = (last_close - base_close) / base_close * 100
        if gain_60d > GAIN_60D_MAX:
            result.add(
                f"V2 近60日累计涨幅{gain_60d:.1f}%>{GAIN_60D_MAX}%",
                ACTION_REMOVE,
            )

    # --- V7 距近60日最低收盘价的涨幅 > 80% ---
    if low_60 > 0:
        from_low = (last_close - low_60) / low_60 * 100
        if from_low > FROM_60D_LOW_MAX:
            result.add(
                f"V7 距60日最低收盘涨幅{from_low:.1f}%>{FROM_60D_LOW_MAX}%",
                ACTION_REMOVE,
            )

    # --- V8 反弹逼近前高压力位 ---
    # 三个条件同时满足才算"反弹回前高"（持续创新高的主升股不受影响）：
    #   1) 60 日最高收盘形成于 ≥3 个交易日前（当前并未创新高）
    #   2) 当前收盘距该前高 < 10%
    #   3) 期间曾深跌：60 日最低收盘 ≤ 最高收盘 × 0.85
    if high_60 > 0:
        bars_from_high = int(len(window) - 1 - int(window.values.argmax()))
        dist_to_high = (high_60 - last_close) / high_60 * 100
        if (
            bars_from_high >= HIGH_NOT_RECENT_BARS
            and 0 <= dist_to_high < NEAR_HIGH_PCT
            and low_60 <= high_60 * PULLBACK_DEPTH_RATIO
        ):
            result.add(
                f"V8 反弹逼近{bars_from_high}日前高点(距{dist_to_high:.1f}%<{NEAR_HIGH_PCT}%)，"
                f"期间最大回撤{(1 - low_60 / high_60) * 100:.1f}%",
                ACTION_SKIP,
            )

    if result.vetoed:
        logger.info(f"🚫 负面清单否决 {tag}: {'；'.join(result.reasons)} → {ACTION_LABELS[result.action]}")

    return result


# ==================== 外部数据规则（V1 公告 / V5 主力资金）====================

def _iter_dicts(obj: Any) -> Iterator[Dict[str, Any]]:
    """深度遍历嵌套结构，产出其中的所有 dict（用于防御式解析妙想响应）。"""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_dicts(item)


def _to_float(value: Any) -> Optional[float]:
    """宽松转 float，失败返回 None。"""
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def _to_yuan(value: float, column_name: str) -> float:
    """按列名中的单位提示把数值换算成元（默认视为元）。"""
    if "亿" in column_name:
        return value * 1e8
    if "万" in column_name:
        return value * 1e4
    return value


def _parse_datetime(text: str) -> Optional[datetime]:
    """解析常见日期字符串，失败返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt) + 2], fmt)
        except ValueError:
            continue
    return None


def _extract_series(resp: Optional[Dict[str, Any]], keywords: Tuple[str, ...]) -> Tuple[str, List[float]]:
    """从妙想查数响应中提取指标序列。

    Args:
        resp: query_financial_data 的原始响应
        keywords: 指标中文名需同时包含的关键词（如 ('主力', '净流入')）

    Returns:
        (匹配到的指标名, 数值列表)；未匹配到则返回 ("", [])
    """
    if not resp:
        return "", []

    for table_obj in _iter_dicts(resp):
        table = table_obj.get("table")
        name_map = table_obj.get("nameMap")
        if not isinstance(table, dict) or not isinstance(name_map, dict):
            continue
        for key, values in table.items():
            if key == "headName" or not isinstance(values, (list, tuple)):
                continue
            column_name = str(name_map.get(key, key))
            if not all(kw in column_name for kw in keywords):
                continue
            nums = [v for v in (_to_float(x) for x in values) if v is not None]
            if nums:
                return column_name, nums
    return "", []


def _extract_news_items(resp: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """从妙想资讯搜索响应中提取 {title, date} 列表（防御式）。"""
    items: List[Dict[str, str]] = []
    if not resp:
        return items

    for obj in _iter_dicts(resp):
        title_key = next(
            (k for k in obj if "title" in str(k).lower() or "标题" in str(k)), None
        )
        if not title_key or not isinstance(obj[title_key], (str, int, float)):
            continue
        date_key = next(
            (
                k
                for k in obj
                if any(t in str(k).lower() for t in ("date", "time", "publish"))
                or "日期" in str(k)
                or "时间" in str(k)
            ),
            None,
        )
        items.append(
            {
                "title": str(obj[title_key]),
                "date": str(obj.get(date_key, "")) if date_key else "",
            }
        )
    return items


def _check_announcement_veto(code: str, name: str, mx_service: Any) -> Optional[str]:
    """V1：近 20 日发布过异常波动 / 风险提示公告 → 返回触发原因，否则 None。"""
    if mx_service is None:
        return None

    try:
        resp = mx_service.search_news(
            f"{name}({code}) 最近一个月 股票交易异常波动公告 风险提示公告"
        )
        items = _extract_news_items(resp)
    except Exception as e:
        logger.warning(f"  {name}({code}): V1 公告检索失败，放行（{e}）")
        return None

    if not items:
        logger.debug(f"  {name}({code}): V1 未检索到公告条目")
        return None

    cutoff = datetime.now() - timedelta(days=ANNOUNCEMENT_LOOKBACK_DAYS)
    for item in items:
        title = item.get("title", "")
        if not any(kw in title for kw in ANNOUNCEMENT_KEYWORDS):
            continue
        published = _parse_datetime(item.get("date", ""))
        # 日期无法解析时按"较新"处理（保守方向：宁可错杀）
        if published is None or published >= cutoff:
            return f"V1 近{ANNOUNCEMENT_LOOKBACK_DAYS}日风险公告《{title[:24]}》"
    return None


def fetch_main_net_inflow(code: str, name: str, mx_service: Any,
                          days: int = FUND_FLOW_DAYS) -> Optional[float]:
    """近 N 日主力资金净流入合计（元）。

    供 V5（净流出否决）与开仓档位「收紧档资金确认」共用——两者口径必须一致，
    否则会出现"V5 判为流出、档位判为流入"的自相矛盾。

    Returns:
        净流入合计（元，负数=净流出）；取不到或解析失败返回 None（fail-open）
    """
    if mx_service is None:
        return None

    tag = f"{name}({code})"
    try:
        resp = mx_service.query_financial_data(f"{name}({code}) 近{days}个交易日 每日主力净流入")
        column_name, values = _extract_series(resp, keywords=("主力", "净流入"))
    except Exception as e:
        logger.warning(f"  {tag}: 主力资金查询失败，返回 None（{e}）")
        return None

    if not values:
        logger.warning(f"  {tag}: 未取到主力净流入序列，返回 None")
        return None

    recent = values[-days:]
    recent = [_to_yuan(v, column_name) for v in recent]
    return sum(recent)


def _check_fund_flow_veto(code: str, name: str, mx_service: Any, fetcher: Any) -> Optional[str]:
    """V5：近 5 日主力资金净流出 > 流通市值 1% → 返回触发原因，否则 None。"""
    if mx_service is None or fetcher is None:
        return None

    tag = f"{name}({code})"

    net_flow = fetch_main_net_inflow(code, name, mx_service, days=FUND_FLOW_DAYS)
    if net_flow is None:
        logger.warning(f"  {tag}: V5 主力资金数据缺失，放行")
        return None
    if net_flow >= 0:
        return None

    try:
        quote = fetcher.get_realtime_quote(code)
    except Exception as e:
        logger.warning(f"  {tag}: V5 流通市值获取失败，放行（{e}）")
        return None

    circ_mv = getattr(quote, "circ_mv", None) if quote is not None else None
    if not circ_mv or circ_mv <= 0:
        logger.warning(f"  {tag}: V5 流通市值缺失，放行")
        return None

    outflow = abs(net_flow)
    ratio = outflow / circ_mv * 100
    if ratio > FUND_FLOW_SANITY_PCT:
        # 净流出不可能超过流通市值的一半，多半是单位换算问题 → 放行并告警
        logger.warning(
            f"  {tag}: V5 净流出占流通市值{ratio:.1f}%，疑似单位异常，放行"
        )
        return None

    if ratio > FUND_FLOW_OUTFLOW_PCT:
        return (
            f"V5 近{len(recent)}日主力净流出{outflow / 1e8:.2f}亿，"
            f"占流通市值{ratio:.2f}%>{FUND_FLOW_OUTFLOW_PCT}%"
        )
    return None


def check_external_veto(
    code: str,
    name: str,
    mx_service: Any = None,
    fetcher: Any = None,
) -> VetoResult:
    """负面清单 — 外部数据规则（V1 公告 / V5 主力资金）。

    依赖妙想 API 且有日调用限额，只对已产出信号的候选惰性调用。
    数据缺失或解析失败一律 fail-open（放行）并记 warning。

    Args:
        code: 股票代码
        name: 股票名称
        mx_service: MXService 实例（None 时跳过全部外部规则）
        fetcher: DataFetcherManager 实例，用于取流通市值

    Returns:
        VetoResult；外部规则动作均为 skip（暂时性风险，不剔除自选池）
    """
    result = VetoResult()

    reason = _check_announcement_veto(code, name, mx_service)
    if reason:
        result.add(reason, ACTION_SKIP)

    reason = _check_fund_flow_veto(code, name, mx_service, fetcher)
    if reason:
        result.add(reason, ACTION_SKIP)

    if result.vetoed:
        logger.info(f"🚫 负面清单否决 {name}({code}): {'；'.join(result.reasons)} → {ACTION_LABELS[result.action]}")

    return result
