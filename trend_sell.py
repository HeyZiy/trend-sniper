# -*- coding: utf-8 -*-
"""
===================================
趋势策略 — 尾盘卖出执行（每交易日 14:45 后）
===================================

每交易日 14:45 尾盘运行，承担「检测 → 执行 → 出报告 → 通知」闭环：

1. 读取妙想模拟仓股票持仓（持仓事实来源）
2. 复用 src/trend/sell_rules.py 的完全分类规则检测卖出信号
   （第一卖点减仓50% / 第二卖点清仓 / 止盈保护减半 / 门控硬拦截全清）
3. 命中即自动下模拟仓市价单（委托数量为 100 整数倍，按可用股数收敛）
4. 自行渲染成交报告并推送

执行约定：
- reduce_half（减仓50%）/ clear（清仓）/ 硬拦截全清 均自动下市价单，无需人工确认。
- 逐只隔离：单只取数或下单失败不影响其余持仓，最终汇总四类结果
  （success / failed / manual_skip / insufficient）。
- 不可交易标的（1 开头深市 ETF/LOF，见 src/mx/client.py）记为 manual_skip 单列提示。
- 下单前按 avail_count 收敛：可用股数可能小于总持仓（T+1 或已挂单），
  不足一手则跳过，避免废单。
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from data_provider import canonical_stock_code
from src.config import setup_env
from src.logging_config import setup_logging
from src.market_state.market_gate import check_market_gate, fetch_gate_inputs
from src.mx.client import MXMoniClient, is_mx_untradable
from src.mx.position_utils import (
    filter_stock_positions, get_last_buy_dates_safe, position_profit_pct,
)
from src.notify.service import NotificationService
from src.trend.sell_rules import (
    HoldingRow, SellSignal, detect_sell_signals, fetch_sector_pct_map, match_sector_pct,
)
from src.trend.signal_detector import UNKNOWN_SECTOR
setup_env()

logger = logging.getLogger(__name__)

# 执行结果状态
ST_SUCCESS = "success"          # 委托成功
ST_FAILED = "failed"            # 委托失败（含非交易时段）
ST_MANUAL = "manual_skip"       # 模拟仓不可交易，需用户手动
ST_INSUFFICIENT = "insufficient"  # 可用股数不足一手
ST_DRY_RUN = "dry_run"          # 试运行，未真正下单

ACTION_LABEL = {"reduce_half": "🟠 减仓50%", "clear": "🔴 清仓"}
STATUS_LABEL = {
    ST_SUCCESS: "✅ 已成交",
    ST_FAILED: "❌ 委托失败",
    ST_MANUAL: "⚠️ 需手动",
    ST_INSUFFICIENT: "➖ 不足一手",
    ST_DRY_RUN: "🔍 试运行",
}


@dataclass
class SellExecution:
    """单只持仓的卖出执行结果。

    组合而非复制：持仓/信号/板块事实复用 HoldingRow（sell_rules 定义的领域类型，
    sector_skipped 语义在那里定义），这里只追加执行侧字段。
    """
    row: HoldingRow                          # 持仓 + 卖出信号 + 板块判定事实
    shares: int = 0                          # 实际委托股数
    status: str = ""                         # 空 = 未执行（无信号）
    message: str = ""                        # 失败原因 / 备注
    data_ok: bool = True                     # 行情是否取到（False=判不了）

    @property
    def code(self) -> str:
        return self.row.code

    @property
    def name(self) -> str:
        return self.row.name

    @property
    def signal(self) -> Optional[SellSignal]:
        return self.row.signal

    @property
    def executed(self) -> bool:
        return self.status in (ST_SUCCESS, ST_FAILED, ST_DRY_RUN)


def _f(v) -> str:
    """价格格式化；不可用时返回 '-' 而不是抛异常。"""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "-"


def _f_pct(v) -> str:
    """百分比格式化。"""
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "-"


def _force_clear_signal(position: dict, sector: str,
                        sector_pct: Optional[float], reason: str) -> Optional[SellSignal]:
    """硬拦截兜底：行情缺失导致规则判不了时，仍按持仓全量构造清仓信号。

    硬拦截的业务语义是「当日应清仓所有持仓」，不该因为单只行情取不到而漏清。
    现价取不到时无法构造合法信号（SellSignal 校验 current_price>0），返回 None 由调用方标注。
    """
    count = int(position.get("count", 0) or 0)
    price = float(position.get("current_price", 0) or 0)
    if count <= 0 or price <= 0:
        return None
    return SellSignal(
        code=str(position.get("code", "") or ""),
        name=str(position.get("name", "") or "") or str(position.get("code", "") or ""),
        action="clear",
        reasons=[reason],
        current_price=price,
        cost_price=float(position.get("cost_price", 0) or 0),
        profit_pct=position_profit_pct(position),
        count=count,
        suggest_shares=count,
        sector=sector,
        sector_pct=sector_pct,
    )


def execute_sell(sig: SellSignal, position: dict, client: MXMoniClient,
                 dry_run: bool = False) -> Tuple[int, str, str]:
    """对单只持仓执行卖出委托。

    Args:
        sig: 卖出信号（suggest_shares 已为 100 整数倍）
        position: 妙想持仓 dict（取 avail_count 做可用量收敛）
        client: 妙想模拟仓客户端
        dry_run: True 时只计算不下单

    Returns:
        (shares, status, message)
    """
    code = sig.code

    # 妙想模拟仓无法识别 1 开头深市 ETF/LOF 的市场号，只能手动处理
    if is_mx_untradable(code):
        return 0, ST_MANUAL, "妙想模拟仓无法交易该标的，需手动卖出"

    avail = int(position.get("avail_count", 0) or 0)
    if avail <= 0:
        # 部分接口不返回 avail_count，退化为总持仓
        avail = int(position.get("count", 0) or 0)

    # 按可用量收敛并向下取整到 100 整数倍（妙想对非整手直接拒单）
    shares = min(sig.suggest_shares, avail)
    shares = (shares // 100) * 100
    if shares < 100:
        return 0, ST_INSUFFICIENT, f"可卖不足一手（可用{avail}股，建议{sig.suggest_shares}股）"

    if dry_run:
        return shares, ST_DRY_RUN, "试运行，未下单"

    resp = client.trade("sell", code, shares, use_market_price=True)
    if resp and resp.get("code") in ("0", "200"):
        return shares, ST_SUCCESS, ""
    message = (resp or {}).get("message", "未知错误")
    return shares, ST_FAILED, message


def run_sell(analyzer, client: MXMoniClient, dry_run: bool = False
             ) -> Tuple[List[SellExecution], str, bool]:
    """检测持仓卖出信号并执行。

    Returns:
        (执行结果列表, 市场状态, 是否硬拦截)
    """
    gate_inputs = fetch_gate_inputs(analyzer.fetcher)
    _, _, market_summary, regime, hard_intercept = check_market_gate(gate_inputs)
    logger.info(market_summary)
    if hard_intercept:
        logger.warning("硬拦截触发！当日全部持仓按清仓处理")

    positions = filter_stock_positions(client.get_positions())
    if not positions:
        logger.info("妙想模拟仓当前无股票持仓")
        return [], regime, hard_intercept

    entry_map = get_last_buy_dates_safe(client)
    sector_pct_map = fetch_sector_pct_map()
    if not sector_pct_map:
        logger.warning("板块行情不可用，板块类卖出规则（板块走弱/主线退潮）跳过")

    results: List[SellExecution] = []
    for p in positions:
        code = canonical_stock_code(p.get("code", ""))
        name = p.get("name", "") or code

        sector = analyzer._fetch_stock_sector(code)
        sector_pct = match_sector_pct(sector, sector_pct_map)
        sector_skipped = sector == UNKNOWN_SECTOR or sector_pct is None
        if sector_skipped:
            logger.warning(
                f"⚠️ {name}({code}): 板块信息不可用（板块={sector}，"
                f"板块行情={'有' if sector_pct is not None else '无'}）→ 板块类卖出规则已跳过"
            )

        row = HoldingRow(
            position=p, sector=sector,
            sector_pct=sector_pct, sector_skipped=sector_skipped,
        )
        res = SellExecution(row=row)

        try:
            df = analyzer.fetch_stock_data(code)
            if df is not None and len(df) >= 10:
                df = df.sort_values('date').reset_index(drop=True)
                df = analyzer.trend_analyzer._calculate_mas(df)
                sig = detect_sell_signals(
                    code, name, df, p, regime, hard_intercept,
                    sector=sector, sector_pct=sector_pct,
                    entry_date=entry_map.get(code, ""),
                )
            else:
                sig = None
                res.data_ok = False

            # 硬拦截兜底：行情缺失也要清仓，否则"全部清仓"被单只取数失败架空
            if hard_intercept and sig is None:
                sig = _force_clear_signal(
                    p, sector, sector_pct, "市场门控硬拦截，全部清仓（行情缺失，按持仓全量清仓）"
                )
                if sig is None:
                    # 判不了 ≠ 可继续持有，归入「行情缺失未判」而不是「继续持有」
                    res.data_ok = False
                    res.message = "硬拦截但现价/股数不可用，无法自动清仓"
                    logger.warning(f"    {name}({code}) {res.message}")

            row.signal = sig
            if sig is not None:
                shares, status, message = execute_sell(sig, p, client, dry_run=dry_run)
                res.shares, res.status, res.message = shares, status, message
                label = ACTION_LABEL.get(sig.action, sig.action)
                tail = f" | {message}" if message else ""
                logger.info(
                    f"{label} {name}({code}) {shares}股 → {STATUS_LABEL.get(status, status)}"
                    f" | {'；'.join(sig.reasons)}{tail}"
                )
            elif res.data_ok:
                logger.info(f"    {name}({code}) 无卖出信号，继续持有")
            else:
                logger.warning(f"    {name}({code}) 行情获取失败，无法检测卖出信号")

        except Exception as e:
            # 单只失败不阻断其余持仓
            res.data_ok = False
            res.status = ST_FAILED
            res.message = str(e)
            logger.warning(f"处理 {name}({code}) 失败: {e}")

        results.append(res)

    return results, regime, hard_intercept


def render_report(results: List[SellExecution], regime: str,
                  hard_intercept: bool, dry_run: bool) -> str:
    """渲染成交报告（Markdown）。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        f"# 趋势策略 — 尾盘卖出执行报告",
        "",
        f"**执行时间**：{now}" + ("（试运行，未下单）" if dry_run else ""),
        f"**市场状态**：{regime}" + ("　🔴 **硬拦截：全部清仓**" if hard_intercept else ""),
        f"**持仓检查**：{len(results)} 只股票持仓",
        "",
    ]

    # 单次遍历分类结果
    categorized = {
        'succeeded': [], 'failed': [], 'manual': [], 'insufficient': [],
        'holds': [], 'undetermined': []
    }
    for r in results:
        if r.signal is None:
            if r.data_ok:
                categorized['holds'].append(r)
            else:
                categorized['undetermined'].append(r)
        else:
            if r.status in (ST_SUCCESS, ST_DRY_RUN):
                categorized['succeeded'].append(r)
            elif r.status == ST_FAILED:
                categorized['failed'].append(r)
            elif r.status == ST_MANUAL:
                categorized['manual'].append(r)
            elif r.status == ST_INSUFFICIENT:
                categorized['insufficient'].append(r)

    triggered = [r for r in results if r.signal is not None]
    succeeded = categorized['succeeded']
    failed = categorized['failed']
    manual = categorized['manual']
    insufficient = categorized['insufficient']
    holds = categorized['holds']
    undetermined = categorized['undetermined']

    lines.extend([
        "## 执行概览",
        "",
        f"- 触发卖出信号：**{len(triggered)}** 只（已委托 {len(succeeded)}，失败 {len(failed)}）",
        f"- 需手动处理：{len(manual)} 只　可用不足一手：{len(insufficient)} 只",
        f"- 继续持有：{len(holds)} 只　行情缺失未判：{len(undetermined)} 只",
        "",
    ])

    if triggered:
        to_show = [r for r in triggered if r.status]
        lines.extend([
            "## 卖出执行明细",
            "",
            "| 股票 | 动作 | 现价 | 成本 | 盈亏 | 委托股数 | 结果 | 触发规则 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for r in to_show:
            sig = r.signal
            lines.append(
                f"| {r.name}({r.code}) | {ACTION_LABEL.get(sig.action, sig.action)} "
                f"| {_f(sig.current_price)} | {_f(sig.cost_price)} | {_f_pct(sig.profit_pct)} "
                f"| {r.shares}股 | {STATUS_LABEL.get(r.status, r.status)}"
                f"{f'（{r.message}）' if r.message else ''} "
                f"| {'；'.join(sig.reasons)} |"
            )
        lines.append("")

    # 板块不可用必须显式标注：否则「无卖出信号」会被误读成"板块没走弱"
    skipped = [r for r in results if r.row.sector_skipped]
    if skipped:
        shown = "、".join(f"{r.name}({r.code})" for r in skipped[:10])
        if len(skipped) > 10:
            shown += " 等"
        lines.extend([
            "> ⚠️ 以下标的板块信息不可用，板块类卖出规则（板块走弱 / 主线退潮）已跳过：",
            f"> {shown}。",
            "> 这些标的的「无卖出信号」不代表板块没走弱。",
            "",
        ])

    if holds:
        lines.extend([
            "## 继续持有（无卖出信号）",
            "",
            "| 股票 | 现价 | 成本 | 盈亏 | 板块 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for r in holds:
            if r.row.sector == UNKNOWN_SECTOR:
                sector_label = f"未知板块{'（规则已跳过）' if r.row.sector_skipped else ''}"
            else:
                sector_label = r.row.sector
                if r.row.sector_pct is not None:
                    sector_label += f"（{_f_pct(r.row.sector_pct)}）"
            lines.append(
                f"| {r.name}({r.code}) | {_f(r.row.position.get('current_price', 0))} "
                f"| {_f(r.row.position.get('cost_price', 0))} "
                f"| {_f_pct(position_profit_pct(r.row.position))} "
                f"| {sector_label} |"
            )
        lines.append("")

    if undetermined:
        lines.extend([
            "## 行情缺失，未判定",
            "",
        ])
        for r in undetermined:
            reason = r.message or "行情获取失败"
            lines.append(f"- {r.name}({r.code})：{reason}")
        lines.append("")

    return "\n".join(lines)


def _save_report(report: str) -> str:
    """保存报告到 reports/ 并返回路径。"""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"trend_sell_{datetime.now().strftime('%Y%m%d')}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存: {path}")
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='趋势策略 — 尾盘卖出执行（检测 + 自动下模拟仓单 + 出报告）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-notify', action='store_true', help='不发送推送通知')
    parser.add_argument('--dry-run', action='store_true', help='只检测不下单（调试用）')
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    setup_logging(log_prefix="trend_sell", debug=args.debug)

    logger.info("=" * 60)
    logger.info("趋势策略 — 尾盘卖出执行启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    if not os.getenv("MX_APIKEY"):
        logger.error("未配置 MX_APIKEY，无法读取持仓与下单")
        return 1

    try:
        # 复用盘后分析器的取数能力（日线 + 板块）
        from trend_analysis import SimpleTechnicalAnalyzer

        analyzer = SimpleTechnicalAnalyzer()
        client = MXMoniClient()

        results, regime, hard_intercept = run_sell(analyzer, client, dry_run=args.dry_run)
        if not results:
            logger.info("无股票持仓，无需出报告")
            return 0

        report = render_report(results, regime, hard_intercept, args.dry_run)
        _save_report(report)

        if not args.no_notify:
            notifier = NotificationService()
            if notifier.is_available():
                if notifier.send(report):
                    logger.info("通知发送成功")
                else:
                    logger.warning("通知发送失败")
            else:
                logger.warning("通知服务未配置")

        logger.info("运行完成")
        return 0

    except Exception as e:
        logger.exception(f"运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
