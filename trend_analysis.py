# -*- coding: utf-8 -*-
"""
===================================
趋势交易策略 — 日度分析与信号检测
===================================

定位：趋势波段系统。只做主线中的强趋势股，只在缩量回踩时介入。

职责：
1. 读取妙想自选股或指定股票列表，直接进行技术分析
2. 市场环境过滤（调用 market_gate 模块）
3. 纯技术分析（缩量回踩MA5等规则）
4. 观察池维护（趋势破坏自动剔除）

拦截层（自下而上，越靠下越"硬"）：
- market_gate：市场环境层，硬拦截时不开仓
- veto_rules：负面清单（9 条硬否决），任一触发即不进信号池、不看评分
- removal_rules：趋势破坏剔除（连续2天跌破10日线等）
- signal_detector：买点形态 + 信号质量门（情绪过热等）

核心策略：
- 买点：主升中的缩量回踩MA5（不破5日线 + 换手率>5%）
- 不做：加速追高、情绪高潮接力、连续大阳后追涨
- 环境过滤：见 strategy/market.md
- 趋势不走坏即保留，连续2天跌破10日线才剔除

使用方式：
    python trend_analysis.py                    # 正常运行
    python trend_analysis.py --debug            # 调试模式
    python trend_analysis.py --no-notify        # 不发送通知
    python trend_analysis.py --stocks CODE1,CODE2  # 指定股票分析
    python trend_analysis.py --max-stocks N     # 最多分析N只股票（按跌幅排序）
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_provider import DataFetcherManager, canonical_stock_code
from src.indicators import add_standard_indicators
from src.config import setup_env
from src.notify.service import NotificationService
from src.mx.service import MXService
from src.mx.position_utils import filter_stock_positions, get_last_buy_dates_safe
from src.mx.client import MXMoniClient
from src.market_state.market_gate import (
    check_market_gate, diagnose_regime, fetch_gate_inputs,
)
from src.trend.analyzer import StockTrendAnalyzer
from src.trend.entry_tier import (
    TIER_TIGHT, resolve_tier, screen_by_tier, tier_rule,
)
from src.trend.removal_rules import (
    RemovalStats, check_removal_rules, check_removal_rules_detail,
)
from src.trend.veto_rules import (
    ACTION_REMOVE, check_external_veto, check_market_veto, fetch_main_net_inflow,
)
from src.trend.signal_detector import (
    UNKNOWN_SECTOR, TechnicalSignal, detect_pullback_signals,
)
from src.trend.report import generate_technical_report
setup_env()

logger = logging.getLogger(__name__)

# 松筛默认关键词：宽松条件，保证池子不会饿死。精筛由 signal_detector 负责。
DEFAULT_SCREEN_KEYWORD = (
    "市值大于30亿小于500亿；均线多头排列；换手率大于3%；"
    "不要科创板不要创业板不要北交所不要ST"
)


class SimpleTechnicalAnalyzer:
    """
    简化版技术分析器
    
    不依赖 LLM，纯技术指标计算
    """
    
    def __init__(self):
        self.fetcher = DataFetcherManager()
        self.mx_service = MXService()
        self.trend_analyzer = StockTrendAnalyzer()  # 复用 main.py 的技术指标计算
    
    def get_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        from src.trading_calendar import get_trading_dates as _get_trading_dates
        return _get_trading_dates(start_date, end_date)

    def get_stocks_pct_change(self, stock_list: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        获取股票列表的涨跌幅

        Args:
            stock_list: [(code, name), ...]

        Returns:
            {code: pct_change} 涨跌幅百分比
        """
        pct_changes = {}
        for code, name in stock_list:
            try:
                quote = self.fetcher.get_realtime_quote(code)
                if quote and hasattr(quote, 'pct_chg'):
                    pct_changes[code] = float(quote.pct_chg) if quote.pct_chg is not None else 0.0
                else:
                    pct_changes[code] = 0.0
            except Exception as e:
                logger.debug(f"获取 {code} 涨跌幅失败: {e}")
                pct_changes[code] = 0.0
        return pct_changes

    def fetch_stock_data(self, code: str, days: int = 95) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据（直接从网络获取）

        天数口径为自然日：95 天约 65 个交易日，满足负面清单 60 日类规则
        （V2/V7/V8 需 61 根 K 线）与 MA60 的计算需求。

        Args:
            code: 股票代码
            days: 获取天数（自然日）

        Returns:
            DataFrame 或 None
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            result = self.fetcher.get_daily_data(code, start_str, end_str)
            if isinstance(result, tuple) and len(result) >= 1:
                df = result[0]
            else:
                df = result

            if df is not None and hasattr(df, 'empty') and not df.empty:
                df_latest_date = pd.to_datetime(df['date'].max()).date()
                trading_dates = self.get_trading_dates(end_date - timedelta(days=30), end_date)
                if trading_dates:
                    last_trading_day = trading_dates[-1]
                    if df_latest_date < last_trading_day:
                        logger.error(f"❌ {code} 网络获取的数据仍过期(最新:{df_latest_date}, 需要:{last_trading_day})")
                        return None
                # 数据层不再算指标，入口取数后统一追加（ma5/ma10/ma20/volume_ratio）
                return add_standard_indicators(df)
            else:
                logger.error(f"❌ {code} 从网络获取数据失败")
                return None

        except Exception as e:
            logger.warning(f"获取 {code} 数据失败: {e}")
            return None

    def _fetch_stock_sector(self, code: str) -> str:
        """获取股票所属板块名称。

        取不到时降级为「未知板块」并记 warning —— 不返回空串：
        空串在 Markdown 表格里是空单元格，看起来像列错位，且无法与"确实没板块"区分。
        """
        try:
            from data_provider.fetchers.efinance_fetcher import EfinanceFetcher
            ef = EfinanceFetcher()
            df = ef.get_belong_board(code)
            if df is not None and not df.empty and "板块名称" in df.columns:
                sector = str(df["板块名称"].iloc[0]).strip()
                if sector:
                    return sector
            logger.warning(
                f"⚠️ {code}: 未取到所属板块 → 降级为「{UNKNOWN_SECTOR}」，板块类规则已跳过"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ {code}: 获取所属板块失败（{e}）→ 降级为「{UNKNOWN_SECTOR}」，板块类规则已跳过"
            )
        return UNKNOWN_SECTOR

    def screen_new_candidates(self, keyword: str, page_size: int = 30) -> List[Tuple[str, str]]:
        """MX 松筛发现新候选，返回 [(code, name), ...]

        仅用于发现观察池新名字，不做精筛（精筛交给 signal_detector）。
        解析 MX 返回的中文列名，字段名可能带日期后缀，做防御式匹配。
        """
        try:
            rows, total = self.mx_service.screen_stocks(keyword, page_no=1, page_size=page_size)
            if not rows:
                logger.warning(f"松筛无结果（关键词: {keyword}）")
                return []
            candidates = []
            for row in rows:
                code = None
                name = ""
                for k, v in row.items():
                    ks = str(k)
                    # 代码列：列名可能是"代码"或"股票代码"，排除"市场代码简称"
                    if "市场" in ks:
                        continue
                    if "代码" in ks and code is None:
                        code = str(v or "").strip()
                    elif "简称" in ks or "名称" in ks:
                        name = str(v or "").strip()
                if code:
                    code = code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
                    candidates.append((code, name or code))
            logger.info(f"松筛返回 {total} 条，解析到 {len(candidates)} 个候选")
            return candidates
        except Exception as e:
            logger.warning(f"松筛失败: {e}")
            return []

    def analyze_all_stocks(self, stock_list: List[Tuple[str, str]], 
                          max_stocks: Optional[int] = None,
                          sort_by_pct: bool = True
                          ) -> Tuple[List[TechnicalSignal],
                                     List[Tuple[str, str, str]],
                                     List[Tuple[str, str, str]],
                                     List[Tuple[str, str, str, str]],
                                     RemovalStats]:
        """
        分析所有关注股票，返回技术信号列表、剔除列表、失败列表、否决列表与剔除规则统计

        准入顺序：剔除规则（趋势破坏） → 负面清单行情类否决 → 买点信号检测
                  → 负面清单外部数据类否决（仅信号候选）

        Args:
            stock_list: [(code, name), ...]
            max_stocks: 最大分析数量，超过则按跌幅排序取前N只
            sort_by_pct: 是否按跌幅排序（优先分析跌幅大的股票）

        Returns:
            (技术信号列表,
             [(code, name, 剔除原因), ...],
             [(code, name, 失败原因), ...],
             [(code, name, 否决动作, 否决原因), ...],
             剔除规则逐条统计 RemovalStats)
             否决动作为 ACTION_SKIP（仅跳过当日信号）或 ACTION_REMOVE（剔除自选池）
        """
        all_signals = []
        removed_stocks = []
        failed_stocks = []
        vetoed_stocks = []
        stats = RemovalStats()

        if max_stocks and len(stock_list) > max_stocks:
            if sort_by_pct:
                logger.info(f"获取涨跌幅数据，股票数量 {len(stock_list)} 超过限制 {max_stocks}，按跌幅排序...")
                pct_changes = self.get_stocks_pct_change(stock_list)
                sorted_stocks = sorted(stock_list, key=lambda x: pct_changes.get(x[0], 0))
                stock_list = sorted_stocks[:max_stocks]
                logger.info(f"已选取跌幅最大的 {max_stocks} 只股票进行分析")
            else:
                stock_list = stock_list[:max_stocks]

        logger.info(f"开始处理 {len(stock_list)} 只股票...")

        for i, (code, name) in enumerate(stock_list):
            try:
                df = self.fetch_stock_data(code)

                # 统一计算 MA，避免在剔除检查和信号检测中重复计算
                if df is not None and len(df) >= 10:
                    df = df.sort_values('date').reset_index(drop=True)
                    df = self.trend_analyzer._calculate_mas(df)

                # 逐条跑完 4 条规则：既给出剔除结论，也累积「检查 N 只 / 触发 N 只」统计
                checks = check_removal_rules_detail(code, df)
                stats.record(f"{name}({code})", checks)
                should_remove, remove_reason = check_removal_rules(code, df)
                if should_remove:
                    removed_stocks.append((code, name, remove_reason))
                    logger.info(f"❌ 剔除 {name}({code}): {remove_reason}")
                    continue

                # 负面清单（行情类）：任一规则触发即否决，不进信号池、不看评分
                market_veto = check_market_veto(code, name, df)
                if market_veto.vetoed:
                    vetoed_stocks.append(
                        (code, name, market_veto.action, '；'.join(market_veto.reasons))
                    )
                    continue

                signals = detect_pullback_signals(code, name, df)

                if signals:
                    # 负面清单（外部数据类）：只对已产出信号的候选惰性调用妙想 API
                    ext_veto = check_external_veto(code, name, self.mx_service, self.fetcher)
                    if ext_veto.vetoed:
                        vetoed_stocks.append(
                            (code, name, ext_veto.action, '；'.join(ext_veto.reasons))
                        )
                        continue

                    sector = self._fetch_stock_sector(code)
                    for s in signals:
                        s.sector = sector
                    logger.info(f"✅ {name}({code}) [{sector}]: 发现 {len(signals)} 个信号")
                else:
                    logger.info(f"    {name}({code}) 无信号")
                all_signals.extend(signals)

                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(stock_list)}")

            except Exception as e:
                logger.warning(f"分析 {name}({code}) 失败: {e}")
                failed_stocks.append((code, name, str(e)))
                continue

        all_signals.sort(key=lambda x: x.score, reverse=True)

        veto_removed = sum(1 for v in vetoed_stocks if v[2] == ACTION_REMOVE)
        kept_count = len(stock_list) - len(removed_stocks) - len(vetoed_stocks) - len(failed_stocks)
        logger.info(
            f"处理完成 | 保留:{kept_count} 剔除:{len(removed_stocks)} "
            f"负面清单否决:{len(vetoed_stocks)}(其中剔除自选池{veto_removed}) "
            f"失败:{len(failed_stocks)} 信号:{len(all_signals)}"
        )
        # 逐条输出剔除规则的检查/触发统计（让"剔除 N 只"可解释、未实现项可见）
        stats.log_summary()
        return all_signals, removed_stocks, failed_stocks, vetoed_stocks, stats


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='简化版趋势跟踪系统 - 无 LLM（集成模拟交易）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )

    parser.add_argument(
        '--no-notify',
        action='store_true',
        help='不发送推送通知'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        help='指定要分析的股票代码，逗号分隔（覆盖妙想自选股）'
    )

    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='每天最多分析多少只股票（按跌幅排序优先分析跌幅大的）'
    )

    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='不执行松筛选股（默认在非 --stocks 模式下自动执行，往自选池补充新候选）'
    )

    parser.add_argument(
        '--screen-keyword',
        type=str,
        default=None,
        help='松筛选股关键词（默认使用内置宽松条件，可用 SMART_SCREEN_KEYWORD 覆盖）'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='仅列出当前自选池，不执行分析'
    )

    trade_group = parser.add_argument_group('交易模式（可选）')
    trade_group.add_argument(
        '--trade',
        action='store_true',
        help='盘后分析模式：分析技术信号并生成次日交易计划'
    )
    trade_group.add_argument(
        '--trade-execute',
        action='store_true',
        help='盘中执行模式：检查持仓止损止盈，执行买入'
    )
    trade_group.add_argument(
        '--trade-plan',
        action='store_true',
        help='查看当前交易计划'
    )

    return parser.parse_args()


def _list_self_selected(analyzer: 'SimpleTechnicalAnalyzer') -> int:
    """列出当前妙想自选池。"""
    try:
        stock_codes, name_mapping = analyzer.mx_service.fetch_self_selected()
        if not stock_codes:
            logger.info("当前自选池为空")
            return 0
        logger.info(f"当前自选池共 {len(stock_codes)} 只:")
        for code in stock_codes:
            logger.info(f"  {code} {name_mapping.get(code, '')}")
        return 0
    except Exception as e:
        logger.error(f"获取自选池失败: {e}")
        return 1


def _fetch_held_codes() -> set:
    """读取妙想模拟仓股票持仓代码集合，用于抑制「已持仓又提示买入」。

    只取代码，不做卖出检测（卖出由尾盘任务 trend_sell.py 负责）。

    Returns:
        股票持仓代码集合（取不到时返回空集合，买入信号照常输出）
    """
    if not os.getenv("MX_APIKEY"):
        logger.warning("未配置 MX_APIKEY，跳过持仓读取（不抑制已持仓买入信号）")
        return set()

    try:
        positions = filter_stock_positions(MXMoniClient().get_positions())
    except Exception as e:
        logger.warning(f"读取妙想持仓失败: {e}（不抑制已持仓买入信号）")
        return set()

    return {canonical_stock_code(p.get("code", "")) for p in positions}


def _save_report(report: str) -> str:
    """将报告保存到文件并返回路径。"""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    report_path = os.path.join(reports_dir, f"technical_simple_{today_str}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存: {report_path}")
    return report_path


def _send_notification(report: str) -> bool:
    """发送通知，如果已配置且可用。"""
    notifier = NotificationService()
    if not notifier.is_available():
        logger.warning("通知服务未配置")
        return False
    success = notifier.send(report)
    if success:
        logger.info("通知发送成功")
    else:
        logger.warning("通知发送失败")
    return success


def main():
    """主入口"""
    args = parse_arguments()
    
    # 配置日志
    from src.logging_config import setup_logging
    setup_logging(log_prefix="stock_analysis_simple", debug=args.debug)
    
    logger.info("=" * 60)
    logger.info("趋势交易策略 — 日度分析启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        analyzer = SimpleTechnicalAnalyzer()

        # 0. 列出自选池模式
        if args.list:
            logger.info("模式: 列出当前自选池")
            return _list_self_selected(analyzer)

        max_stocks = args.max_stocks
        if max_stocks is None:
            max_stocks = int(os.getenv('MAX_STOCKS_PER_DAY', '0')) or None
        
        # 1. 获取股票列表（从妙想或命令行）
        if args.stocks:
            # 使用命令行指定的股票
            stock_codes = [canonical_stock_code(c) for c in args.stocks.split(',') if c.strip()]
            name_mapping = {code: code for code in stock_codes}
            logger.info(f"使用指定股票列表: {stock_codes}")
        else:
            # 从妙想获取当前自选池
            stock_codes, name_mapping = analyzer.mx_service.fetch_self_selected()

            # 1.5 松筛：发现新候选并补充进自选池（观察池发现环节，不做精筛）
            if not args.no_screen:
                keyword = args.screen_keyword or os.getenv('SMART_SCREEN_KEYWORD') or DEFAULT_SCREEN_KEYWORD
                logger.info(f"执行松筛选股: {keyword}")
                candidates = analyzer.screen_new_candidates(keyword)
                if candidates:
                    existing = set(stock_codes)
                    new_codes = [c for c, _ in candidates if c not in existing]
                    if new_codes:
                        logger.info(f"发现 {len(new_codes)} 只新候选: {new_codes}")
                        # 写入妙想自选池，并合并进待分析列表
                        added = analyzer.mx_service.add_self_select(",".join(new_codes))
                        if added:
                            logger.info(f"已加入妙想自选池 {len(new_codes)} 只")
                            for c, n in candidates:
                                if c in new_codes:
                                    stock_codes.append(c)
                                    name_mapping[c] = n
                        else:
                            logger.warning("加入自选池失败，新候选本次不分析")
                    else:
                        logger.info("无新候选，池子已覆盖")
                else:
                    logger.info("松筛无结果，跳过")

        if not stock_codes:
            logger.error("没有获取到股票列表，退出")
            return 1
        
        # 2. 技术分析（包含剔除检查）
        stock_list = list(zip(stock_codes, [name_mapping.get(c, c) for c in stock_codes]))
        logger.info(f"当前关注列表: {len(stock_list)} 只股票")

        signals, removed_stocks, failed_stocks, vetoed_stocks, removal_stats = analyzer.analyze_all_stocks(
            stock_list, max_stocks=max_stocks, sort_by_pct=False
        )

        # 3. 从妙想删除剔除的股票（趋势破坏 + 负面清单极端过热类；命令行指定模式不删）
        codes_to_remove = [code for code, _, _ in removed_stocks]
        codes_to_remove += [c for c, _, action, _ in vetoed_stocks if action == ACTION_REMOVE]
        if codes_to_remove and not args.stocks:
            success = analyzer.mx_service.remove_stocks(codes_to_remove)
            if success:
                logger.info(f"已从妙想删除 {len(codes_to_remove)} 只自选股")
            else:
                logger.warning("从妙想删除失败")
        
        # 4. 市场环境检查 + 调节信号评分（入口取数 → 纯判定）
        gate_inputs = fetch_gate_inputs(analyzer.fetcher)
        can_trade, market_conditions, market_summary, market_regime, hard_intercept = check_market_gate(gate_inputs)
        logger.info(market_summary)
        # 单独取一次诊断明细（均线排列 / 偏离 MA20 / 命中路径）供报告展示。
        # 不改 check_market_gate 的返回值，避免影响 etf_observe.py 的调用。
        regime_diag = diagnose_regime(
            gate_inputs.get("index_df"), sum(1 for v in market_conditions.values() if v)
        )
        logger.info(f"市场状态判定明细 → {regime_diag.describe()}")
        if hard_intercept:
            logger.warning("硬拦截触发！当日应清仓所有持仓，不执行任何买入操作")

        # 开仓规则档位：环境调整落到具体规则（档位收紧 + 亏损限额），不再乘评分系数
        tier = resolve_tier(market_regime)
        rule = tier_rule(tier)
        tier_note = rule.label if rule else "不开仓（未启用档位）"
        logger.info(f"开仓规则档位：{tier_note}" + (f"（{rule.describe()}）" if rule else ""))
        # 经 apply_regime 登记档位：漏跑这一步会在渲染层抛异常，
        # 而不是让 effective_score 静默保持 0 把信号全判成"暂不关注"
        for s in signals:
            s.apply_regime(tier, tier_note)

        # 档位收紧过滤：位置（本地）+ 资金确认（收紧档，惰性外部取数）
        def _net_inflow(code: str, name: str):
            return fetch_main_net_inflow(code, name, analyzer.mx_service)

        signals, tier_blocked = screen_by_tier(
            signals, tier,
            net_inflow_fn=_net_inflow if tier == TIER_TIGHT else None,
        )

        # 4.5 已持仓股票抑制买入信号（避免"持有又提示买入"）
        held_codes = _fetch_held_codes()
        if held_codes:
            filtered = [s for s in signals if canonical_stock_code(s.code) not in held_codes]
            if len(filtered) != len(signals):
                logger.info(f"抑制 {len(signals) - len(filtered)} 只持仓股票的买入信号")
            signals = filtered

        report = generate_technical_report(signals, removed_stocks,
                                           market_env=(can_trade, market_conditions, market_summary, market_regime),
                                           failed_stocks=failed_stocks,
                                           vetoed_stocks=vetoed_stocks,
                                           removal_stats=removal_stats,
                                           regime_diag=regime_diag,
                                           tier_blocked=tier_blocked)

        # 5. 保存报告
        _save_report(report)

        # 6. 发送通知
        if not args.no_notify:
            _send_notification(report)
        
        logger.info("运行完成")
        return 0
        
    except Exception as e:
        logger.exception(f"运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
