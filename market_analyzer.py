# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd

from config import get_config
from search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    north_flow: float = 0.0             # 北向资金净流入（亿元）
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情（A股/美股）
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    
    {{ Eddie Peng: Modify - 支持A股和美股双市场分析，根据.env配置自动判断。20260113 }}
    """
    
    # A股主要指数代码
    CN_MAIN_INDICES = {
        '000001': '上证指数',
        '399001': '深证成指',
        '399006': '创业板指',
        '000688': '科创50',
        '000016': '上证50',
        '000300': '沪深300',
    }
    
    # 美股主要指数代码（使用Yahoo Finance代码格式）
    US_MAIN_INDICES = {
        '^GSPC': '标普500',
        '^DJI': '道琼斯',
        '^IXIC': '纳斯达克',
        '^RUT': '罗素2000',
    }
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        初始化大盘分析器
        
        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        
        # {{ Eddie Peng: Add - 根据配置判断要分析的市场。20260113 }}
        self.analyze_cn = bool(self.config.cn_stock_list)  # 是否分析A股
        self.analyze_us = bool(self.config.us_stock_list)  # 是否分析美股
        
        if self.analyze_cn and self.analyze_us:
            logger.info("[大盘] 配置双市场分析模式: A股 + 美股")
        elif self.analyze_cn:
            logger.info("[大盘] 配置单市场分析模式: 仅A股")
        elif self.analyze_us:
            logger.info("[大盘] 配置单市场分析模式: 仅美股")
        else:
            logger.warning("[大盘] 未配置任何股票列表，大盘复盘可能无法执行")
        
    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据（根据配置获取A股/美股/双市场）
        
        {{ Eddie Peng: Modify - 支持根据配置获取不同市场数据。20260113 }}
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情（根据配置决定获取哪个市场）
        all_indices = []
        if self.analyze_cn:
            cn_indices = self._get_cn_main_indices()
            all_indices.extend(cn_indices)
        if self.analyze_us:
            us_indices = self._get_us_main_indices()
            all_indices.extend(us_indices)
        overview.indices = all_indices
        
        # 2. 获取A股涨跌统计（仅在配置了A股时）
        if self.analyze_cn:
            self._get_market_statistics(overview)
            # 3. 获取A股板块涨跌榜
            self._get_sector_rankings(overview)
            # 4. 获取北向资金（可选）
            self._get_north_flow(overview)
        
        return overview
    
    def _get_cn_main_indices(self) -> List[MarketIndex]:
        """获取A股主要指数实时行情"""
        indices = []
        
        try:
            logger.info("[大盘-A股] 获取主要指数实时行情...")
            
            # 使用 akshare 获取指数行情
            df = ak.stock_zh_index_spot_em()
            
            if df is not None and not df.empty:
                for code, name in self.CN_MAIN_INDICES.items():
                    # 查找对应指数
                    row = df[df['代码'] == code]
                    if row.empty:
                        # 尝试带前缀查找
                        row = df[df['代码'].str.contains(code)]
                    
                    if not row.empty:
                        row = row.iloc[0]
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=float(row.get('最新价', 0) or 0),
                            change=float(row.get('涨跌额', 0) or 0),
                            change_pct=float(row.get('涨跌幅', 0) or 0),
                            open=float(row.get('今开', 0) or 0),
                            high=float(row.get('最高', 0) or 0),
                            low=float(row.get('最低', 0) or 0),
                            prev_close=float(row.get('昨收', 0) or 0),
                            volume=float(row.get('成交量', 0) or 0),
                            amount=float(row.get('成交额', 0) or 0),
                        )
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        indices.append(index)
                        
                logger.info(f"[大盘-A股] 获取到 {len(indices)} 个指数行情")
                
        except Exception as e:
            logger.error(f"[大盘-A股] 获取指数行情失败: {e}")
        
        return indices
    
    def _get_us_main_indices(self) -> List[MarketIndex]:
        """
        获取美股主要指数实时行情
        
        {{ Eddie Peng: Modify - 使用 yfinance 主数据源 + akshare 备选，防止限流。20260113 }}
        """
        indices = []
        
        # 方案1: 尝试使用 yfinance（主数据源）
        try:
            logger.info("[大盘-美股] 尝试使用 yfinance 获取指数行情...")
            
            import yfinance as yf
            import time
            
            # 使用 yfinance 获取美股指数行情
            for idx, (code, name) in enumerate(self.US_MAIN_INDICES.items()):
                try:
                    logger.debug(f"[大盘-美股] 获取 {name} ({code})...")
                    
                    # 创建 Ticker 对象
                    ticker = yf.Ticker(code)
                    
                    # 获取最近2天的历史数据（包含今天和昨天）
                    hist = ticker.history(period='2d')
                    
                    if hist is None or hist.empty:
                        logger.warning(f"[大盘-美股] {name} ({code}) 无数据")
                        continue
                    
                    # 获取最新一天的数据
                    latest = hist.iloc[-1]
                    
                    # 获取昨天的收盘价（用于计算涨跌）
                    prev_close = hist.iloc[-2]['Close'] if len(hist) >= 2 else latest['Close']
                    
                    current = float(latest['Close'])
                    open_price = float(latest['Open'])
                    high = float(latest['High'])
                    low = float(latest['Low'])
                    volume = float(latest['Volume'])
                    
                    # 计算涨跌
                    change = current - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    index = MarketIndex(
                        code=code,
                        name=name,
                        current=current,
                        change=change,
                        change_pct=change_pct,
                        open=open_price,
                        high=high,
                        low=low,
                        prev_close=prev_close,
                        volume=volume,
                        amount=0,  # yfinance 不提供成交额
                    )
                    
                    # 计算振幅
                    if index.prev_close > 0:
                        index.amplitude = (index.high - index.low) / index.prev_close * 100
                    
                    indices.append(index)
                    logger.debug(f"[大盘-美股] {name}: {current:.2f} ({change_pct:+.2f}%)")
                    
                    # 添加延迟，避免触发限流（除了最后一个）
                    if idx < len(self.US_MAIN_INDICES) - 1:
                        time.sleep(1.5)
                    
                except Exception as e:
                    # 检查是否是限流错误
                    error_msg = str(e)
                    if 'Rate limited' in error_msg or 'Too Many Requests' in error_msg:
                        logger.warning(f"[大盘-美股] yfinance 触发限流，切换到备选方案")
                        break  # 跳出循环，使用备选方案
                    logger.warning(f"[大盘-美股] 获取 {name} ({code}) 失败: {e}")
                    continue
            
            # 如果成功获取到数据，直接返回
            if indices:
                logger.info(f"[大盘-美股] yfinance 获取到 {len(indices)} 个指数行情")
                return indices
                
        except ImportError:
            logger.warning("[大盘-美股] yfinance 未安装，将使用备选方案")
        except Exception as e:
            logger.warning(f"[大盘-美股] yfinance 获取失败: {e}，将使用备选方案")
        
        # 方案2: 使用 akshare 作为备选方案
        logger.info("[大盘-美股] 使用 akshare 备选方案获取指数行情...")
        try:
            df = ak.index_us_stock_sina()
            
            if df is not None and not df.empty:
                # 代码映射（akshare 使用不同的代码格式）
                code_mapping = {
                    '^GSPC': ['GSPC', 'SPX', '.INX', '标普500'],
                    '^DJI': ['DJI', 'DJIA', '道琼斯'],
                    '^IXIC': ['IXIC', 'COMP', '纳斯达克'],
                    '^RUT': ['RUT', 'RUI', '罗素'],
                }
                
                for code, name in self.US_MAIN_INDICES.items():
                    search_terms = code_mapping.get(code, [code.replace('^', '')])
                    row = None
                    
                    # 尝试多种匹配方式
                    for term in search_terms:
                        # 按 symbol 匹配
                        if 'symbol' in df.columns:
                            row = df[df['symbol'].str.contains(term, case=False, na=False)]
                        if row is not None and not row.empty:
                            break
                        
                        # 按 cname 匹配
                        if 'cname' in df.columns:
                            row = df[df['cname'].str.contains(term, na=False)]
                        if row is not None and not row.empty:
                            break
                    
                    if row is not None and not row.empty:
                        row = row.iloc[0]
                        
                        # 提取数据（兼容不同的列名）
                        current = float(row.get('trade', row.get('now', row.get('price', 0))) or 0)
                        prev_close = float(row.get('settlement', row.get('preclose', 0)) or 0)
                        open_price = float(row.get('open', 0) or 0)
                        high = float(row.get('high', 0) or 0)
                        low = float(row.get('low', 0) or 0)
                        volume = float(row.get('volume', 0) or 0)
                        
                        change = current - prev_close if prev_close > 0 else 0
                        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=current,
                            change=change,
                            change_pct=change_pct,
                            open=open_price,
                            high=high,
                            low=low,
                            prev_close=prev_close,
                            volume=volume,
                            amount=0,
                        )
                        
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        
                        indices.append(index)
                        logger.debug(f"[大盘-美股] {name}: {current:.2f} ({change_pct:+.2f}%)")
                
                logger.info(f"[大盘-美股] akshare 获取到 {len(indices)} 个指数行情")
        except Exception as e:
            logger.error(f"[大盘-美股] akshare 备选方案也失败: {e}")
        
        return indices
    
    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")
            
            # 获取全部A股实时行情
            df = ak.stock_zh_a_spot_em()
            
            if df is not None and not df.empty:
                # 涨跌统计
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    overview.up_count = len(df[df[change_col] > 0])
                    overview.down_count = len(df[df[change_col] < 0])
                    overview.flat_count = len(df[df[change_col] == 0])
                    
                    # 涨停跌停统计（涨跌幅 >= 9.9% 或 <= -9.9%）
                    overview.limit_up_count = len(df[df[change_col] >= 9.9])
                    overview.limit_down_count = len(df[df[change_col] <= -9.9])
                
                # 两市成交额
                amount_col = '成交额'
                if amount_col in df.columns:
                    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                    overview.total_amount = df[amount_col].sum() / 1e8  # 转为亿元
                
                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")
                
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")
    
    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")
            
            # 获取行业板块行情
            df = ak.stock_board_industry_name_em()
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5
                    top = df.nlargest(5, change_col)
                    overview.top_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    def _get_north_flow(self, overview: MarketOverview):
        """获取北向资金流入"""
        try:
            logger.info("[大盘] 获取北向资金...")
            
            # 获取北向资金数据
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            
            if df is not None and not df.empty:
                # 取最新一条数据
                latest = df.iloc[-1]
                if '当日净流入' in df.columns:
                    overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
                elif '净流入' in df.columns:
                    overview.north_flow = float(latest['净流入']) / 1e8
                    
                logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻（根据配置搜索A股/美股/双市场）
        
        {{ Eddie Peng: Modify - 支持根据配置搜索不同市场的新闻。20260113 }}
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []
        today = datetime.now()
        month_str = f"{today.year}年{today.month}月"
        
        # {{ Eddie Peng: Modify - 根据配置构建不同的搜索查询。20260113 }}
        search_queries = []
        
        # A股搜索查询
        if self.analyze_cn:
            search_queries.extend([
                f"A股 大盘 复盘 {month_str}",
                f"A股 市场 热点 板块 {month_str}",
            ])
        
        # 美股搜索查询
        if self.analyze_us:
            search_queries.extend([
                f"US stock market review {today.strftime('%B %Y')}",
                f"S&P 500 Nasdaq market analysis {today.strftime('%B %Y')}",
            ])
        
        try:
            logger.info(f"[大盘] 开始搜索市场新闻 (A股:{self.analyze_cn}, 美股:{self.analyze_us})...")
            
            for query in search_queries:
                # 使用 search_stock_news 方法，传入"大盘"作为股票名
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name="大盘" if "A股" in query else "US Market",
                    max_results=2,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        try:
            logger.info("[大盘] 调用大模型生成复盘报告...")
            
            generation_config = {
                'temperature': 0.7,
                'max_output_tokens': 2048,
            }
            
            # 根据 analyzer 使用的 API 类型调用
            if self.analyzer._use_openai:
                # 使用 OpenAI 兼容 API
                review = self.analyzer._call_openai_api(prompt, generation_config)
            else:
                # 使用 Gemini API
                response = self.analyzer._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                review = response.text.strip() if response and response.text else None
            
            if review:
                logger.info(f"[大盘] 复盘报告生成成功，长度: {len(review)} 字符")
                return review
            else:
                logger.warning("[大盘] 大模型返回为空")
                return self._generate_template_review(overview, news)
                
        except Exception as e:
            logger.error(f"[大盘] 大模型生成复盘报告失败: {e}")
            return self._generate_template_review(overview, news)
    
    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """
        构建复盘报告 Prompt
        
        {{ Eddie Peng: Modify - 根据配置生成不同市场的 Prompt。20260113 }}
        """
        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息（仅在有A股数据时显示）
        sectors_section = ""
        if self.analyze_cn and (overview.top_sectors or overview.bottom_sectors):
            top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
            bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])
            sectors_section = f"""
## 板块表现
领涨: {top_sectors_text}
领跌: {bottom_sectors_text}
"""
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        # 根据配置决定分析师角色和市场描述
        if self.analyze_cn and self.analyze_us:
            analyst_role = "全球股市市场分析师"
            market_desc = "A股和美股市场"
            title = "全球市场复盘"
        elif self.analyze_cn:
            analyst_role = "A股市场分析师"
            market_desc = "A股市场"
            title = "A股大盘复盘"
        else:
            analyst_role = "美股市场分析师"
            market_desc = "美股市场"
            title = "美股市场复盘"
        
        # 构建市场概况部分（根据是否有A股数据）
        market_stats_section = ""
        if self.analyze_cn and overview.up_count > 0:
            market_stats_section = f"""
## 市场概况（A股）
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元
- 北向资金: {overview.north_flow:+.2f} 亿元
"""
        
        prompt = f"""你是一位专业的{analyst_role}，请根据以下数据生成一份简洁的{market_desc}复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_text}
{market_stats_section}
{sectors_section}

## 市场新闻
{news_text if news_text else "暂无相关新闻"}

---

# 输出格式模板（请严格按此格式输出）

## 📊 {overview.date} {title}

### 一、市场总结
（2-3句话概括今日市场整体表现，包括主要指数涨跌、成交量变化）

### 二、指数点评
（分析各主要指数走势特点和市场结构）

### 三、资金动向
（解读成交额和资金流向的含义，如有北向资金则分析）

### 四、热点解读
（分析市场热点、板块轮动背后的逻辑和驱动因素）

### 五、后市展望
（结合当前走势和新闻，给出市场预判）

### 六、风险提示
（需要关注的风险点）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
        return prompt
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用模板生成复盘报告（无大模型时的备选方案）
        
        {{ Eddie Peng: Modify - 支持根据配置生成不同市场的模板。20260113 }}
        """
        
        # 根据配置决定市场描述
        if self.analyze_cn and self.analyze_us:
            market_name = "全球市场"
            title = "全球市场复盘"
        elif self.analyze_cn:
            market_name = "A股市场"
            title = "A股大盘复盘"
        else:
            market_name = "美股市场"
            title = "美股市场复盘"
        
        # 判断市场走势（优先使用上证指数，其次标普500）
        main_index = next((idx for idx in overview.indices if idx.code == '000001'), None)
        if not main_index:
            main_index = next((idx for idx in overview.indices if idx.code == '^GSPC'), None)
        
        if main_index:
            if main_index.change_pct > 1:
                market_mood = "强势上涨"
            elif main_index.change_pct > 0:
                market_mood = "小幅上涨"
            elif main_index.change_pct > -1:
                market_mood = "小幅下跌"
            else:
                market_mood = "明显下跌"
        else:
            market_mood = "震荡整理"
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:6]:  # 显示前6个指数
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息（仅在有A股数据时显示）
        sector_section = ""
        if self.analyze_cn and (overview.top_sectors or overview.bottom_sectors):
            top_text = "、".join([s['name'] for s in overview.top_sectors[:3]])
            bottom_text = "、".join([s['name'] for s in overview.bottom_sectors[:3]])
            sector_section = f"""
### 四、板块表现
- **领涨**: {top_text}
- **领跌**: {bottom_text}
"""
        
        # 涨跌统计（仅在有A股数据时显示）
        stats_section = ""
        if self.analyze_cn and overview.up_count > 0:
            stats_section = f"""
### 三、涨跌统计（A股）
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |
| 北向资金 | {overview.north_flow:+.2f}亿 |
"""
        
        report = f"""## 📊 {overview.date} {title}

### 一、市场总结
今日{market_name}整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text}
{stats_section}
{sector_section}

### 五、风险提示
市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程
        
        Returns:
            复盘报告文本
        """
        logger.info("========== 开始大盘复盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盘复盘分析完成 ==========")
        
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
