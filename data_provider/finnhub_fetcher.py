# -*- coding: utf-8 -*-
"""
===================================
FinnhubFetcher - 美股数据源
===================================
{{ Eddie Peng: Add - 新增 Finnhub 数据源，专门用于获取美股数据。20260113 }}

数据来源：Finnhub API (https://finnhub.io/)
特点：支持美股实时和历史数据
适用：美股（US Market）

使用前需要：
1. 注册 Finnhub 账号获取免费 API Key
2. 在 .env 中配置 FINNHUB_API_KEY

免费额度：
- 60 次/分钟 API 调用
- 支持美股、外汇、加密货币
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, RateLimitError

logger = logging.getLogger(__name__)


class FinnhubFetcher(BaseFetcher):
    """
    Finnhub 数据源实现
    
    优先级：1（美股最高优先级）
    数据来源：Finnhub API
    
    特点：
    - 专为美股设计
    - 免费 API，60次/分钟
    - 支持实时和历史K线数据
    """
    
    name = "FinnhubFetcher"
    priority = 1  # 美股最高优先级
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 FinnhubFetcher
        
        Args:
            api_key: Finnhub API Key（可选，默认从环境变量读取）
        """
        import os
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        self._client = None
        
        if not self.api_key:
            logger.warning("未配置 FINNHUB_API_KEY，FinnhubFetcher 将不可用")
    
    def _get_client(self):
        """懒加载 Finnhub 客户端"""
        if self._client is None and self.api_key:
            try:
                import finnhub
                self._client = finnhub.Client(api_key=self.api_key)
                logger.info("Finnhub 客户端初始化成功")
            except ImportError:
                logger.error("未安装 finnhub-python 库，请运行: pip install finnhub-python")
                raise DataFetchError("缺少 finnhub-python 依赖")
        return self._client
    
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        return self.api_key is not None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Finnhub 获取美股K线数据
        
        Args:
            stock_code: 美股代码（如 AAPL, TSLA）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            原始数据 DataFrame
        """
        if not self.is_available():
            raise DataFetchError("Finnhub API Key 未配置")
        
        client = self._get_client()
        
        # 转换日期为 Unix 时间戳
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        # 确保代码大写
        symbol = stock_code.strip().upper()
        
        logger.info(f"[Finnhub] 获取 {symbol} K线数据: {start_date} ~ {end_date}")
        
        try:
            # 调用 Finnhub API 获取日K线
            res = client.stock_candles(symbol, 'D', start_ts, end_ts)
            
            if res.get('s') == 'no_data':
                logger.warning(f"[Finnhub] {symbol} 无数据返回")
                return pd.DataFrame()
            
            # 转换为 DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(res['t'], unit='s'),
                'open': res['o'],
                'high': res['h'],
                'low': res['l'],
                'close': res['c'],
                'volume': res['v'],
            })
            
            logger.info(f"[Finnhub] {symbol} 获取成功，共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"[Finnhub] 获取 {symbol} 失败: {e}")
            raise DataFetchError(f"Finnhub API 调用失败: {e}")
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Finnhub 数据
        
        Finnhub 返回的数据已经是标准格式，主要需要：
        1. 添加 amount 列（Finnhub 不提供成交额）
        2. 计算涨跌幅
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 添加股票代码
        df['code'] = stock_code.upper()
        
        # 估算成交额（价格 * 成交量）
        df['amount'] = df['close'] * df['volume']
        
        # 计算涨跌幅
        df['pct_chg'] = df['close'].pct_change() * 100
        df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
        
        # 选择标准列
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        existing_cols = [c for c in standard_cols if c in df.columns]
        
        return df[existing_cols]
    
    def get_realtime_quote(self, stock_code: str) -> Optional[dict]:
        """
        获取美股实时行情
        
        Args:
            stock_code: 美股代码
            
        Returns:
            实时行情字典，包含 price, change, percent_change 等
        """
        if not self.is_available():
            return None
        
        client = self._get_client()
        symbol = stock_code.strip().upper()
        
        try:
            quote = client.quote(symbol)
            
            return {
                'symbol': symbol,
                'price': quote.get('c', 0),  # 当前价
                'change': quote.get('d', 0),  # 涨跌额
                'percent_change': quote.get('dp', 0),  # 涨跌幅
                'high': quote.get('h', 0),  # 最高价
                'low': quote.get('l', 0),  # 最低价
                'open': quote.get('o', 0),  # 开盘价
                'prev_close': quote.get('pc', 0),  # 昨收价
                'timestamp': quote.get('t', 0),  # 时间戳
            }
            
        except Exception as e:
            logger.warning(f"[Finnhub] 获取 {symbol} 实时行情失败: {e}")
            return None
