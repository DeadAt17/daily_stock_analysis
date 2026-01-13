# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器
===================================

设计模式：策略模式 (Strategy Pattern)
- BaseFetcher: 抽象基类，定义统一接口
- DataFetcherManager: 策略管理器，实现自动切换

防封禁策略：
1. 每个 Fetcher 内置流控逻辑
2. 失败自动切换到下一个数据源
3. 指数退避重试机制
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# 配置日志
logger = logging.getLogger(__name__)


# === 标准化列名定义 ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']


# {{ Eddie Peng: Add - 市场类型枚举和识别函数，支持A股和美股。20260113 }}
class MarketType:
    """市场类型枚举"""
    CN = "CN"  # A股（中国大陆）
    US = "US"  # 美股


def detect_market(stock_code: str) -> str:
    """
    自动识别股票代码所属市场
    
    识别规则：
    - A股：6位纯数字（如 600519, 000001, 300750）
    - 美股：1-5位字母组合（如 AAPL, TSLA, NVDA, MSFT）
    
    Args:
        stock_code: 股票代码
        
    Returns:
        MarketType.CN 或 MarketType.US
    """
    code = stock_code.strip().upper()
    
    # 去除可能的后缀
    for suffix in ['.SH', '.SZ', '.SS']:
        code = code.replace(suffix, '')
    
    # A股：6位纯数字
    if code.isdigit() and len(code) == 6:
        return MarketType.CN
    
    # 美股：1-5位字母（部分美股可能包含数字，如 BRK.A）
    if code.replace('.', '').replace('-', '').isalpha() or (
        len(code) <= 5 and code[0].isalpha()
    ):
        return MarketType.US
    
    # 默认按A股处理（向后兼容）
    logger.warning(f"无法识别股票代码 {stock_code} 的市场类型，默认按A股处理")
    return MarketType.CN


class DataFetchError(Exception):
    """数据获取异常基类"""
    pass


class RateLimitError(DataFetchError):
    """API 速率限制异常"""
    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常"""
    pass


class BaseFetcher(ABC):
    """
    数据源抽象基类
    
    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    3. 实现通用的技术指标计算
    
    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """
    
    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）
        
        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            原始数据 DataFrame（列名因数据源而异）
        """
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）
        
        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass
    
    def get_daily_data(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）
        
        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名
        4. 计算技术指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）
            
        Returns:
            标准化的 DataFrame，包含技术指标
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 默认获取最近 30 个交易日（按日历日估算，多取一些）
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")
        
        try:
            # Step 1: 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # Step 2: 标准化列名
            df = self._normalize_data(raw_df, stock_code)
            
            # Step 3: 数据清洗
            df = self._clean_data(df)
            
            # Step 4: 计算技术指标
            df = self._calculate_indicators(df)
            
            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        处理：
        1. 确保日期列格式正确
        2. 数值类型转换
        3. 去除空值行
        4. 按日期排序
        """
        df = df.copy()
        
        # 确保日期列为 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 数值列类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除关键列为空的行
        df = df.dropna(subset=['close', 'volume'])
        
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        计算指标：
        - MA5, MA10, MA20: 移动平均线
        - Volume_Ratio: 量比（今日成交量 / 5日平均成交量）
        """
        df = df.copy()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # 量比：当日成交量 / 5日平均成交量
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # 保留2位小数
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df
    
    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        智能随机休眠（Jitter）
        
        防封禁策略：模拟人类行为的随机延迟
        在请求之间加入不规则的等待时间
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


class DataFetcherManager:
    """
    数据源策略管理器
    
    {{ Eddie Peng: Modify - 支持根据市场类型选择不同数据源，并根据配置决定是否初始化。20260113 }}
    
    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口
    4. 根据市场类型（A股/美股）选择合适的数据源
    5. 根据配置决定是否初始化对应市场的数据源
    
    切换策略：
    - A股：AkshareFetcher -> TushareFetcher -> BaostockFetcher -> YfinanceFetcher
    - 美股：FinnhubFetcher -> YfinanceFetcher
    """
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._cn_fetchers: List[BaseFetcher] = []  # A股数据源
        self._us_fetchers: List[BaseFetcher] = []  # 美股数据源
        self._fetchers: List[BaseFetcher] = []     # 兼容旧接口
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
            self._cn_fetchers = self._fetchers  # 默认作为A股源
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()
    
    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表
        
        {{ Eddie Peng: Modify - 根据配置决定是否初始化A股/美股数据源。20260113 }}
        
        A股数据源（按优先级）：
        1. AkshareFetcher (Priority 1)
        2. TushareFetcher (Priority 2)
        3. BaostockFetcher (Priority 3)
        4. YfinanceFetcher (Priority 4)
        
        美股数据源（按优先级）：
        1. FinnhubFetcher (Priority 1)
        2. YfinanceFetcher (Priority 2)
        """
        from config import get_config
        config = get_config()
        
        # 检查是否配置了A股列表
        has_cn_stocks = bool(config.cn_stock_list)
        # 检查是否配置了美股列表
        has_us_stocks = bool(config.us_stock_list)
        
        # 只有配置了A股列表才初始化A股数据源
        if has_cn_stocks:
            from .akshare_fetcher import AkshareFetcher
            from .tushare_fetcher import TushareFetcher
            from .baostock_fetcher import BaostockFetcher
            from .yfinance_fetcher import YfinanceFetcher
            
            self._cn_fetchers = [
                AkshareFetcher(),
                TushareFetcher(),
                BaostockFetcher(),
                YfinanceFetcher(),
            ]
            self._cn_fetchers.sort(key=lambda f: f.priority)
            logger.info(f"已初始化 A股数据源: " + 
                       ", ".join([f.name for f in self._cn_fetchers]))
        else:
            logger.info("未配置 CN_STOCK_LIST，跳过 A股数据源初始化")
        
        # 只有配置了美股列表才初始化美股数据源
        if has_us_stocks:
            from .yfinance_fetcher import YfinanceFetcher
            
            try:
                from .finnhub_fetcher import FinnhubFetcher
                finnhub = FinnhubFetcher()
                if finnhub.is_available():
                    self._us_fetchers.append(finnhub)
                    logger.info("Finnhub 数据源已启用（美股）")
                else:
                    logger.info("Finnhub API Key 未配置，美股将使用 YfinanceFetcher")
            except ImportError:
                logger.warning("finnhub-python 未安装，美股将使用 YfinanceFetcher")
            
            # YfinanceFetcher 作为美股备用
            self._us_fetchers.append(YfinanceFetcher())
            logger.info(f"已初始化 美股数据源: " + 
                       ", ".join([f.name for f in self._us_fetchers]))
        else:
            logger.info("未配置 US_STOCK_LIST，跳过 美股数据源初始化")
        
        # 兼容旧接口（默认使用A股源）
        self._fetchers = self._cn_fetchers if self._cn_fetchers else self._us_fetchers
    
    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)
    
    def get_daily_data(
        self, 
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据（自动切换数据源）
        
        {{ Eddie Peng: Modify - 根据市场类型自动选择数据源。20260113 }}
        
        故障切换策略：
        1. 自动识别股票市场（A股/美股）
        2. 选择对应市场的数据源列表
        3. 从最高优先级数据源开始尝试
        4. 捕获异常后自动切换到下一个
        5. 所有数据源失败后抛出详细异常
        
        Args:
            stock_code: 股票代码（A股如 600519，美股如 AAPL）
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数
            
        Returns:
            Tuple[DataFrame, str]: (数据, 成功的数据源名称)
            
        Raises:
            DataFetchError: 所有数据源都失败时抛出
        """
        # 识别市场类型，选择对应数据源
        market = detect_market(stock_code)
        
        if market == MarketType.US:
            fetchers = self._us_fetchers
            market_name = "美股"
        else:
            fetchers = self._cn_fetchers
            market_name = "A股"
        
        # 检查对应市场的数据源是否已初始化
        if not fetchers:
            raise DataFetchError(
                f"未初始化{market_name}数据源（请检查配置文件中是否设置了对应的股票列表）"
            )
        
        logger.info(f"[{stock_code}] 识别为 {market_name}，使用对应数据源")
        
        errors = []
        
        for fetcher in fetchers:
            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )
                
                if df is not None and not df.empty:
                    logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                    return df, fetcher.name
                    
            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # 继续尝试下一个数据源
                continue
        
        # 所有数据源都失败
        error_summary = f"所有{market_name}数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)
    
    @property
    def available_fetchers(self) -> List[str]:
        """返回可用数据源名称列表"""
        return [f.name for f in self._fetchers]
    
    @property
    def cn_fetchers_available(self) -> bool:
        """A股数据源是否可用"""
        return bool(self._cn_fetchers)
    
    @property
    def us_fetchers_available(self) -> bool:
        """美股数据源是否可用"""
        return bool(self._us_fetchers)
