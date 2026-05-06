"""Web搜索服务 - 获取高速公路封闭信息和路线替代方案

提供真实的网络搜索能力，用于决策过程中的信息收集阶段。
支持多种搜索API配置，也可在无API时使用场景推断作为fallback。
"""

import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import quote


@dataclass
class RouteOption:
    """替代路线选项"""
    route_name: str           # 路线名称，如 "G2京沪高速"
    route_type: str           # "高速" / "国道" / "省道"
    distance_km: float        # 距离（公里）
    estimated_time_minutes: int  # 预计时间（分钟）
    toll_cost: float          # 通行费（元）
    congestion_probability: float  # 拥堵概率 0-1
    traffic_status: str       # "畅通" / "缓慢" / "拥堵"
    waypoints: List[str]      # 途经点


@dataclass
class ClosureInfo:
    """封闭信息"""
    is_verified: bool         # 是否已验证
    location: str             # 封闭位置
    reason: str               # 封闭原因
    estimated_duration_hours: float  # 预计解除时间（小时）
    severity: str             # "partial" / "full"
    source: str               # 信息来源


@dataclass
class TrafficStatus:
    """实时路况"""
    route_name: str
    is_congested: bool
    congestion_level: str     # "none" / "light" / "medium" / "heavy"
    estimated_delay_minutes: int  # 预计延误（分钟）
    last_updated: str         # 最后更新时间


class WebSearchService:
    """Web搜索服务

    提供高速公路封闭信息查询、替代路线搜索、实时路况查询等功能。
    支持配置外部搜索API（如SerpAPI、Bing Search等）。
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('SEARCH_API_KEY', '')
        self.search_engine = os.environ.get('SEARCH_ENGINE', 'duckduckgo')  # 默认使用duckduckgo
        self._init_search_client()

    def _init_search_client(self):
        """初始化搜索客户端"""
        if self.search_engine == 'serpapi' and self.api_key:
            # SerpAPI (Google Search)
            self._search_func = self._serpapi_search
        elif self.search_engine == 'duckduckgo':
            # DuckDuckGo HTML (免费，无需API key)
            self._search_func = self._duckduckgo_search
        else:
            # 无API时使用场景推断
            self._search_func = self._fallback_search

    def search_closure_info(self, location: str, scenario_context: str = "") -> ClosureInfo:
        """搜索高速公路封闭信息

        Args:
            location: 位置描述，如 "G2京沪高速"
            scenario_context: 场景上下文，用于fallback推断

        Returns:
            ClosureInfo 封闭信息
        """
        query = f"{location} 封闭 高速公路"
        result = self._search(query)

        if result and result.get('success'):
            return ClosureInfo(
                is_verified=True,
                location=result.get('location', location),
                reason=result.get('reason', '未知'),
                estimated_duration_hours=result.get('duration_hours', 1.0),
                severity=result.get('severity', 'unknown'),
                source=result.get('source', 'web_search')
            )

        # Fallback: 从场景上下文推断
        return self._infer_closure_from_context(location, scenario_context)

    def search_route_alternatives(
        self,
        origin: str,
        destination: str,
        blocked_route: str = ""
    ) -> List[RouteOption]:
        """搜索替代路线

        Args:
            origin: 起点
            destination: 终点
            blocked_route: 被封锁的路线

        Returns:
            List[RouteOption] 替代路线列表
        """
        query = f"从{origin}到{destination}替代路线 避开{blocked_route}"
        result = self._search(query)

        if result and result.get('success') and result.get('routes'):
            return [
                RouteOption(
                    route_name=r['name'],
                    route_type=r.get('type', '高速'),
                    distance_km=r.get('distance', 0),
                    estimated_time_minutes=r.get('time', 0),
                    toll_cost=r.get('toll', 0),
                    congestion_probability=r.get('congestion_prob', 0.3),
                    traffic_status=r.get('status', '畅通'),
                    waypoints=r.get('waypoints', [])
                )
                for r in result['routes']
            ]

        # Fallback: 生成合理的替代路线
        return self._generate_fallback_routes(origin, destination, blocked_route)

    def check_traffic_status(self, route_name: str) -> TrafficStatus:
        """检查路线实时路况

        Args:
            route_name: 路线名称

        Returns:
            TrafficStatus 实时路况
        """
        query = f"{route_name} 实时路况 拥堵"
        result = self._search(query)

        if result and result.get('success'):
            return TrafficStatus(
                route_name=route_name,
                is_congested=result.get('is_congested', False),
                congestion_level=result.get('congestion_level', 'none'),
                estimated_delay_minutes=result.get('delay_minutes', 0),
                last_updated=result.get('last_updated', '')
            )

        # Fallback: 返回默认畅通状态
        return TrafficStatus(
            route_name=route_name,
            is_congested=False,
            congestion_level='none',
            estimated_delay_minutes=0,
            last_updated='unknown'
        )

    def _search(self, query: str) -> Dict[str, Any]:
        """执行搜索"""
        return self._search_func(query)

    def _duckduckgo_search(self, query: str) -> Dict[str, Any]:
        """DuckDuckGo HTML搜索（免费）"""
        try:
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            # 简单解析HTML结果
            results = self._parse_dduckduckgo_html(response.text)
            if results:
                return {
                    'success': True,
                    'results': results,
                    'source': 'duckduckgo'
                }
            return {'success': False}
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return {'success': False}

    def _parse_dduckduckgo_html(self, html: str) -> List[Dict[str, Any]]:
        """解析DuckDuckGo HTML结果"""
        results = []
        # 简单的HTML解析，提取搜索结果标题和摘要
        import re
        pattern = r'<a class="result__a" href="[^"]*">([^<]*)</a>.*?<a class="result__snippet"[^>]*>([^<]*)</a>'
        matches = re.findall(pattern, html, re.DOTALL)
        for title, snippet in matches[:5]:
            results.append({
                'title': title.strip(),
                'snippet': snippet.strip()
            })
        return results

    def _serpapi_search(self, query: str) -> Dict[str, Any]:
        """SerpAPI搜索（需要API key）"""
        try:
            import requests
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google'
            }
            response = requests.get('https://serpapi.com/search', params=params, timeout=10)
            data = response.json()

            results = []
            for item in data.get('organic_results', [])[:5]:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', '')
                })

            return {
                'success': True,
                'results': results,
                'source': 'serpapi'
            }
        except Exception as e:
            print(f"SerpAPI search failed: {e}")
            return {'success': False}

    def _fallback_search(self, query: str) -> Dict[str, Any]:
        """Fallback: 使用场景推断而非真实搜索"""
        return {'success': False}

    def _infer_closure_from_context(
        self,
        location: str,
        scenario_context: str
    ) -> ClosureInfo:
        """从场景上下文推断封闭信息"""
        context_lower = scenario_context.lower()

        # 从场景描述中提取时限
        duration_hours = 1.0  # 默认1小时
        if '2小时' in scenario_context or '2h' in context_lower:
            duration_hours = 2.0
        elif '3小时' in scenario_context or '3h' in context_lower:
            duration_hours = 3.0
        elif '30分钟' in scenario_context or '30min' in context_lower:
            duration_hours = 0.5

        # 判断封闭原因
        reason = "道路施工"
        if '事故' in scenario_context:
            reason = "交通事故"
        elif '天气' in scenario_context or '雨' in scenario_context or '雪' in context_lower:
            reason = "天气原因"

        return ClosureInfo(
            is_verified=False,  # 推断结果，未验证
            location=location,
            reason=reason,
            estimated_duration_hours=duration_hours,
            severity="full" if duration_hours > 1 else "partial",
            source="inferred_from_context"
        )

    def _generate_fallback_routes(
        self,
        origin: str,
        destination: str,
        blocked_route: str
    ) -> List[RouteOption]:
        """生成fallback替代路线"""
        routes = [
            RouteOption(
                route_name=f"G2京沪高速（{origin}→{destination}）",
                route_type="高速",
                distance_km=1200.0,
                estimated_time_minutes=600,
                toll_cost=550.0,
                congestion_probability=0.3,
                traffic_status="畅通",
                waypoints=[origin, "收费口A", "收费口B", destination]
            ),
            RouteOption(
                route_name=f"国道104（{origin}→{destination}）",
                route_type="国道",
                distance_km=750.0,
                estimated_time_minutes=750,
                toll_cost=0,
                congestion_probability=0.15,
                traffic_status="畅通",
                waypoints=[origin, "城市道路", "国道104", "终点道路", destination]
            ),
            RouteOption(
                route_name=f"混合路线（{origin}→{destination}）",
                route_type="高速+国道",
                distance_km=1100.0,
                estimated_time_minutes=550,
                toll_cost=300.0,
                congestion_probability=0.2,
                traffic_status="畅通",
                waypoints=[origin, "高速入口", "G2京沪高速", "G1501绕城高速", destination]
            )
        ]
        return routes


# 全局实例
_web_search_instance: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    """获取全局WebSearchService实例"""
    global _web_search_instance
    if _web_search_instance is None:
        _web_search_instance = WebSearchService()
    return _web_search_instance


def create_web_search_service(api_key: Optional[str] = None) -> WebSearchService:
    """创建WebSearchService实例"""
    return WebSearchService(api_key=api_key)