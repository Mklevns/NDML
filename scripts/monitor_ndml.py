#!/usr/bin/env python3
"""Real-time monitoring dashboard for NDML system"""

import asyncio
import aiohttp
import time
import argparse
from datetime import datetime
import curses
import json
from typing import Dict, Any, List
import numpy as np


class NDMLMonitor:
    """Real-time monitoring for NDML system"""

    def __init__(self, server_url: str, refresh_rate: int = 1):
        self.server_url = server_url
        self.refresh_rate = refresh_rate
        self.running = True
        self.stats_history = []
        self.max_history = 100

    async def fetch_stats(self) -> Dict[str, Any]:
        """Fetch current stats from NDML server"""
        async with aiohttp.ClientSession() as session:
            try:
                # Get health status
                async with session.get(f"{self.server_url}/health") as resp:
                    health = await resp.json()

                # Get memory stats
                async with session.get(f"{self.server_url}/memory/stats") as resp:
                    memory_stats = await resp.json()

                return {
                    'timestamp': time.time(),
                    'health': health,
                    'memory': memory_stats
                }
            except Exception as e:
                return {
                    'timestamp': time.time(),
                    'error': str(e)
                }

    def draw_dashboard(self, stdscr, stats: Dict[str, Any]):
        """Draw monitoring dashboard"""
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Title
        title = "NDML System Monitor"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        # Timestamp
        timestamp = datetime.fromtimestamp(stats['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        stdscr.addstr(1, 0, f"Last Update: {timestamp}")

        if 'error' in stats:
            stdscr.addstr(3, 0, f"ERROR: {stats['error']}", curses.color_pair(1))
            return

        # Health Status
        health = stats['health']
        status_color = curses.color_pair(2) if health['status'] == 'healthy' else curses.color_pair(1)
        stdscr.addstr(3, 0, f"Status: {health['status'].upper()}", status_color)
        stdscr.addstr(4, 0, f"Uptime: {health['uptime']:.1f} seconds")

        # GPU Status
        if health['gpu_available']:
            gpu_usage = health.get('gpu_memory_used', 0) / health.get('gpu_memory_total', 1) * 100
            stdscr.addstr(5, 0, f"GPU Memory: {gpu_usage:.1f}%")
        else:
            stdscr.addstr(5, 0, "GPU: Not Available")

        # Memory System Stats
        memory = stats['memory']
        row = 7

        stdscr.addstr(row, 0, "=== Memory System ===", curses.A_BOLD)
        row += 1

        # Gateway stats
        gateway_stats = memory.get('gateway_stats', {})
        stdscr.addstr(row, 0, f"Total Queries: {gateway_stats.get('total_queries', 0)}")
        row += 1
        stdscr.addstr(row, 0, f"Total Updates: {gateway_stats.get('total_updates', 0)}")
        row += 1

        # System summary
        summary = memory.get('system_summary', {})
        stdscr.addstr(row, 0, f"Total Memories: {summary.get('total_memories', 0)}")
        row += 1
        stdscr.addstr(row, 0, f"Average Utilization: {summary.get('average_utilization', 0):.1%}")
        row += 2

        # Cluster stats
        stdscr.addstr(row, 0, "=== Cluster Statistics ===", curses.A_BOLD)
        row += 1

        cluster_stats = memory.get('cluster_stats', [])
        for cluster in cluster_stats[:5]:  # Show first 5 clusters
            utilization = cluster.get('utilization', 0) * 100
            color = curses.color_pair(2) if utilization < 80 else curses.color_pair(1)
            stdscr.addstr(row, 0,
                          f"Cluster {cluster['cluster_id']}: "
                          f"{cluster.get('total_memories', 0)} memories, "
                          f"{utilization:.1f}% full", color)
            row += 1

        # Performance metrics
        if len(self.stats_history) > 1:
            row += 1
            stdscr.addstr(row, 0, "=== Performance ===", curses.A_BOLD)
            row += 1

            # Calculate rates
            time_diff = self.stats_history[-1]['timestamp'] - self.stats_history[-2]['timestamp']
            if time_diff > 0:
                queries_diff = (self.stats_history[-1]['memory']['gateway_stats']['total_queries'] -
                                self.stats_history[-2]['memory']['gateway_stats']['total_queries'])
                qps = queries_diff / time_diff

                updates_diff = (self.stats_history[-1]['memory']['gateway_stats']['total_updates'] -
                                self.stats_history[-2]['memory']['gateway_stats']['total_updates'])
                ups = updates_diff / time_diff

                stdscr.addstr(row, 0, f"Queries/sec: {qps:.1f}")
                row += 1
                stdscr.addstr(row, 0, f"Updates/sec: {ups:.1f}")

        # Instructions
        stdscr.addstr(height - 2, 0, "Press 'q' to quit, 'r' to refresh", curses.A_DIM)

    async def run_monitor(self, stdscr):
        """Run the monitoring loop"""
        # Setup colors
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

        # Make getch non-blocking
        stdscr.nodelay(True)

        while self.running:
            # Fetch stats
            stats = await self.fetch_stats()

            # Add to history
            if 'error' not in stats:
                self.stats_history.append(stats)
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)

            # Draw dashboard
            self.draw_dashboard(stdscr, stats)
            stdscr.refresh()

            # Check for user input
            key = stdscr.getch()
            if key == ord('q'):
                self.running = False
            elif key == ord('r'):
                continue  # Immediate refresh

            # Wait for next update
            await asyncio.sleep(self.refresh_rate)

    def start(self):
        """Start the monitor"""
        curses.wrapper(lambda stdscr: asyncio.run(self.run_monitor(stdscr)))


def main():
    parser = argparse.ArgumentParser(description='Monitor NDML System')
    parser.add_argument('--server', default='http://localhost:8000', help='NDML server URL')
    parser.add_argument('--refresh', type=int, default=1, help='Refresh rate in seconds')
    args = parser.parse_args()

    monitor = NDMLMonitor(args.server, args.refresh)
    monitor.start()


if __name__ == "__main__":
    main()