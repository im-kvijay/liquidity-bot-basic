"""Thread-safe metrics registry with histogram support and drift monitoring."""

from __future__ import annotations

import math
import re
import threading
import time
from collections import defaultdict, deque
from statistics import mean, median
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

_METRIC_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_:]")


def _sanitize_metric_name(name: str) -> str:
    """Return a Prometheus-safe metric name."""

    sanitized = _METRIC_SANITIZE_RE.sub("_", name)
    if not sanitized:
        return "_"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


class MetricsRegistry:
    """In-memory metrics store backing the dashboard and Prometheus endpoint with drift monitoring."""

    def __init__(self, *, max_hist_samples: int = 1024) -> None:
        self._lock = threading.RLock()
        self._counters: MutableMapping[str, float] = defaultdict(float)
        self._gauges: MutableMapping[str, float] = {}
        self._histograms: MutableMapping[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_hist_samples)
        )
        self._mappings: MutableMapping[str, Dict[str, float]] = {}
        
        # Drift monitoring for trading metrics
        self._drift_tracking: MutableMapping[str, List[Tuple[float, float]]] = defaultdict(list)  # timestamp, value
        self._drift_window_seconds = 3600  # 1 hour window
        self._drift_alert_thresholds = {
            'slippage_drift_pct': 25.0,  # Alert if slippage deteriorates > 25%
            'fee_drift_pct': 25.0,       # Alert if fees increase > 25%
            'edge_drift_pct': 20.0       # Alert if edge decreases > 20%
        }

    def increment(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += amount

    def get(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(float(value))

    def set_mapping(self, name: str, values: Mapping[str, float]) -> None:
        with self._lock:
            self._mappings[name] = {key: float(val) for key, val in values.items()}

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {key: self._histogram_stats(values) for key, values in self._histograms.items()}
            mappings = {key: dict(value) for key, value in self._mappings.items()}
        return {
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
            "mappings": mappings,
        }

    def export_prometheus(self) -> str:
        snap = self.snapshot()
        lines = []
        for name, value in snap["counters"].items():
            sanitized = _sanitize_metric_name(name)
            lines.append(f"# TYPE {sanitized} counter")
            lines.append(f"{sanitized} {value}")
        for name, value in snap["gauges"].items():
            sanitized = _sanitize_metric_name(name)
            lines.append(f"# TYPE {sanitized} gauge")
            lines.append(f"{sanitized} {value}")
        for name, stats in snap["histograms"].items():
            if not stats:
                continue
            base = _sanitize_metric_name(name)
            lines.append(f"# TYPE {base} summary")
            for quantile in ("p50", "p90", "p99"):
                if quantile in stats:
                    lines.append(f"{base}{{quantile=\"{quantile}\"}} {stats[quantile]}")
            lines.append(f"{base}_count {stats.get('count', 0)}")
            if "avg" in stats:
                lines.append(f"{base}_avg {stats['avg']}")
        return "\n".join(lines) + "\n"

    def track_trading_metric(self, metric_name: str, expected_value: float, actual_value: float) -> Optional[Dict]:
        """Track expected vs actual values for drift monitoring."""
        with self._lock:
            current_time = time.time()
            
            # Store the drift data (timestamp, expected, actual)
            drift_key = f"{metric_name}_drift"
            self._drift_tracking[drift_key].append((current_time, expected_value, actual_value))
            
            # Clean old data (keep only last hour)
            cutoff_time = current_time - self._drift_window_seconds
            self._drift_tracking[drift_key] = [
                entry for entry in self._drift_tracking[drift_key] 
                if entry[0] > cutoff_time
            ]
            
            # Calculate drift if we have enough data
            if len(self._drift_tracking[drift_key]) >= 5:  # Need at least 5 samples
                return self._calculate_drift_alert(metric_name, drift_key)
            
            return None

    def _calculate_drift_alert(self, metric_name: str, drift_key: str) -> Optional[Dict]:
        """Calculate drift and return alert if threshold exceeded."""
        try:
            data = self._drift_tracking[drift_key]
            
            if len(data) < 5:
                return None
            
            # Calculate median expected vs actual over the window
            expected_values = [entry[1] for entry in data]
            actual_values = [entry[2] for entry in data]
            
            median_expected = median(expected_values)
            median_actual = median(actual_values)
            
            if median_expected == 0:
                return None
            
            # Calculate drift percentage
            drift_pct = ((median_actual - median_expected) / median_expected) * 100
            
            # Check if drift exceeds threshold
            threshold_key = f"{metric_name}_drift_pct"
            threshold = self._drift_alert_thresholds.get(threshold_key, 25.0)
            
            if abs(drift_pct) > threshold:
                alert = {
                    'metric': metric_name,
                    'drift_pct': drift_pct,
                    'threshold': threshold,
                    'median_expected': median_expected,
                    'median_actual': median_actual,
                    'sample_count': len(data),
                    'alert_type': 'drift_alert'
                }
                
                # Record alert metric
                self.increment(f"{metric_name}_drift_alert", 1)
                self.gauge(f"{metric_name}_drift_pct", drift_pct)
                
                return alert
            
            # Record drift even if no alert
            self.gauge(f"{metric_name}_drift_pct", drift_pct)
            
            return None
            
        except Exception as e:
            # Log error but don't fail
            return None

    def get_drift_summary(self) -> Dict[str, Dict]:
        """Get summary of all tracked drift metrics."""
        with self._lock:
            summary = {}
            
            for drift_key, data in self._drift_tracking.items():
                if not data:
                    continue
                
                metric_name = drift_key.replace('_drift', '')
                
                # Calculate current drift
                expected_values = [entry[1] for entry in data]
                actual_values = [entry[2] for entry in data]
                
                if expected_values and actual_values:
                    median_expected = median(expected_values)
                    median_actual = median(actual_values)
                    
                    drift_pct = 0.0
                    if median_expected != 0:
                        drift_pct = ((median_actual - median_expected) / median_expected) * 100
                    
                    summary[metric_name] = {
                        'drift_pct': drift_pct,
                        'median_expected': median_expected,
                        'median_actual': median_actual,
                        'sample_count': len(data),
                        'window_hours': self._drift_window_seconds / 3600
                    }
            
            return summary

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._mappings.clear()
            self._drift_tracking.clear()

    def _histogram_stats(self, values: Iterable[float]) -> Dict[str, float]:
        data = list(values)
        if not data:
            return {}
        data.sort()
        count = len(data)
        return {
            "count": float(count),
            "avg": mean(data),
            "p50": self._percentile(data, 0.5),
            "p90": self._percentile(data, 0.9),
            "p99": self._percentile(data, 0.99),
        }

    def _percentile(self, data: Iterable[float], percentile: float) -> float:
        items = list(data)
        if not items:
            return 0.0
        index = max(int(math.ceil(percentile * len(items))) - 1, 0)
        return float(items[min(index, len(items) - 1)])


METRICS = MetricsRegistry()


__all__ = ["METRICS", "MetricsRegistry"]
