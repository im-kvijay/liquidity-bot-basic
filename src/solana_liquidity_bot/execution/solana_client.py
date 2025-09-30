"""Solana RPC client wrapper with retry logic and simulation."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - exercised when dependency installed
    from solana.rpc.api import Client
    from solana.rpc.types import TxOpts
    from solana.transaction import Transaction
    from solders.pubkey import Pubkey
    from solders.signature import Signature
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    from .solana_compat import Client, TxOpts, Transaction, Pubkey, Signature
from tenacity import retry, stop_after_attempt, wait_fixed

from ..config.settings import RPCConfig, get_app_config
from ..monitoring.logger import get_logger
from ..monitoring.metrics import METRICS


class SolanaClient:
    def __init__(self, config: Optional[RPCConfig] = None) -> None:
        self._config = config or get_app_config().rpc
        self._endpoints = [str(self._config.primary_url), *map(str, self._config.fallback_urls)]
        self._clients = [Client(endpoint, timeout=self._config.request_timeout) for endpoint in self._endpoints]
        self._logger = get_logger(__name__)
        
        # Priority fee tracking for adaptive fees
        self._recent_priority_fees: List[Tuple[float, int]] = []  # (timestamp, fee_lamports)
        self._priority_fee_cache_ttl = 30  # seconds

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def send_transaction(self, transaction, signers):
        last_exc: Optional[Exception] = None
        for endpoint, client in zip(self._endpoints, self._clients):
            try:
                response = client.send_transaction(transaction, *signers)
                self._logger.info("Submitted transaction: %s", response)
                return response
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._logger.warning("Transaction submission failed on %s: %s", endpoint, exc)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Transaction submission failed")

    def get_balance(self, public_key: str) -> float:
        response = self._clients[0].get_balance(public_key)
        value = response.get("result", {}).get("value", 0)
        return value / 1_000_000_000

    def health_check(self) -> bool:
        for endpoint, client in zip(self._endpoints, self._clients):
            try:
                response = client.get_health()
                if response.get("result") == "ok":
                    return True
            except Exception:  # noqa: BLE001
                self._logger.debug("Health check failed on %s", endpoint)
                continue
        return False

    def simulate_transaction(self, transaction: Transaction, signers: List = None) -> Dict:
        """Simulate transaction to validate success and estimate CU usage."""
        try:
            # Use the first available client for simulation
            client = self._clients[0]
            
            # Simulate the transaction
            opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            response = client.simulate_transaction(transaction, opts)
            
            if hasattr(response, 'value') and response.value:
                result = response.value
                
                # Extract simulation results
                simulation_result = {
                    'success': not result.err,
                    'error': str(result.err) if result.err else None,
                    'compute_units_consumed': getattr(result, 'units_consumed', 0),
                    'logs': getattr(result, 'logs', []),
                    'accounts': getattr(result, 'accounts', [])
                }
                
                # Log simulation results
                if simulation_result['success']:
                    self._logger.debug(
                        f"Transaction simulation successful: CU={simulation_result['compute_units_consumed']}"
                    )
                    METRICS.increment("transaction_simulation_success", 1)
                else:
                    self._logger.warning(
                        f"Transaction simulation failed: {simulation_result['error']}"
                    )
                    METRICS.increment("transaction_simulation_failure", 1)
                
                METRICS.observe("transaction_simulation_cu", simulation_result['compute_units_consumed'])
                return simulation_result
            else:
                return {
                    'success': False,
                    'error': 'No simulation result returned',
                    'compute_units_consumed': 0,
                    'logs': [],
                    'accounts': []
                }
                
        except Exception as e:
            self._logger.error(f"Transaction simulation error: {e}")
            METRICS.increment("transaction_simulation_error", 1)
            return {
                'success': False,
                'error': str(e),
                'compute_units_consumed': 200000,  # Conservative estimate
                'logs': [],
                'accounts': []
            }

    def get_recent_prioritization_fees(self, accounts: Optional[List[str]] = None) -> List[int]:
        """Get recent prioritization fees for adaptive fee calculation."""
        try:
            client = self._clients[0]
            
            # Get recent prioritization fees
            response = client.get_recent_prioritization_fees(accounts or [])
            
            if hasattr(response, 'value') and response.value:
                fees = [fee_info.prioritization_fee for fee_info in response.value]
                
                # Update cache
                current_time = time.time()
                self._recent_priority_fees = [(current_time, fee) for fee in fees]
                
                self._logger.debug(f"Retrieved {len(fees)} recent priority fees")
                METRICS.observe("recent_priority_fees_count", len(fees))
                
                return fees
            else:
                return []
                
        except Exception as e:
            self._logger.warning(f"Failed to get recent prioritization fees: {e}")
            return []

    def calculate_adaptive_priority_fee(self, percentile: float = 0.75, max_budget_usd: float = 0.05) -> int:
        """Calculate adaptive priority fee based on recent network conditions."""
        try:
            # Get recent fees
            fees = self.get_recent_prioritization_fees()
            
            if not fees:
                # Fallback to conservative estimate
                return 5000  # 5000 lamports
            
            # Calculate percentile
            fees_sorted = sorted(fees)
            index = int(len(fees_sorted) * percentile)
            percentile_fee = fees_sorted[min(index, len(fees_sorted) - 1)]
            
            # Convert USD budget to lamports using oracle price if available
            try:
                from ..ingestion.pricing import PriceOracle
                oracle = PriceOracle()
                sol_price = oracle.get_price("So11111111111111111111111111111111111111112") or 150.0
            except Exception:
                sol_price = 150.0  # Fallback
            
            max_budget_lamports = int((max_budget_usd / sol_price) * 1_000_000_000)
            
            # Cap the fee at the budget
            adaptive_fee = min(percentile_fee, max_budget_lamports)
            
            self._logger.debug(
                f"Adaptive priority fee: {adaptive_fee} lamports "
                f"(p{percentile*100:.0f} of {len(fees)} recent fees, "
                f"capped at ${max_budget_usd})"
            )
            
            METRICS.observe("adaptive_priority_fee_lamports", adaptive_fee)
            METRICS.observe("priority_fee_percentile", percentile_fee)
            
            return adaptive_fee
            
        except Exception as e:
            self._logger.error(f"Error calculating adaptive priority fee: {e}")
            return 5000  # Conservative fallback

    def simulate_then_send(self, transaction: Transaction, signers: List, 
                          max_cu_budget: int = 300000, priority_fee_lamports: Optional[int] = None) -> Dict:
        """Simulate transaction, estimate CU usage, then send with optimized fees."""
        try:
            # Step 1: Simulate transaction
            simulation = self.simulate_transaction(transaction, signers)
            
            if not simulation['success']:
                return {
                    'success': False,
                    'error': f"Simulation failed: {simulation['error']}",
                    'signature': None,
                    'simulation': simulation
                }
            
            # Step 2: Estimate compute budget
            estimated_cu = simulation['compute_units_consumed']
            if estimated_cu == 0:
                estimated_cu = 100000  # Conservative default
            
            # Add 20% buffer for safety
            cu_budget = min(int(estimated_cu * 1.2), max_cu_budget)
            
            # Step 3: Calculate adaptive priority fee if not provided
            if priority_fee_lamports is None:
                app_config = get_app_config()
                percentile = getattr(app_config.execution, 'priority_fee_percentile', 0.75)
                max_budget = getattr(app_config.execution, 'max_priority_fee_budget_usd', 0.05)
                priority_fee_lamports = self.calculate_adaptive_priority_fee(percentile, max_budget)
            
            # Step 4: Add compute budget and priority fee instructions to transaction
            try:
                from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
                
                # Calculate compute unit price (micro-lamports per CU)
                cu_price_micro_lamports = max(1, priority_fee_lamports // max(cu_budget, 1))
                
                # Create compute budget instructions
                cu_limit_ix = set_compute_unit_limit(cu_budget)
                cu_price_ix = set_compute_unit_price(cu_price_micro_lamports)
                
                # Add instructions to transaction (prepend for proper ordering)
                if hasattr(transaction, 'instructions'):
                    transaction.instructions.insert(0, cu_price_ix)
                    transaction.instructions.insert(0, cu_limit_ix)
                elif hasattr(transaction, 'add'):
                    # Create new transaction with compute budget instructions
                    transaction.add(cu_limit_ix, cu_price_ix)
                
                self._logger.info(
                    f"Added compute budget: {cu_budget} CU limit, "
                    f"{cu_price_micro_lamports} micro-lamports/CU price "
                    f"(total: {priority_fee_lamports} lamports)"
                )
                
            except ImportError:
                # Fallback if compute budget instructions not available
                self._logger.warning(
                    f"ComputeBudgetProgram not available - sending without CU instructions "
                    f"(estimated: {cu_budget} CU, {priority_fee_lamports} lamports)"
                )
            except Exception as e:
                self._logger.warning(f"Failed to add compute budget instructions: {e}")
                # Continue without compute budget instructions
            
            # Step 5: Send transaction
            response = self.send_transaction(transaction, signers)
            
            return {
                'success': True,
                'error': None,
                'signature': response.get('result'),
                'simulation': simulation,
                'cu_budget': cu_budget,
                'priority_fee': priority_fee_lamports
            }
            
        except Exception as e:
            self._logger.error(f"Error in simulate_then_send: {e}")
            return {
                'success': False,
                'error': str(e),
                'signature': None,
                'simulation': None
            }


__all__ = ["SolanaClient"]
