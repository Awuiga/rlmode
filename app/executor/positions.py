from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(self, side: str, price: float, qty: float):
        if qty <= 0:
            return
        if side == "BUY":
            signed = qty
        else:
            signed = -qty

        if self.qty == 0:
            self.qty = signed
            self.avg_price = price
            return

        # If same direction, update average price
        if (self.qty > 0 and signed > 0) or (self.qty < 0 and signed < 0):
            total_cost = self.avg_price * abs(self.qty) + price * abs(qty)
            self.qty += signed
            if self.qty != 0:
                self.avg_price = total_cost / (abs(self.qty))
        else:
            # Closing position partially or fully
            close_qty = min(abs(self.qty), abs(qty))
            pnl = (price - self.avg_price) * close_qty * (1 if self.qty > 0 else -1)
            self.realized_pnl += pnl
            # Reduce position
            if abs(qty) > abs(self.qty):
                # Flips direction
                remainder = signed + (-self.qty)
                self.qty = remainder
                self.avg_price = price
            else:
                # Same direction remains
                self.qty += signed
                if self.qty == 0:
                    self.avg_price = 0.0

