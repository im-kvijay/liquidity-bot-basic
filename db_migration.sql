-- Database migration script to add new fields for production bot
-- Run this before starting the bot with new features

-- Add peak price tracking fields to positions table
ALTER TABLE positions ADD COLUMN peak_price REAL DEFAULT NULL;
ALTER TABLE positions ADD COLUMN peak_timestamp TEXT DEFAULT NULL;

-- Add drift monitoring fields to fills table  
ALTER TABLE fills ADD COLUMN expected_slippage_bps REAL DEFAULT NULL;
ALTER TABLE fills ADD COLUMN actual_slippage_bps REAL DEFAULT NULL;
ALTER TABLE fills ADD COLUMN expected_fee_usd REAL DEFAULT NULL;
ALTER TABLE fills ADD COLUMN expected_price_usd REAL DEFAULT NULL;
ALTER TABLE fills ADD COLUMN actual_price_usd REAL DEFAULT NULL;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_fills_expected_actual ON fills(expected_slippage_bps, actual_slippage_bps);
CREATE INDEX IF NOT EXISTS idx_positions_peak ON positions(peak_price, peak_timestamp);

-- Verify migration
SELECT 'Migration completed successfully' as status;
