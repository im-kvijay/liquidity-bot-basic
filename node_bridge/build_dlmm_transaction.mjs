import { Connection, PublicKey } from "@solana/web3.js";
import { BN } from "@coral-xyz/anchor";
import { DLMM, StrategyType } from "@meteora-ag/dlmm";
import fs from "node:fs";

const STRATEGY_MAP = new Map([
  ["spot", StrategyType.Spot],
  ["curve", StrategyType.Curve],
  ["bidask", StrategyType.BidAsk],
]);

function readInput() {
  const raw = fs.readFileSync(0, "utf8");
  if (!raw) {
    throw new Error("Missing JSON payload");
  }
  return JSON.parse(raw);
}

function clampStrategyBounds(strategy, lbPair) {
  let { minBinId, maxBinId } = strategy;
  const protocolMin = Number(lbPair.parameters.minBinId);
  const protocolMax = Number(lbPair.parameters.maxBinId);
  if (minBinId < protocolMin) minBinId = protocolMin;
  if (maxBinId > protocolMax) maxBinId = protocolMax;
  if (minBinId > maxBinId) {
    const activeId = Number(lbPair.activeId);
    minBinId = Math.min(activeId, protocolMax);
    maxBinId = Math.max(activeId, protocolMin);
  }
  return { ...strategy, minBinId, maxBinId };
}

function parseStrategy(input, lbPair) {
  const activeId = Number(lbPair.activeId);
  const binSpan = input.binSpan ?? 2;
  const rawType = (input.strategyType ?? "spot").toLowerCase();
  const strategyType = STRATEGY_MAP.get(rawType);
  if (strategyType === undefined) {
    throw new Error(`Unsupported strategy type: ${rawType}`);
  }
  const minBinId =
    typeof input.minBinId === "number"
      ? input.minBinId
      : activeId - Math.max(1, binSpan);
  const maxBinId =
    typeof input.maxBinId === "number"
      ? input.maxBinId
      : activeId + Math.max(1, binSpan);
  return clampStrategyBounds(
    {
      minBinId,
      maxBinId,
      strategyType,
      singleSidedX: Boolean(input.singleSidedX ?? false),
    },
    lbPair,
  );
}

function toInstructionPayload(ix) {
  return {
    programId: ix.programId.toBase58(),
    data: Buffer.from(ix.data).toString("base64"),
    accounts: ix.keys.map((key) => ({
      pubkey: key.pubkey.toBase58(),
      isSigner: key.isSigner,
      isWritable: key.isWritable,
    })),
  };
}

async function main() {
  const input = readInput();
  const connection = new Connection(input.rpcUrl, input.commitment ?? "confirmed");
  const pool = new PublicKey(input.poolAddress);
  const user = new PublicKey(input.userPublicKey);
  const position = new PublicKey(input.positionPublicKey);
  const dlmmList = await DLMM.create(connection, pool, { commitment: input.commitment ?? "confirmed" });
  const dlmm = Array.isArray(dlmmList) ? dlmmList[0] : dlmmList;
  if (!dlmm) {
    throw new Error("Unable to initialise DLMM client for pool");
  }
  const strategy = parseStrategy(input, dlmm.lbPair);
  const totalXAmount = new BN(String(input.totalXAmount ?? "0"));
  const totalYAmount = new BN(String(input.totalYAmount ?? "0"));
  const slippage = typeof input.slippage === "number" ? input.slippage : 50;

  const tx = await dlmm.initializePositionAndAddLiquidityByStrategy({
    positionPubKey: position,
    totalXAmount,
    totalYAmount,
    strategy,
    user,
    slippage,
  });

  const instructions = tx.instructions.map(toInstructionPayload);
  const response = {
    instructions,
    strategy: {
      minBinId: strategy.minBinId,
      maxBinId: strategy.maxBinId,
      strategyType: Number(strategy.strategyType),
      singleSidedX: Boolean(strategy.singleSidedX ?? false),
    },
    metadata: {
      activeId: Number(dlmm.lbPair.activeId),
      binStep: Number(dlmm.lbPair.binStep),
      tokenXMint: dlmm.lbPair.tokenXMint.toBase58(),
      tokenYMint: dlmm.lbPair.tokenYMint.toBase58(),
    },
  };
  process.stdout.write(JSON.stringify(response));
}

main().catch((err) => {
  const message = err instanceof Error ? err.message : String(err);
  const stack = err instanceof Error ? err.stack : undefined;
  process.stderr.write(JSON.stringify({ error: message, stack }) + "\n");
  process.exit(1);
});
