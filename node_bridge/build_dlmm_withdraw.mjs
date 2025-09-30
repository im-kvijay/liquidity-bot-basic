import { Connection, PublicKey } from "@solana/web3.js";
import { BN } from "@coral-xyz/anchor";
import { DLMM } from "@meteora-ag/dlmm";
import fs from "node:fs";

function readInput() {
  const raw = fs.readFileSync(0, "utf8");
  if (!raw) {
    throw new Error("Missing JSON payload");
  }
  return JSON.parse(raw);
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
  const commitment = input.commitment ?? "confirmed";
  const connection = new Connection(input.rpcUrl, commitment);
  const poolAddress = new PublicKey(input.poolAddress);
  const userPublicKey = new PublicKey(input.userPublicKey);
  const positionPublicKey = new PublicKey(input.positionPublicKey);

  const dlmmList = await DLMM.create(connection, poolAddress, { commitment });
  const dlmm = Array.isArray(dlmmList) ? dlmmList[0] : dlmmList;
  if (!dlmm) {
    throw new Error("Unable to initialise DLMM client for pool");
  }

  const position = await dlmm.getPosition(positionPublicKey);
  if (!position) {
    throw new Error("Position not found for supplied public key");
  }
  const binIds = position.positionBinData.map((bin) => Number(bin.binId));
  if (binIds.length === 0) {
    throw new Error("Position contains no liquidity to withdraw");
  }
  const fromBinId = Math.min(...binIds);
  const toBinId = Math.max(...binIds);
  const bpsValue = typeof input.bps === "number" ? input.bps : 10000;
  const bps = new BN(String(bpsValue));
  const shouldClaimAndClose = Boolean(input.shouldClaimAndClose ?? true);
  const skipUnwrapSOL = Boolean(input.skipUnwrapSOL ?? false);

  const txs = await dlmm.removeLiquidity({
    user: userPublicKey,
    position: positionPublicKey,
    fromBinId,
    toBinId,
    bps,
    shouldClaimAndClose,
    skipUnwrapSOL,
  });

  const instructions = [];
  const txArray = Array.isArray(txs) ? txs : [txs];
  for (const tx of txArray) {
    for (const ix of tx.instructions) {
      instructions.push(toInstructionPayload(ix));
    }
  }

  const response = {
    instructions,
    metadata: {
      fromBinId,
      toBinId,
      bins: binIds,
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

