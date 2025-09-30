import { Connection, PublicKey } from "@solana/web3.js";
import { BN } from "@coral-xyz/anchor";
import AmmImpl from "@meteora-ag/dynamic-amm-sdk";
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

  const pool = await AmmImpl.create(connection, poolAddress, {
    commitment,
    programId: input.programId ? new PublicKey(input.programId) : undefined,
  });
  if (!pool) {
    throw new Error("Unable to initialise DAMM client for pool");
  }
  await pool.updateState();

  const poolTokenAmount = new BN(String(input.poolTokenAmount ?? "0"));
  if (poolTokenAmount.isZero()) {
    throw new Error("poolTokenAmount must be greater than zero");
  }

  const slippageBps = typeof input.slippageBps === "number" ? input.slippageBps : 100;
  const quote = pool.getWithdrawQuote(poolTokenAmount, slippageBps);
  const transaction = await pool.withdraw(
    userPublicKey,
    quote.poolTokenAmountIn,
    quote.tokenAOutAmount,
    quote.tokenBOutAmount,
  );

  const instructions = transaction.instructions.map(toInstructionPayload);
  const response = {
    instructions,
    quote: {
      poolTokenAmountIn: quote.poolTokenAmountIn.toString(),
      tokenAOutAmount: quote.tokenAOutAmount.toString(),
      tokenBOutAmount: quote.tokenBOutAmount.toString(),
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

