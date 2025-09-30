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
  const connection = new Connection(input.rpcUrl, input.commitment ?? "confirmed");
  const poolAddress = new PublicKey(input.poolAddress);
  const userPublicKey = new PublicKey(input.userPublicKey);

  const pool = await AmmImpl.create(connection, poolAddress, {
    commitment: input.commitment ?? "confirmed",
    programId: input.programId ? new PublicKey(input.programId) : undefined,
  });
  if (!pool) {
    throw new Error("Unable to initialise DAMM client for pool");
  }
  await pool.updateState();

  const tokenAAmount = new BN(String(input.tokenAAmount ?? "0"));
  const tokenBAmount = new BN(String(input.tokenBAmount ?? "0"));
  if (tokenAAmount.isZero() && tokenBAmount.isZero()) {
    throw new Error("Contribution amounts must be greater than zero");
  }

  const slippageBps = typeof input.slippageBps === "number" ? input.slippageBps : 100;
  const quote = pool.getDepositQuote(tokenAAmount, tokenBAmount, true, slippageBps);
  const transaction = await pool.deposit(
    userPublicKey,
    quote.tokenAInAmount,
    quote.tokenBInAmount,
    quote.minPoolTokenAmountOut,
  );

  const instructions = transaction.instructions.map(toInstructionPayload);
  const response = {
    instructions,
    quote: {
      poolTokenAmountOut: quote.poolTokenAmountOut.toString(),
      minPoolTokenAmountOut: quote.minPoolTokenAmountOut.toString(),
      tokenAInAmount: quote.tokenAInAmount.toString(),
      tokenBInAmount: quote.tokenBInAmount.toString(),
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
