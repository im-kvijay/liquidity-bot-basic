'use strict';

Object.defineProperty(exports, "__esModule", { value: true });

const ZERO = 0n;

function normalizeBuffer(buf) {
  if (buf == null) {
    throw new TypeError('Expected a buffer-like value');
  }
  if (Buffer.isBuffer(buf)) {
    return Buffer.from(buf);
  }
  if (buf instanceof Uint8Array) {
    return Buffer.from(buf);
  }
  if (ArrayBuffer.isView(buf)) {
    return Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  if (buf instanceof ArrayBuffer) {
    return Buffer.from(buf);
  }
  if (Array.isArray(buf)) {
    return Buffer.from(buf);
  }
  if (typeof buf === 'string') {
    const normalized = buf.trim();
    if (normalized.length === 0) {
      return Buffer.alloc(0);
    }
    const hex = normalized.startsWith('0x') || normalized.startsWith('0X')
      ? normalized.slice(2)
      : normalized;
    if (hex.length % 2 !== 0) {
      throw new RangeError('Hex string must have an even number of characters');
    }
    return Buffer.from(hex, 'hex');
  }
  throw new TypeError('Unsupported buffer input type');
}

function coerceBigInt(value) {
  if (typeof value === 'bigint') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || Math.trunc(value) !== value) {
      throw new RangeError('Expected a finite integer');
    }
    return BigInt(value);
  }
  if (typeof value === 'string') {
    if (value.trim().length === 0) {
      return ZERO;
    }
    return BigInt(value);
  }
  if (value != null && typeof value.valueOf === 'function') {
    const raw = value.valueOf();
    if (typeof raw === 'bigint' || typeof raw === 'number' || typeof raw === 'string') {
      return coerceBigInt(raw);
    }
  }
  throw new TypeError('Expected bigint-compatible value');
}

function assertWidth(width) {
  if (!Number.isInteger(width) || width < 0) {
    throw new RangeError('Width must be a non-negative integer');
  }
}

function ensureFits(value, width) {
  if (width === 0) {
    if (value === ZERO) {
      return;
    }
    throw new RangeError('Value does not fit in zero-length buffer');
  }
  const maxBits = BigInt(width) * 8n;
  if (value < ZERO) {
    throw new RangeError('Negative values are not supported');
  }
  if (value >> maxBits !== ZERO) {
    throw new RangeError('Value exceeds width');
  }
}

function toBigIntLE(buf) {
  const bytes = normalizeBuffer(buf);
  if (bytes.length === 0) {
    return ZERO;
  }
  const reversed = Buffer.from(bytes);
  reversed.reverse();
  const hex = reversed.toString('hex');
  return hex.length === 0 ? ZERO : BigInt(`0x${hex}`);
}
exports.toBigIntLE = toBigIntLE;

function toBigIntBE(buf) {
  const bytes = normalizeBuffer(buf);
  if (bytes.length === 0) {
    return ZERO;
  }
  const hex = bytes.toString('hex');
  return hex.length === 0 ? ZERO : BigInt(`0x${hex}`);
}
exports.toBigIntBE = toBigIntBE;

function toBufferLE(num, width) {
  assertWidth(width);
  const value = coerceBigInt(num);
  ensureFits(value, width);
  if (width === 0) {
    return Buffer.alloc(0);
  }
  const hex = value.toString(16).padStart(width * 2, '0');
  const buffer = Buffer.from(hex, 'hex');
  buffer.reverse();
  return buffer;
}
exports.toBufferLE = toBufferLE;

function toBufferBE(num, width) {
  assertWidth(width);
  const value = coerceBigInt(num);
  ensureFits(value, width);
  if (width === 0) {
    return Buffer.alloc(0);
  }
  const hex = value.toString(16).padStart(width * 2, '0');
  return Buffer.from(hex, 'hex');
}
exports.toBufferBE = toBufferBE;
