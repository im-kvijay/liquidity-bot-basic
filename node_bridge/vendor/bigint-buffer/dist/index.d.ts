/// <reference types="node" />
export declare function toBigIntLE(buf: Buffer | ArrayBuffer | ArrayBufferView | ArrayLike<number> | string): bigint;
export declare function toBigIntBE(buf: Buffer | ArrayBuffer | ArrayBufferView | ArrayLike<number> | string): bigint;
export declare function toBufferLE(num: bigint | number | string, width: number): Buffer;
export declare function toBufferBE(num: bigint | number | string, width: number): Buffer;
