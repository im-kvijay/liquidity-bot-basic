# References

Verified resources used while implementing the DAMM v2/DLMM, compliance, and observability stack.

## Protocols and SDKs

* Meteora Dynamic AMM SDK – https://github.com/MeteoraAG/dynamic-amm-sdk
* Meteora DLMM SDK – https://github.com/MeteoraAG/dlmm-sdk
* Meteora program IDs & documentation – https://docs.meteora.ag/
* Solana Web3.js – https://solana-labs.github.io/solana-web3.js/
* Solana Python SDK (`solana` package) – https://github.com/michaelhly/solana-py

## Market data & compliance

* Solana Token Registry – https://github.com/solana-labs/token-list
* Jupiter price oracle – https://station.jup.ag/docs/apis/price-api
* OFAC sanctions list (CSV) – https://www.treasury.gov/ofac/downloads/sdn.csv
* Chainalysis sanctions oracle – https://go.chainalysis.com/chainalysis-sanctions-oracle.html

## Operations & observability

* FastAPI – https://fastapi.tiangolo.com/
* Prometheus exposition format – https://prometheus.io/docs/instrumenting/exposition_formats/
* Uvicorn server configuration – https://www.uvicorn.org/
* SQLite best practices – https://www.sqlite.org/whentouse.html

> Always cross-check third-party documentation for updates before shipping to production; protocol upgrades can invalidate API
> contracts and program IDs.
