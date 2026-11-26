# 可选：Tushare 原始数据归档

`quant_research_lab.tushare_source` 可以把 Tushare 接口的原始返回（日线、复权因子、涨跌停、停复牌、交易日历）归档到本地，并记录请求参数和文件哈希，之后离线生成数据包。它不是默认流程，也不是运行本项目的必要条件；推荐的数据来源见 README 的"数据来源"一节。

## 运行

```bash
export TUSHARE_TOKEN=...              # 不设置时，脚本在终端隐藏输入
export TUSHARE_BASE_URL=...           # Tushare 接口地址，或你自己的转发地址
bash scripts/archive_tushare_sdk.sh \
  --core-only --symbols 000001.SZ --start 20200102 --end 20241231 \
  --cache output/tushare_cache --output output/my_archive
```

- 支持两种传输方式：`--transport http`（自带的 HTTP JSON 客户端，默认 `https://api.tushare.pro`）和 `--transport sdk`（官方 tushare 包，需 `python -m pip install -e '.[tushare]'`）。
- token 只从环境变量或隐藏输入读取，不写入代码、`.env`、缓存或日志；异常信息会抑制服务端原文，避免回显凭据。
- 使用 HTTP（非 HTTPS）地址时必须显式加 `--allow-http`，因为 token 会明文发送。
- 请求窗口、分页和日期都会校验；返回重复页或窗口外数据即停止。中断后用相同缓存目录重跑，已完成的请求不会重复。

## 生成数据包

```bash
.venv/bin/python -m quant_research_lab.data_bundle \
  --archive output/my_archive --output output/my_bundle
quant-lab research --bundle output/my_bundle --output output/my_research
```

## 核对记录

平安银行 2020-01-02 至 2024-12-31 的归档结果：日线、复权因子、涨跌停各 1,212 行，原始未复权 OHLC 与 BaoStock 逐日比对全部一致。双源一致不代表它们的上游相互独立，也不能证明历史发布版本完整；manifest 中股票池与公司行动完整性仍标记为 false。
