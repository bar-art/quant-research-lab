# 跨项目数据包（可选）

`scripts/run_open_data.sh` 是给"三个项目并列放在同一目录"这种用法准备的便捷脚本：它读取 ashare-data-pipeline 的本地 BaoStock 缓存（目录名为 `ashare-data-pipeline` 或旧名 `a-share-quant-agent` 均可），生成带文件哈希的 `research_bundle_v1` 数据包，再分别交给本项目和 alpha-evidence-lab 运行。单独使用本项目时不需要它，直接用 CSV + manifest 即可。

```bash
# 三个项目并列放置，并分别安装好各自的 .venv
bash scripts/run_open_data.sh                       # 只读本地缓存
bash scripts/run_open_data.sh --online \
  --start 2020-01-02 --end 2024-12-31 \
  --codes 600000.SH 000001.SZ --output output/my_bundle_run   # 允许联网补齐缺口
bash scripts/run_open_data.sh --export-only          # 只导出数据包
```

- 默认只使用本地缓存：运行前检查请求区间、交易日历、证券资料和缓存文件完整性，缺失就停止，不会暗中联网或改用模拟数据。`--online` 允许补齐缺口，不会强制刷新已有缓存。
- 默认样本为 2020-01-02 至 2024-12-31 的 6 只股票（000001.SZ、300750.SZ、600000.SH、600518.SH、600519.SH、688981.SH），用于检查研究设施，不是无幸存者偏差的股票池。停牌行保留；新上市证券保留实际起始日期。
- alpha 默认计划要求每条路径至少 630 行，不足时明确停止；不能为通过门槛而删掉短历史或已退市证券。
- 两个项目分别校验数据包的文件哈希；它们的研究问题和使用窗口不同，两份收益不能当成策略之间的比赛。

## 验收记录

6 只股票、7,144 条日线（34 条停牌、5 条涨跌停未知，未知且未豁免时拒绝成交）：

- quant：57 个滚动测试段、171 次候选评估；
- alpha：378 次尝试、42 批冻结，审计与文件哈希核验通过；random 和 memory 相对 ordered 均未通过研究门槛。跑通软件不代表策略有效。

## Yahoo Finance

Yahoo 数据缺少停牌、历史 ST 和涨跌停信息，不能直接用于本项目的 A 股撮合；ashare-data-pipeline 只把它用作价格核对来源。yfinance 是调用 Yahoo 公共接口的开源工具，不是官方客户端，使用前应按用途核对其条款。

## 研究边界

示例股票池是静态的；原始发布版本、公司行动现金流、交易单位和最低佣金仍不完整。manifest 的 `historical_universe_complete` 与 `corporate_actions_complete` 均为 false。
