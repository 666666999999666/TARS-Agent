# 模型配置说明

日常配置步骤集中在[运行手册](../../RUNBOOK.md#模型与安全配置)。配置位于真实进程环境指定的 `TARS_HOME/config.toml`，默认是 `~/.tars-baseline/config.toml`。项目文件不能自行授予端点、Docker 或 MCP 权限。

自定义 Anthropic 兼容端点必须使用专用密钥；不要把密钥写进代码、Git、截图或验收报告。官方服务密钥不会自动发往自定义端点。

历史 V1 验收请求的模型名为 `deepseek-flash`，端点为 `https://api.deepseek.com/anthropic`。这是当时的请求配置，不能据此推定未返回的底层型号或当前价格。历史结果见[验证摘要](VERIFICATION_SUMMARY.md)，当前工程结果与冻结实验归属见[工程验证](../evaluation/VALIDATION.md)，AppWorld 专用配置见[复现说明](../evaluation/REPRODUCE.md)。本页作为配置导航保留；通用配置示例只在运行手册维护。

请求限额默认 100，按同一账本累计，重启不重置。调整为正整数或 `unlimited` 只能在受信配置或真实进程环境中进行，由项目所有者按当前使用需求决定。

本次验收将临时会话、产物和日志放在独立测试目录，同时继续使用原请求账本；不会为了绕过额度而新建计数起点。
