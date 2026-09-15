# TARS-Agent

TARS-Agent 是由 liaoqizai 个人自主开发和维护的 Python 3.12 本地 Agent 项目：在终端提交任务，由模型决定是否调用工具，工具完成后把真实结果交回模型。V1 支持连续对话、会话历史、审批、Docker 工具隔离、子 Agent、MCP 和只读 Web 面板。

**CLI 和 TUI 都是正式任务入口。** `tars chat` 用于连续对话，`tars run --goal ...` 完成一次任务后退出客户端，`tars-tui` 提供终端界面。三种方式共用后台 Core；关闭客户端不会关闭 Core。

版本可用 `uv run --no-sync tars --version` 查看。验证结果与环境边界见[V1 基准记录](docs/baseline/VERIFICATION_SUMMARY.md)。模型调用按受信配置计量。

## 每天怎么用

完成首次准备后，在项目目录运行：

```powershell
uv run --no-sync tars core start
uv run --no-sync tars chat
```

看到 `> ` 后输入任务，按 Enter 发送；一轮结束后可以继续追问。需要审批时，检查工具和完整参数，按当前提示输入 `y` 或 `n`。运行中按 Ctrl+C 请求取消，处理结束后仍可继续聊天；空闲时 Ctrl+C 退出，EOF 也会断开客户端并保留会话。

只执行一个目标时使用：

```powershell
uv run --no-sync tars run --goal "列出当前工作区的文件，并说明有哪些文件"
```

一次性命令在主 Run 结束后退出，不等待仍在运行的后台子 Agent，会列出这些子 Run 的 ID。它使用严格退出码：正常为 `0`，参数错误为 `2`，取消为 `130`；模型、连接或任何已发生的工具失败为 `1`，即使模型后来补救成功。管道或重定向等非交互场景不会自动批准工具；需要人工审批时拒绝并报错。完整规则见[运行手册](RUNBOOK.md#cli-退出码与非交互使用)。

喜欢终端界面时，运行 `uv run --no-sync tars-tui`。在输入框写任务，Enter 发送，Alt+Enter 换行；TUI 的快捷键保持如下：

- **Ctrl+Q**：退出界面，后台任务继续。
- **Ctrl+X**：取消当前任务并等待资源清理；取消超时表示还没有确认结束。
- 重新打开旧会话：先用 `uv run --no-sync tars sessions list` 查 ID，再用 `uv run --no-sync tars chat --resume <session_id>` 或 `uv run --no-sync tars-tui --resume <session_id>`。
- 结束后台服务：`uv run --no-sync tars core stop`。这会请求停止仍在运行的任务。

新会话使用启动 CLI/TUI 时所在的目录作为工作区；恢复会话继续使用原来的目录。第一次真实工具演示请使用专用临时目录，按[运行手册中的例子](RUNBOOK.md#做一个真实工具任务)操作，不直接拿重要文件试写。

## 首次准备

需要 Python 3.12、uv，以及启用 Linux 引擎的 Docker Desktop。在当前项目目录安装依赖：

```powershell
uv sync --locked --no-dev
```

按[模型与安全配置](RUNBOOK.md#模型与安全配置)在本机设置模型、端点和密钥，再准备工具镜像：

```powershell
uv run --no-sync tars sandbox build
uv run --no-sync tars sandbox doctor
```

默认 `sandbox.mode = "required"`，Docker 或镜像不可用时启动失败。当前镜像名为 `tars-agent-sandbox:0.8.0`；首次安装、工具 worker 改动后需要构建，日常使用无需重建。

CLI/TUI 不需要 Node/npm、Chromium 或 Web 面板。默认数据目录为 `~/.tars-baseline`：SQLite 保存会话、运行和事件，`artifacts/sessions` 保存笔记及逐任务诊断文件。旧文件会话和旧仓库数据不会自动导入，已有数据库和笔记不因本次源码精简而清空。

## 保留的其他能力

- 子 Agent 可以承担独立子任务，前台等待结果或后台运行。内建 `reviewer` 只提供读取文件和列目录工具；它不能写文件、执行 shell 或再派生任务。
- MCP 连接已在本机受信配置中登记的外部工具服务，仍需遵守工具审批。外部服务的操作不属于本地 Docker 隔离范围。
- `/compact` 可以压缩当前会话上下文；长任务也可触发自动压缩，这些操作会调用模型并计入请求额度。
- Web 只读面板：Core 启动后在另一个终端运行 `uv run --no-sync tars-web --open`，查看会话、运行和事件。关闭这个终端中的 Web 服务不会代替停止 Core。

管理命令保持如下形式：

```powershell
uv run --no-sync tars core status
uv run --no-sync tars run status <run_id>
uv run --no-sync tars run metrics <run_id>
uv run --no-sync tars run cancel <run_id>
```

模型请求默认按同一个 `TARS_HOME` 下的账本累计，限额为 100 次，重启不会清零。子 Agent、重试和压缩共同计数。调整额度需自己确认成本并修改受信配置，不能把历史验收里的 `unlimited` 当成本次授权。

## 从哪里学习

先读[架构与阅读顺序](docs/baseline/ARCHITECTURE.md)，沿着一条任务学习：CLI/TUI 收输入 → RuntimeService 保存请求并启动任务 → AgentRunner 准备执行 → AgentLoop 请求模型和调用工具 → RuntimeService 保存结果。CLI 的两种使用方式共用 `cli/client.py`，没有新增第二个 AgentLoop。

- [运行与故障排查](RUNBOOK.md)：安装、配置、真实任务与常见问题。
- [当前验证摘要](docs/baseline/VERIFICATION_SUMMARY.md)和[已知限制](docs/baseline/LIMITATIONS.md)：哪些已验证，哪些仍受阻。
- [Wire Protocol V2](WIRE_PROTOCOL.md)：需要理解 TUI 与 Core 通信时再看。

开发时使用 `uv sync --locked --group qa --group security`。仅修改或构建 Web 前端时才需要在 `web` 下安装 npm 依赖；浏览器验收另准备 Chromium。测试命令与每次实际结果记录在验证摘要中。

## 作者与许可

作者及维护者：**liaoqizai**。Copyright © 2026 liaoqizai，保留所有权利。项目许可见 [LICENSE](LICENSE)；第三方依赖遵循各自的许可证。
