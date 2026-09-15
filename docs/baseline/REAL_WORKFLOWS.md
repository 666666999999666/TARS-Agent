# 真实 CLI 与 Core 工作流验收

日常任务可从 `tars chat`、`tars run --goal` 或 `tars-tui` 提交。客户端验收与底层 Core 验收分别记录，不能用 RPC 成功冒充 CLI 输入、退出码或 TUI 界面已经通过。

## 真实 CLI 验收脚本

`scripts/acceptance_cli.py` 在 Windows ConPTY 中启动实际 `tars chat` 和 `tars run --goal`，输入任务、审批字符、Ctrl+C 和 EOF；使用真实 Core、Provider 和隔离工作区，保存终端输入输出、进程与退出码。它复用已有验收工具准备独立 Core，不向用户原 `state.db` 写入测试会话。

```powershell
# 只打印验收计划，不调用模型。
uv run --no-sync python scripts/acceptance_cli.py

# 只检查本地终端输入与退出，不调用模型。
uv run --no-sync python scripts/acceptance_cli.py --pty-self-test

# 已取得本次授权后，实际运行 CLI 验收。
uv run --no-sync python scripts/acceptance_cli.py --execute --output-root build/v1-cli/real-cli
```

场景包括 goal 成功、拒绝、工具失败和取消的退出码，以及 chat 连续多轮、取消后继续、EOF 退出和会话恢复。结果应同时核对实际 CLI 进程与 SQLite/文件；单独的 ConPTY 自检只证明终端工具可用。非 TTY 审批、后台子 Run 不阻塞 goal 退出等合同，还需对应的客户端回归或专项实测，不能从未包含的场景推断通过。

两个脚本各自的参数以 `--help` 为准。以下章节说明 `scripts/acceptance_workflows.py`：它直接通过 SocketClient 调用 CoreApp、RuntimeService 和 Provider，使用真实模型和工具，并自动处理限定范围审批；它没有启动正式 CLI 或操作 TUI。

## 执行条件与状态隔离

运行前确认本次真实模型消耗、工作目录和资源操作范围，并检查 Docker 引擎与镜像可用。

脚本目前固定使用本机已准备的 `deepseek-flash` 和 `https://api.deepseek.com/anthropic` 配置。启动脚本时，`TARS_HOME` 仍须指向原来的 `~/.tars-baseline`；如设置 `TARS_CONFIG`，它只能指向该目录的 `config.toml`。原请求账本必须已经存在。此脚本只验收本地工具，要求 `sandbox.mode=required`，并拒绝包含 MCP 服务的配置；MCP 另行验收。

脚本子进程先读取原受信配置，并保留原 `llm.request_budget_path`，再把 Core 的运行状态切换到独立目录：

- 默认证据目录为 `build/v1-simplify/real-workflows/<UTC时间戳-随机后缀>`，可用 `--output-root` 改变根目录。
- 默认状态目录为本轮证据目录下的 `state`，也可用 `--state-home` 指定一个尚不存在的新目录。它不能与原模型配置目录重叠。
- 会话数据库、artifacts、控制文件、日志和 trace 都写入这个状态目录；不读取或修改原 `state.db`，不接管或停止原 daemon。
- 模型配置与请求账本继续使用原路径。不会复制密钥、创建替代账本或重置累计请求数。

## 命令与请求限制

在已经安装项目依赖的正式仓库中运行：

```powershell
# 只打印计划，不读取真实配置、不创建目录、不发送请求。
uv run --no-sync python scripts/acceptance_workflows.py

# 取得本次授权后执行三类工作流，每类连续三轮。
uv run --no-sync python scripts/acceptance_workflows.py --execute --workflow all --output-root build/v1-simplify/real-workflows

# 只复测会话类，仍连续三轮。
uv run --no-sync python scripts/acceptance_workflows.py --execute --workflow session
```

`--request-limit` 接受正整数或 `unlimited`，默认值为 75。这个数字是原账本的累计上限，不是本次额外获得的次数。实际限制取脚本参数和受信生产配置中更严格的有限值；生产默认值为 100。只有本次明确授权不限次时，才使用 `--request-limit unlimited`；受信生产配置仍有限时，脚本依然受它约束。两者都不限次时仍继续计数，不退款、不清空旧记录。

验收发送限制沿用同一 RequestLedger 数据库；重试和压缩同样消耗请求。报告区分脚本限制、受信生产限制和实际限制。其他程序可以继续使用原 daemon，但若同时消耗同一账本，账户累计差值不能全部算成本次工作流的用量。

默认结束时关闭本次会话和 daemon，并保留证据。只有本次明确要求保留服务供 TUI 接续时才使用 `--retain`：脚本会重启同一隔离状态目录中的 Core，取消额外脚本限制，保留受信生产限制和原账本。接续时使用 manifest 记录的端口、状态目录和会话 ID；客户端应使用只含 Core 地址的独立配置。停止管理也必须针对该隔离状态目录，不能使用原 HOME 的控制文件。

## 三类任务及失败处理

- **文件**：列目录、读取随机标记、写报告、读回核对、保存会话笔记；同时检查实际文件和持久工具记录。
- **权限**：拒绝指定文件写入，确认没有该步副作用且模型说明拒绝；再通过新的逐次授权完成替代文件。只批准本轮约定的文件及精确、限时的取消探针命令，拒绝未知工具和宿主回退。
- **会话**：保存随机代号、客户端断开后恢复、真实手动压缩、重启本次 Core 后恢复、取消已经开始的容器命令、检查延迟副作用没有发生，再继续合法任务。取消消息不能进入成功上下文。

每类连续执行三轮。同一类别失败后停止该类别剩余轮次并保留证据，其他独立类别可继续；预算耗尽则停止新的模型操作并标明未完成项。修复后重跑受影响流程，不能把旧的候选结果当作修复后通过。

## 如何判断通过

每个 Run 都核对 RPC 与 SQLite 终态。状态为 `succeeded`、`failed`、`cancelled` 或 `interrupted` 时，归属该 Run 的工具记录不能仍是 `queued` 或 `running`；结果 JSON 中的 `terminal_tool_check` 记录检查结果及未结束工具 ID，发现残留立即使该次验收失败。这个检查不代替实际工具停止验证，还要核对容器清理和延迟文件副作用。

证据包括 manifest、每个 Run 的 RPC/数据库/指标快照、事件、审批、压缩、日志及中文工作区。manifest 标明实际运行的源码 HEAD、dirty 状态、源码指纹、脚本哈希、所属 PID/端口和前后累计计数。原始证据保留在本轮本地目录，不提交 Git，也不输出密钥或控制 token。

任务等待上限为 180 秒；取消若返回尚未确认，不能计作已停止。结束后只检查并回收本次拥有的进程和容器。`all_nine_complete=true` 只表示这九轮 RPC 工作流通过，不代表正式 CLI、真实 TUI、外部 MCP、安装产物或全部运行环境已通过。

当前结果及仍未验证的范围统一见[验证摘要](VERIFICATION_SUMMARY.md)，本文件不重复记录阶段数字。
