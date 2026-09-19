# 运行与故障排查

## 每天怎么用

完成首次安装和配置后，在项目目录启动 Core，再选择一种操作方式：

```powershell
uv run --no-sync tars core start
uv run --no-sync tars chat
```

`chat` 中看到 `> ` 后输入任务，Enter 发送，一轮结束后继续追问。只做一个目标时用 `uv run --no-sync tars run --goal "你的任务"`；喜欢终端界面则用 `uv run --no-sync tars-tui`。Core 已经运行时只需打开客户端。

CLI 聊天请将一次任务整理成一行再提交。多行粘贴会被拆成多次输入，执行中的后续行不会自动排队；Windows ConPTY 的多行粘贴显示限制见[已知限制](docs/baseline/LIMITATIONS.md)。

聊天运行中按 Ctrl+C 请求取消，处理结束后留在当前会话继续输入；空闲时 Ctrl+C 退出客户端。EOF 也会退出聊天并保留会话，Windows 终端通常用 Ctrl+Z 再按 Enter。CLI 退出不调用 `session.close` 或 `core.shutdown`，不会顺手关闭 daemon。

## CLI 退出码与非交互使用

一次性命令 `tars run --goal "任务"` 在主 Run 结束后返回。若还有后台子 Run，它会列出 ID 以及查询、取消命令，不等待这些后台任务全部完成。

| 退出码 | 含义 |
| --- | --- |
| `0` | goal 正常完成且没有工具失败；或 chat 收到 EOF 后正常退出。 |
| `1` | 客户端配置或连接失败；goal 的模型、工具失败，或无法获得所需审批。goal 中任一工具失败，即使后续补救成功也返回 `1`。chat 的单轮失败会显示错误，仍可继续输入。 |
| `2` | 参数错误，例如缺少目标、空目标或把 `--goal` 与查询子命令混用。 |
| `130` | goal 被取消，或 chat 空闲时按 Ctrl+C 退出。chat 运行中取消一轮后仍可继续输入。 |

PowerShell 可在命令结束后用 `$LASTEXITCODE` 查看退出码。CLI 的严格结果与数据库 Run 状态不是同一个概念：模型补救后 Run 可以是 `succeeded`，但该 goal 出现过工具失败时，CLI 仍返回 `1`。后台子 Run 在客户端退出以后才发生的结果，应按所列 ID 单独查询。

chat 是连续会话，单轮失败后仍可继续输入。EOF 返回 `0` 只表示客户端正常退出，不表示之前每一轮都成功；逐轮结果应查看终端提示或 `tars run status`。

在管道、重定向或其他没有交互终端的场景中，客户端不会把预先输入的 `y` 当作授权。遇到需要人工审批的操作会拒绝并请求取消；goal 返回 `1`。需要批准的真实工具任务应在交互终端运行。取消请求返回超时不等于资源已经停止，仍需用 `tars run status <run_id>` 核对。

## 模型与安全配置

环境要求和首次安装见 [README](README.md#首次准备)。已有可用配置时继续沿用，先核对实际模型、端点和工作目录，不用下面的示例覆盖真实配置。

默认数据目录为 `~/.tars-baseline`；如果设置了真实进程环境变量 `TARS_HOME`，则使用指定目录。主要配置文件是该目录下的 `config.toml`。在本机编辑器中填写模型设置，以下字段是占位符：

```toml
[llm]
default_model = "YOUR_MODEL"
base_url = "https://YOUR_PROVIDER"
api_key = "YOUR_DEDICATED_KEY"
request_limit = 100
context_budget_tokens = 32768
context_safety_margin = 1024
max_tokens = 8192

[sandbox]
mode = "required"
image = "tars-agent-sandbox:0.8.0"
```

使用 Anthropic 官方端点时省略 `base_url`，也可从真实进程环境提供 `ANTHROPIC_API_KEY`；自定义兼容端点必须配专用密钥，官方 key 不会自动发往自定义端点。不要把密钥贴到聊天、截图或 Git 中。

项目 `.env` 和 `.tars/config.toml` 只能设置受限的普通选项，不能改变 MCP、Docker、端点、日志输出位置或提高请求限额。支持的环境变量见 [.env.example](.env.example)。`llm.router` 和 `sandbox.enabled` 已移除，旧配置若仍包含它们需要删掉；模型使用明确的 `default_model`，沙箱策略使用 `mode`。

**限额 100 指同一个 HOME 账本中的累计请求，不是每个任务 100 次。** 请求发送前预占次数，模型重试、会话压缩和子 Agent 都使用 `acceptance/request-budget.sqlite3`。额度耗尽会在下次发送前停止。提高限额需要明确考虑成本；受信配置可设正整数或 `"unlimited"`，后者仍持续计数。不要删除账本或更换 HOME 绕过限额。

## 上下文预算与旧配置

`llm.context_budget_tokens` 是本地请求预算，默认 32768；它不代表模型端点的真实窗口已经确认。输入可用预算还要减去 `llm.max_tokens` 的输出预留和 `llm.context_safety_margin`（默认 1024）的额外余量。使用真实端点前仍需核对其窗口；代码不再给未知模型自动假定 200k。`llm.usage.context_pct` 表示实际 usage 相对于这个本地预算的比例。

请求前按序列化 UTF-8 JSON 字节数加消息开销保守估算，覆盖 system、工具说明、历史和当前输入；这不是精确的 tokenizer 计数或费用账单。较早的历史可以不进入当前请求，最近需要的完整交互、当前输入和有效约束必须保留。工具调用与结果按具体 ID 配对，不能为了塞进预算拆开。原记录仍可在数据库历史中查看。

出现 `context_budget_exceeded` 表示必需内容在预留输出和余量后放不下；本次或下一步模型请求没有发送。运行中已完成的工具操作不会因此回滚，工具结果仍保留在审计记录中。失败运行的消息不进入下一轮正式历史。先缩短任务或核对实际窗口及预算配置，不要把“增大预算”当作模型确实支持更大窗口。

自动压缩只处理已完整读入的活动历史；仅载入近期部分历史时不会把未读记录标为已压缩。手动 `/compact` 若放不下完整输入会明确失败，原活动记录保持不变。正常压缩、摘要落库和恢复继续沿用事务规则，摘要之后仍保留当前输入。摘要有信息损失的可能，不保证无限期记住所有旧内容。

旧的 `compaction.tool_result_limit`、`compaction.tool_result_keep` 及对应环境变量仍可解析，但会告警说明已弃用、未生效；来源记录将它们放入 `ignored_config`。它们不是工具输出保护。内建文件／命令工具已有的输出限制（例如 `sandbox.output_limit_bytes`）继续生效；本轮没有增加统一工具结果截断或外部结果存储系统。

## 启动、审批与恢复

启动失败先运行 `uv run --no-sync tars sandbox doctor`。默认 required 模式要求 Docker Linux 引擎和镜像可用；首次安装或 worker 改动后运行 `uv run --no-sync tars sandbox build`。Core 启动窗口为 45 秒；超时按命令给出的日志位置排查。

CLI 审批会打印工具名和完整 JSON 参数。通常 `y` 为本次允许，`n` 为拒绝；其他选项以当前提示为准。项目参数变更后必须按新请求决定，非交互输入不能绕过审批。

进入 TUI 后输入任务，Enter 发送，Alt+Enter 换行。斜杠补全菜单出现时，第一次 Enter 选择命令，再按一次 Enter 才执行；例如 `/compact`。审批出现时检查工具名、参数和工作目录。“本次允许”只授权这一次调用；“本会话允许相同参数”不会让变更后的命令自动获得授权。Docker 不可用时，preferred 模式的宿主回退还需要单独审批；默认 required 模式不会回退。

Ctrl+X 取消当前任务。超过 15 秒可能返回 `RUN_CANCEL_TIMEOUT=-32033`，表示取消已请求，但资源清理仍未确认。用 `tars run status <run_id>` 查询后续状态。MCP 外部服务可能已经接受操作，取消本地等待不能保证远端操作撤回。

TUI 的 Ctrl+Q 只退出界面，Core 继续执行。恢复时可以选择 CLI 或 TUI：

```powershell
uv run --no-sync tars sessions list
uv run --no-sync tars chat --resume <session_id>
uv run --no-sync tars-tui --resume <session_id>
```

上面最后两条是两种恢复方式，任选一种即可。

恢复会话会读取原来的历史和工作目录，不自动重发最后一条输入。Core 自身崩溃或被停止后，未完成任务会在恢复检查中标为 interrupted，不自动重放工具。成功消息进入后续正式上下文；失败、取消和不完整模型输出保留为诊断记录。

不再需要后台任务时运行 `uv run --no-sync tars core stop`。不要通过修改数据库制造“完成”状态。

## 做一个真实工具任务

先在项目目录启动 Core，再创建独立演示目录。示例只在新目录生成文件，客户端退出后目录保留，便于自己检查结果。

```powershell
$tarsProject = (Get-Location).Path
$tarsDemo = Join-Path ([System.IO.Path]::GetTempPath()) ("tars-v1-demo-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tarsDemo | Out-Null
Set-Content -LiteralPath (Join-Path $tarsDemo "source.txt") -Value "编号：V1-7391" -Encoding utf8
Push-Location $tarsDemo
& (Join-Path $tarsProject ".venv\Scripts\tars.exe") chat
Pop-Location
```

在 CLI 提示符后输入：

> 读取 source.txt，把其中的编号写入 report.md；再读取 report.md 确认内容，最后告诉我实际读到的编号。

预期会看到读取、写入审批、再次读取和最终答案。检查实际 `report.md`，其中应包含 `V1-7391`；不能只凭模型说“已完成”判断工具真的执行。随后在同一会话询问“刚才生成了哪个文件”，验证连续交互。首次调用前确认本机模型配置和可接受的费用，这个例子会产生真实模型请求。

要演示一次性任务，在新的练习目录准备同样的 `source.txt`，然后调用 `tars.exe run --goal "读取 source.txt，把编号写入 report.md，再读回核对"`，检查实际文件和退出码。要演示 TUI，把上例的 `tars.exe chat` 换成 `tars-tui.exe`，按界面提示审批。三种入口必须分别操作，不能拿其中一种的结果代替另一种。

## 子 Agent、MCP 与 Web

普通任务先直接交给主 Agent。需要独立复核时，可以明确要求“调用 reviewer 子 Agent，只读取 report.md，核对编号并报告结果”。内建 reviewer 的工具白名单只有 `read_file` 和 `list_dir`；它没有 shell、写文件或继续派生子任务的权限。子任务同样使用原会话工作区和审批规则；后台子任务的结果通过 `agent_result` 获取，父任务结束不等于所有后台子任务已经结束。

MCP 仅连接你明确配置的工具服务。把服务条目写入受信 `config.toml`，下面以已有的本地 stdio 服务为例，路径均需替换为真实安装位置：

```toml
[[mcp.servers]]
name = "local-tools"
transport = "stdio"
trusted = true
command = "C:/path/to/python.exe"
args = ["C:/path/to/server.py"]
cwd = "C:/path/to/mcp-project"
```

服务也可使用 `streamable_http` transport 和 `url`。配置变更后，在没有需要保留的运行任务时停止并重新启动 Core。模型看到的工具名形如 `local-tools__工具名`；连接失败不会伪装成工具成功，先检查 Core 日志中的服务名和错误。`trusted = true` 表示允许启动这个服务，不代表它的工具已自动通过每次审批，也不代表服务被 Docker 隔离。

Core 已运行时，在另一个终端执行：

```powershell
uv run --no-sync tars-web --open
```

Web 只监听本机地址，启动器生成有效期 60 秒的一次性登录链接。它提供会话、运行和事件查询，不提交聊天、不审批、不取消任务。看完后在 Web 所在终端按 Ctrl+C 停止该服务。

## 数据与常见故障

SQLite `state.db` 是会话、Run 和持久事件的依据。`artifacts/sessions/<session_id>/notes.md` 保存模型主动记录的会话笔记；每个 `runs/<run_id>/events.jsonl` 只记录该次运行，用于排查，不代替数据库状态。历史文件导入器已退出 V1，既有 SQLite 数据库仍按原有迁移升级，旧文件不会自动转成新会话。

- **模型请求失败**：核对配置文件、模型名、端点和专用 key。默认一次逻辑调用最多 120 秒、最多尝试 2 次；收到流事件后不自动重试，认证或协议错误直接失败。
- **工具失败**：先看实际错误和审批结果。文件路径应在该会话的工作区内；修改任务后重新提交，不把失败回复当作执行成功。
- **长会话**：在没有运行任务时输入 `/compact` 请求压缩。摘要失败时保留原有上下文；压缩也消耗模型请求。`run metrics` 不是账户账单，不能用它冒充包含所有手动压缩的完整费用。
- **诊断文件缺失**：检查日志路径和磁盘权限。日志失败不证明工具没有执行；用 `tars run status <run_id>` 核对运行状态，再检查实际产物。
- **命令参数报错**：连续对话用 `tars chat [--resume <session_id>]`，一次性任务用 `tars run --goal "任务"`；`tars run status/cancel/metrics` 仍只查询或管理已有 Run，不能和 `--goal` 混用。

当前验证范围见[V1 基准记录](docs/baseline/VERIFICATION_SUMMARY.md)与[已知限制](docs/baseline/LIMITATIONS.md)。[模型配置与请求预算](docs/baseline/MODEL_SETUP.md)说明配置方法。真实模型和工具验收须使用当前明确授权的账号、工作目录及调用额度。
