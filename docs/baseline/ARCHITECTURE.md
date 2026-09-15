# 沿着一次任务学习 TARS-Agent

先记住主线：你通过 CLI 或 TUI 提交任务，后台 Core 接收并保存它，AgentLoop 请求模型；模型要求调用工具时，程序检查参数和权限，执行工具，再把真实结果交回模型。运行结束后，负责这次运行的对象先保存结果，再通知客户端。

日常配置和启动见 [README](../../README.md) 与 [RUNBOOK](../../RUNBOOK.md)，实际验收结果见 [验证摘要](VERIFICATION_SUMMARY.md)。本页只解释最终代码，不把完成重构等同于你已经掌握全部实现。

## 1. 用一个文件任务串起主流程

在专用练习目录准备 `input.txt`，内容是“今天完成了两个 Python 练习”。通过 `tars chat`、`tars run --goal` 或 TUI 提交：“读取 input.txt，把一句话摘要写到 summary.txt，完成后说明文件实际内容。”预期过程如下；具体工具次数由模型决定，最后仍要检查生成文件。

1. **客户端提交输入。** CLI 的两个薄命令都调用 [cli/client.py](../../src/tars_agent/cli/client.py) 的 `run_client()`。`TerminalClient.run()` 创建或恢复会话，先订阅事件，再由 `_submit()` 发送 `session.send_message`。TUI 保留 [tui/app.py](../../src/tars_agent/tui/app.py) 的 `on_chat_text_area_submitted()` 和 `_do_send_message()`。它们都用 SocketClient 与同一个 Core 通信，客户端消息 ID 用于识别重复提交。
2. **后台创建运行。** [core/app.py](../../src/tars_agent/core/app.py) 的 `_session_send_handler()` 调用 [runtime/service.py](../../src/tars_agent/core/runtime/service.py) 的 `submit_message()`。运行时保存输入和 Run，`_start_run()` 让 RunSupervisor 管理实际执行的协程。
3. **准备模型看到的内容。** `RuntimeService._execute_run()` 读取会话工作目录、成功历史和笔记，明确传给 [runner.py](../../src/tars_agent/core/runner.py) 的 `AgentRunner.run_and_capture()`。Runner 组织 ExecutionContext、Provider 和工具清单，再调用共享执行函数。它不另写一套会话历史，也不决定数据库终态。
4. **模型决定下一步。** [loop.py](../../src/tars_agent/core/loop.py) 的 `AgentLoop.run()` 将消息和工具描述传给 `AnthropicProvider.chat()`。模型可能返回 `read_file` 的结构化调用请求。程序根据 `stop_reason` 和工具调用字段处理结果，不靠猜测回答中的某句话来执行命令。
5. **程序执行工具。** [tools/invocation.py](../../src/tars_agent/core/tools/invocation.py) 的 `invoke_tool()` 处理工具查找、参数校验、权限检查、超时与错误。文件工具经 RuntimeRouter 进入配置的执行环境；需要批准时，由当前 CLI 或 TUI 显示请求，用户决定以后才继续。非交互 CLI 不会自动批准。
6. **真实结果送回模型。** 工具返回读取内容或错误，AgentLoop 把它追加成 `tool_result`，再次请求模型。后续 `write_file` 也经过同样检查。写入失败必须作为失败结果交回，不能用“准备写入”冒充已经写入。
7. **保存结果并结束。** 模型给出最终回答，或运行触及步数、错误、取消等结束条件后，执行层清理资源并返回 RunOutcome。`RuntimeService._finish_run()` 提交数据库状态和适用的成功历史，再发布完成事件。chat/TUI 显示结果后继续接受输入；goal 核对 Run 快照与工具指标，输出答案和退出码。你核对 `summary.txt` 与回答一致。

这是一轮可以重复多次的模型与工具循环。每得到工具结果，都可能再次请求模型；最终回答、步数限制、错误或取消是它的结束条件。

### 主流程的职责划分

- 保留 CLI 连续聊天、一次性 goal 和 TUI。CLI 两个命令文件只做转发，共用一个 TerminalClient；后台仍只有同一个 AgentLoop 和运行时。
- RuntimeService 管理会话历史和主运行，Runner 接收明确的参数并执行一次任务。
- 主 Agent 和子 Agent 共用 `tools/assembly.py` 的 `build_base_registry()`，文件与计划工具只维护一份基础清单；`execution.py` 共用循环异常处理和资源清理。
- 子运行的终态统一交给 `BackgroundTaskRegistry.finish()`，不用正常完成、取消和关闭三条路径各发一次完成通知。
- 每份事件日志按自己的 `run_id` 过滤，完成记录由保存终态的对象发出。待清理期间继续记录，确认终态后才关闭文件。

`/orchestrate` 将任务分给 planner、executor 和 reviewer。父级提供必要文件工具，子角色仍取自身白名单与父权限的交集；审批和 Docker 约束始终生效，reviewer 只有读取权限。

## 2. 目录与状态分别由谁负责

源码根目录是 `src/tars_agent/`。先认识以下位置，不需要一开始读完每个子目录。

| 位置 | 一句话职责 |
| --- | --- |
| `tui/`、`cli/` | TUI 提供界面；CLI 提供连续聊天、一次性目标和管理命令，两个任务命令共用 `client.py`。 |
| `core/app.py`、`core/transport/` | 启动 Core 的依赖，把客户端命令交给对应函数。 |
| `core/runtime/` | 管理主运行的执行、取消、恢复和成功历史。 |
| `core/runner.py`、`core/execution.py` | 准备和执行一次循环，清理资源，把结果交还给调用方。 |
| `core/loop.py`、`core/context.py` | 管理模型与工具循环，以及本次模型请求需要的消息。 |
| `core/llm/` | 将模型协议转换成内部的响应与工具调用数据。 |
| `core/tools/`、`core/permissions/`、`sandbox/` | 定义工具、检查授权、执行受工作区和运行环境约束的操作。 |
| `core/persistence/`、`core/events/` | 读写 SQLite 状态，保存和交付可回放事件，记录运行日志。 |
| `core/artifacts.py`、`core/task/` | 定位产物、保存笔记，维护模型使用的计划项。 |
| `core/subagent/`、`core/agents/` | 管理子运行和角色的工具范围。 |
| `core/mcp/` | 用官方 SDK 连接已配置的 MCP 工具服务。 |
| `core/skills/`、`core/memory/`、`core/compact/` | 加载任务提示和上下文文件，压缩过长的会话上下文。 |
| `web/` | 提供可选的本地只读查询接口；仓库顶层 `web/` 保存前端源码。 |
| `core/eval/`、`core/observability/`、`core/trace/` | 评测、汇总运行信息和排查过程，不决定运行是否成功。 |

### 四种常被混叫成“任务”的东西

| 概念 | 可以怎样理解 | 保存位置与负责者 |
| --- | --- | --- |
| Session | 可以连续交流的一段会话，固定一个工作目录。 | `state.db`；RuntimeService 管理，状态为 `ready/running/closed`。 |
| Run | 一次实际执行，例如用户发来的一条任务，或它派生的子运行。 | `state.db`；主 Run 由 RuntimeService、子 Run 由 BackgroundTaskRegistry 管理；状态为 `queued/running/succeeded/failed/cancelled/interrupted`。 |
| 计划项 Task | 模型记下的“先读文件，再写摘要”这类待办。 | 当前 Run 的 `.tasks/task_*.json`；TaskManager 管理。完成计划项不会把 Run 自动标为成功。 |
| `asyncio.Task` | Python 当前进程里正在执行的协程句柄。 | 内存；RunSupervisor 或子运行注册表用它等待和取消。进程重启不会恢复这个对象本身。 |

SQLite 保存会话、运行、消息和可回放事件。计划项、笔记和日志各有用途，“SQLite 是运行状态的依据”不等于所有内容必须存在 SQLite。

- 默认运行根为 `~/.tars-baseline`，也可由真实进程环境中的 `TARS_HOME` 指定。
- `artifacts/sessions/<session_id>/notes.md` 由 [ArtifactStore](../../src/tars_agent/core/artifacts.py) 读写，是后续请求可读取的补充内容，不是聊天历史，也不会自动进入别的会话。
- `artifacts/sessions/<session_id>/runs/<run_id>/` 保存该 Run 的事件日志和计划项。JSONL 日志用于检查经过的步骤，不负责决定终态或恢复执行。
- Provider 的请求预算账本统计发送授权额度，不能代替 `state.db` 判断运行状态；`run.metrics` 也不是账户账单。
- 项目不会自动导入其他目录中的个人会话数据。

## 3. 保留的复杂功能解决什么问题

**后台 Core 和事件回放。** 任务需要在客户端退出后继续执行，因此运行不能依赖 TUI 进程存活。Core 保存运行，TUI 通过本地协议查询和订阅。EventBus 依次通知订阅者；DurableEventHub 先把事件保存到 SQLite，再按 cursor 给客户端回放。Core 自己崩溃后，未完成 Run 标为 `interrupted`，不会自动重做可能已经产生副作用的工具。

**CLI 如何等待输入。** [cli/input.py](../../src/tars_agent/cli/input.py) 的 `TerminalInput.read()` 只负责读输入。Windows 交互终端由 `_read_console()` 检查可用按键，区分 Ctrl+C 取消和 Ctrl+Z 输入结束；管道及其他平台由 `_read_lines()` 在线程中把输入行放进队列。等待键盘时，客户端仍能接收事件。这个线程不请求模型、不执行工具，也不是 Core 后台服务；`daemon=True` 只是表示它不会阻止 CLI 进程退出。

**客户端退出与任务结果。** `TerminalClient._finish()` 读取 Core 的最终状态和工具指标。goal 主 Run 结束就退出；后台子 Run 未结束时只列出 ID。任一已发生的工具失败都会使 goal 返回 `1`，即使 Core 的主 Run 补救后成功。chat 的一轮完成或取消后继续输入，空闲 Ctrl+C 退出 `130`，EOF 退出 `0`。客户端不调用 `session.close` 或 `core.shutdown`。

**取消与退出。** CLI 运行中用 Ctrl+C 请求取消；TUI 用 Ctrl+X 取消、Ctrl+Q 只退出界面。请求取消不代表进程或容器已经停止；资源清理未确认时，系统保留待清理状态，事件文件也继续打开。确认后，首次提交终态的调用才发布完成通知。Core 关闭时清理自己管理的执行资源，但外部 MCP 服务已经接受的操作不保证能撤销。

**子 Agent。** 它把目标交给更小的角色上下文，例如 planner 读取目录提出计划、reviewer 检查结果。子 Agent 仍使用同一个 AgentLoop，每个子 Run 有自己的结果，归属父会话并记录 `parent_run_id`。工具范围取角色权限与父运行权限的交集；省略角色时默认不提供工具。内置 reviewer 只有 `read_file/list_dir`。后台子运行返回 ID，由 `agent_result` 取结果，嵌套深度受限制。保留它意味着还要学懂父子 Run、权限继承和取消树；三个角色的存在不证明效果比单 Agent 更好。

**MCP 与模型适配。** MCP 将外部服务的工具接入现有调用流程，协议和服务生命周期交给官方 SDK。当前保留 stdio 与 Streamable HTTP；只有实际配置并验收的服务才可作为个人使用案例。模型层主要是一个 Anthropic 协议 Provider，模型名和兼容端点可配置，不应描述成大量厂商独立适配。

**工具边界。** 用户审批回答“是否允许做这件事”；工作目录限制与 Docker 回答“操作能触及哪里、在哪个环境执行”。两者不能互相替代。默认 required 模式下 Docker 不可用就明确失败；preferred 模式的宿主回退需单独批准，已开始的调用不会换到另一个环境重做。

**压缩与只读 Web。** 压缩用于控制长对话输入量，失败时保留原成功历史；摘要可能丢细节，不能承诺无限记忆。Web 用来查看会话、运行和事件，不提供任务输入与审批；不使用 Web 时，日常 TUI 无需启动它。

## 4. 推荐阅读顺序与三个练习

第一轮先读主执行链，第二轮再学习配置、IPC、事件回放和子 Agent。每次只回答表中的一个问题。

| 顺序 | 打开位置与函数 | 这一遍弄懂什么 |
| --- | --- | --- |
| 1 | [loop.py](../../src/tars_agent/core/loop.py) 的 `AgentLoop.run()` | 模型为什么会被反复调用，工具结果怎样成为下一次输入？ |
| 2 | [context.py](../../src/tars_agent/core/context.py) 的 `ExecutionContext` | 当前有哪些消息，哪些内容会传给模型？ |
| 3 | [tools/invocation.py](../../src/tars_agent/core/tools/invocation.py) 的 `invoke_tool()` | 谁检查参数、审批和失败？ |
| 4 | [runner.py](../../src/tars_agent/core/runner.py) 的 `run_and_capture()` | 模型、工具、工作目录和历史是谁传进来的？ |
| 5 | [runtime/service.py](../../src/tars_agent/core/runtime/service.py) 的 `submit_message()`、`_execute_run()`、`_finish_run()` | 输入何时落盘，成功历史何时提交？ |
| 6 | [cli/client.py](../../src/tars_agent/cli/client.py) 的 `TerminalClient.run()`、`_submit()`、`_finish()` | 两个 CLI 命令怎样共用提交与审批，为什么 Run 成功时 goal 仍可能返回 `1`？ |
| 7 | [tui/app.py](../../src/tars_agent/tui/app.py) 的 `_do_send_message()`、`_socket_loop()` | 界面怎样连接 Core、重新接回会话？ |
| 8 | [subagent/registry.py](../../src/tars_agent/core/subagent/registry.py) 的 `finish()` | 正常完成与取消竞争时，怎样保持状态和通知一致？ |

练习一：不运行模型，阅读 [test_event_writer.py](../../tests/unit/test_event_writer.py) 中的 `test_shared_bus_writers_only_record_their_own_run`，用自己的话解释两个 Run 共用总线为什么不会串写日志。

练习二：阅读 [test_spawn_agent_tool.py](../../tests/unit/test_spawn_agent_tool.py) 中的 `test_orchestrate_profiles_get_only_the_parent_permitted_workspace_tools`。解释为什么 executor 能执行文件操作，而 reviewer 只有读取权限，以及父运行去掉 `bash` 后为何子角色不能把它加回来。

练习三：在专用目录执行开头的摘要任务，检查该 Run 的 JSONL 与真实文件，分别找出“要求执行工具”“工具返回结果”“运行结束”的记录。只有前两处时，不能自行推断任务已经成功结束。真实调用应使用已授权的额度。

能独立解释一次模型工具循环、Session/Run 的区别、审批与隔离、取消后的清理，再练习子 Agent 和事件回放。尚未讲清楚的概念继续列为学习任务，不以答题表现代替工程验收。

## 5. 项目介绍与面试练习

### 项目介绍

TARS-Agent 是由 liaoqizai 个人自主开发和维护的 Python Agent 项目，提供 CLI/TUI 交互、模型工具循环、SQLite 会话与运行记录、Docker 工具隔离、审批、子 Agent 和 MCP 接入。核心设计围绕一套执行循环、明确的数据归属和可检查的失败处理展开。

描述项目时，围绕实际实现与能够解释的代码说明设计。没有测量过的性能、用户规模和线上运行数据不应作为项目指标。

### 常见问题

**模型怎样使用工具？** 模型返回工具名和参数后，程序先检查工具范围、参数与权限，再执行工具，把真实结果追加到消息中发给模型。最终结果来自这条循环，而不是只打印模型准备执行的计划。

**为什么需要后台 Core？** 任务需要在客户端退出后继续执行。Core 保存运行状态，客户端重新连接后按会话和事件游标恢复。客户端退出与取消任务是两个明确的操作。

**为什么 Run 成功时 goal 可能退出 1？** 模型可以补救一个工具错误，最终 Run 因此成功，但一次性 CLI 采用严格退出码：这轮出现过工具失败就返回 1，避免脚本调用方误认为全程无错误。

**怎样保证取消后能继续使用？** 取消沿主子任务关系传递，等待所属工具和资源收尾后保存终态。CLI 随后回到输入状态；回归测试检查终态工具记录、再次提交以及延迟文件是否出现。
