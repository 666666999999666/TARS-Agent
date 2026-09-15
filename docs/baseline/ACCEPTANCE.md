# V1 验收方法

以当前 V1 的代码和使用说明为准。模型消耗、依赖安装、外部操作和服务保留按当前明确授权执行。

每项检查记录环境、操作或命令、实际结果、源码 HEAD、dirty 状态与证据位置，区分通过、失败、未验证和环境阻塞。旧版本通过、测试配置存在或进程能启动，都不能代替本次实际运行。

## 离线检查与构建

```powershell
uv run --no-sync python scripts/qa.py quick
uv run --no-sync python scripts/qa.py coverage
uv run --no-sync python scripts/qa.py build
uv run --no-sync python scripts/qa.py security
uv run --no-sync tars eval run --suite evals/internal-deterministic.json --output build/qa/internal-runtime
```

这些 QA 命令使用 `build/qa` 保存产物。覆盖率门槛为行 65%、分支 49%、combined 62%、改动行 80%；比较基准固定为当前仓库的初始化提交，不允许用 HEAD 替换。初始化提交只保存仓库元数据，随后提交源码，因此首次 V1 检查也覆盖全部新增源码。必须完整克隆历史；缺失依赖、跳过项和未执行的外部检查分别记录。安全依赖检查可能联网，`full` 不包含真实用户验收。

构建成功只证明产物能生成。声称某种安装包可用前，应在隔离目录安装并实际启动该产物，同时检查 `tars chat --help`、`tars run --help` 及实际 CLI 提交路径。源码目录的通过结果不能替代 wheel 或 sdist 的安装验收。

验收通过后可为选定提交创建基准标签。构建、标签、包校验值和实际结果应指向同一版本；发布状态以仓库和交付清单为准。

## CLI 正式入口

真实启动 `tars chat [--resume <session_id>]` 和 `tars run --goal "任务"`，通过终端输入操作。局部测试可以使用受控 Core，但真实验收必须请求真实模型和执行真实工具，并保存命令、终端输入输出、退出码、工作区和对应 Run。

- chat 连续多轮交互；运行中 Ctrl+C 取消后能继续新任务；EOF 退出 `0`、空闲 Ctrl+C 退出 `130`；恢复原会话不重放最后输入。
- goal 成功返回 `0`，参数错误返回 `2`，取消返回 `130`。模型或任一工具失败返回 `1`，即使模型补救后 Run 成功也不能返回 `0`。
- 批准与拒绝均通过 CLI 的实际提示操作；拒绝无副作用。非 TTY 场景需要审批时明确拒绝，不能把管道中的 `y` 当授权。
- 主 Run 结束时 goal 退出，并列出仍运行的后台子 Run；不等待全部后台任务，也不因为退出客户端就取消它们。验证之后仍能用所列 ID 查询。
- 客户端退出后 Core 仍运行，chat 会话仍可恢复。CLI 不发送 `session.close` 或 `core.shutdown`；对外部运行的停止由明确的管理命令负责。

`scripts/acceptance_cli.py` 通过 Windows ConPTY 启动真实 CLI、输入审批字符和控制键。它的结果按实际场景记录；旧 `acceptance_workflows.py` 直接使用 RPC，不能证明 CLI 的输入、提示和退出码已经验收。

## 真实模型、工具与会话

真实任务通过 CLI 和 TUI 分别操作；[真实工作流脚本](REAL_WORKFLOWS.md)补充底层 Core 流程验证。RPC 脚本固定本机 DeepSeek 验收配置，读取原受信配置和已有请求账本；会话库及其他状态写入 `--state-home`，默认位于本轮 `build/v1-simplify/real-workflows` 证据目录中。`--output-root` 可改变证据根目录。不得把原 `state.db` 或仓库旁旧交付目录当成新验收的写入位置。

文件、权限、会话三类各连续验证三轮。某类失败后停止该类剩余轮次，修复后重验；模型额度按实际授权和受信配置约束，原账本持续计数。默认有限配置、显式不限次配置的计数规则由离线回归验证，不能为了测试限额发送无必要的真实请求。

至少同时检查：

- 模型返回真实内容，工具确实执行，最终回答与磁盘产物一致。
- 审批拒绝后没有对应副作用；取消后没有延迟执行，资源确实清理。
- RPC、SQLite 和持久事件的 Run 终态一致。同一终态 Run 下没有 `queued/running` 工具记录；脚本逐 Run 保存 `terminal_tool_check`。
- 连续交互、压缩、退出恢复和 Core 重启符合当前承诺，断线不自动重复提交最后输入。

Docker 验收覆盖真实引擎、镜像、路径与文件操作、Bash、超时、取消和清理，以及 required 失败和 preferred 逐次授权。MCP 使用本地真实 stdio/HTTP 服务验证协议边界；这不能代表未访问过的外部 MCP 服务已经可用。所有操作只针对本次拥有的工作区、会话、进程和容器。

## 界面与证据范围

TUI 需实际检查输入、流式显示、批准与拒绝、停止或失败后继续、退出恢复、断线重连和历史游标。本机可用浏览器 xterm.js 呈现真实 Windows ConPTY 中的 Textual 进程；这仍须真实输入和查看界面，底层 RPC 通过不能冒充 TUI 通过。

检查深浅主题和小窗口时要切换 Textual 自身主题，浏览器 colorScheme 不能替代它。审批标题、参数和选项应能同时辨认，已解决的历史审批不能重新变成可交互控件。该环境不自动覆盖原生 Windows Terminal 的特定字体和 IME。

Web 保持只读，单独验证认证、会话与运行查询、事件显示和断线状态。浏览器、终端桥和前端开发依赖属于验收环境，不是日常 TUI 的启动条件。

## 当前记录

当前结果统一见[验证摘要](VERIFICATION_SUMMARY.md)，限制见[已知限制](LIMITATIONS.md)。原始证据按该摘要中的位置核查。记录必须对应实际验收的源码和工作区；之后发生相关修改时补做受影响验证，不把旧包、旧日志或历史授权写成当前结论。覆盖率也要对应最终源码；旧 `offline-final-coverage.json` 的指纹对应说明不能充当新代码覆盖率报告。
