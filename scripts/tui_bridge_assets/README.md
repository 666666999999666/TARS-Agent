# Windows ConPTY 真实 TUI 验收桥

此目录只属于 QA：固定启动当前 Python 的 `-m tars_agent.tui --resume <session>`，由 Windows ConPTY 传递真实键盘、尺寸和原始终端输出，再由本地 xterm.js 显示。没有 Textual Pilot、模型回复替身、通用 shell 或主 Web 面板入口。

准备依赖：`uv sync --locked --group qa --group security` 与 `npm --prefix web ci`。pywinpty 仅在 Windows 的 qa 组安装；xterm 与 addon-fit 只列在 Web devDependencies，不进入只读面板代码。

启动命令：

```powershell
.venv\Scripts\python.exe scripts/tui_conpty_bridge.py --core-port <Core端口> --session <已有session_id> --workspace <工作区绝对路径> --home <独立客户端HOME> --output build/qa/tui-real --port 0
```

Core 与会话需事先通过正式命令/RPC创建。桥不会创建或重放模型任务。TUI 子进程使用独立的 client-config，只读取会话所需的本地端口配置，不继承模型 key，也不读取 Core 的凭证配置。

启动信息只打印绑定地址、桥 PID 和状态文件路径。输出目录的 `bridge.json` 包含实际端口、PTY PID、generation、尺寸、nonce 和带 fragment 的私有入口 URL；通过自动化工具读取该 URL 打开浏览器，不在公共日志或聊天中回显 nonce。网页加载后移除地址栏 fragment，认证值仅保存在该 origin 的 sessionStorage。

所有 API 都是 POST，必须同时满足精确 Host、精确同源 Origin 和 `X-Tars-Bridge-Token`。可用操作只有 `/api/poll`、`/api/input`、`/api/resize`、`/api/restart`、`/api/shutdown`；restart 必须空对象且当前 TUI 已退出，参数始终复用原 CLI 配置。静态资源从本地 node_modules 白名单加载，无 CDN、目录遍历或执行任意命令的入口。

Playwright 操作时聚焦 `.xterm-helper-textarea`，再使用真实 keyboard 输入。等待屏幕中的 TUI `ready` 和输入提示出现后再输入；桥的 PID 存活只意味着进程已启动。Ctrl+Q 结束 TUI 后，点击“重启同一 TUI”才会再次启动；不会掉入 shell。Ctrl+Q 不关闭 Core 或后台任务。

结束时对 `/api/shutdown` 发送带认证和 Origin 的空对象。桥先发送 Ctrl+Q，再在有限等待后仅终止它创建的仍存活 PTY 进程树；不关闭 Core。Core 的关闭由其所有者使用正式控制 token 完成。全部原始 PTY 输出保存在私有 `pty-output-<generation>.ansi.log`，HTTP输出缓存为8MiB，历史截断会明确提示。

## 已验证与限制

空会话验证通过：真实中文草稿输入、Ctrl+Q、显式同 TUI 重启、缺少 nonce/错误 Origin 的403、通用 exec 接口404；未发送消息，数据库中 runs/messages 均为0。真实终端截图保存在本地 QA 证据目录。

浏览器的 DA1/DA2 能力回复在 ConPTY 的 Windows 输入通道会成为文本。桥只过滤明确的 DA1/DA2 机器应答，原始 PTY 输出不修改，用户键盘/粘贴照常传入；不会无条件删除与 Shift+F3 等按键可能重叠的 DSR 序列。鼠标协议和原生 Windows IME 不在本次空会话验证范围内。

截图使用 xterm.js 6.0.0 与 Cascadia Mono/Consolas/微软雅黑回退字体，字符宽度、抗锯齿、行距、选择区和浏览器焦点行为可能不同于 Windows Terminal。此桥证明实际 TUI/ConPTY 流程，不冒充 Windows Terminal 原生渲染；浅深主题及业务流程由正式视觉验收记录分别说明。

API依据：[pywinpty 官方实现](https://github.com/andfoy/pywinpty/blob/main/winpty/ptyprocess.py)、[xterm.js Terminal API](https://xtermjs.org/docs/api/terminal/classes/terminal/)、[FitAddon 文档](https://xtermjs.org/docs/guides/using-addons/)。安装后还核对了 pywinpty 3.0.5 的低层 PTY docstring 与本地 xterm typings。
