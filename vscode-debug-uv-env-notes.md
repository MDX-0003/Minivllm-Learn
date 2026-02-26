# VS Code Debug 找不到依赖：现象、根因与修复记录（uv 项目内 .venv）

日期：2026-02-26  
仓库：`Minivllm-Learn`

## 现象（你看到的是什么）

- 在 VS Code 里点击 Debug/启动调试运行 Python 代码。
- 代码启动后报错 **找不到依赖**（典型表现是 `ModuleNotFoundError: No module named ...`）。
- 但你明确知道依赖应该已经用 **uv** 安装到了项目路径下的虚拟环境里。

## 根因（为什么会这样）

**VS Code 调试默认使用的 Python 解释器/环境，不一定是 uv 创建的项目内虚拟环境 `.venv`。**

当 Debug 实际使用的是系统 Python 或其它解释器时：

- uv 安装在 `.venv` 里的第三方依赖对这个解释器不可见 → 报“找不到依赖”。
- 另外，本项目是典型的 **src-layout**（包在 `src/` 下），如果运行时工作目录(`cwd`)或 `PYTHONPATH` 不合适，即使依赖都装好了，也可能出现本地包导入不稳定的问题。

## 我为解决问题做的事情（已落到仓库文件里）

### 1) 固定 VS Code 使用 uv 的 `.venv` 解释器

新增文件：`/.vscode/settings.json`

关键配置：

- `python.defaultInterpreterPath`: 指向 `${workspaceFolder}/.venv/bin/python`
- `python.analysis.extraPaths`: 加入 `${workspaceFolder}/src`，提升 `import myvllm` 等导入的稳定性与智能提示准确性

> 效果：让 VS Code（包括调试和 Pylance 分析）“优先/默认”使用项目内的 uv 环境，而不是漂到系统 Python。

### 2) 修复并简化调试配置（launch.json 去重 + 加强运行环境）

修改文件：`/.vscode/launch.json`

改动点：

- 删除重复的调试项（之前存在两个同名 `Python Debugger: Current File` 配置）。
- 保留一个配置并命名为：`Python Debugger: Current File (uv .venv)`
- 增加：
  - `"cwd": "${workspaceFolder}"`：调试时从项目根目录运行，避免相对路径/导入差异
  - `"env": { "PYTHONPATH": "${workspaceFolder}/src" }`：确保 src-layout 在调试时也能正确导入本地包

### 3) 本地验证（在本次会话中已实际跑过）

我用 uv 的解释器做了最小验证，确认环境没问题：

- `/home/dministrator/Minivllm-Learn/.venv/bin/python` 能成功 `import myvllm`
- 并能成功导入关键第三方依赖：`transformers`, `torch`, `xxhash`

> 结论：依赖确实安装在 `.venv` 内，问题集中在 **VS Code Debug 没选对解释器/运行时导入路径**。

## 最终状态（修复后你应该得到什么）

- VS Code Debug 运行会在项目根目录执行。
- Debug 运行会带上 `PYTHONPATH=.../src`。
- VS Code 的默认 Python 解释器会固定为 `.venv/bin/python`。
- 以上组合可以消除“Debug 找不到依赖”的常见来源：**解释器不对** 和 **src-layout 导入路径不对**。

## 备注（如仍有报错，下一步排查建议）

如果后续仍遇到某个特定依赖找不到：

1. 先确认 VS Code 当前解释器是否确实是 `${workspaceFolder}/.venv/bin/python`。
2. 贴出报错堆栈前 20 行，通常能快速判断是：
   - 解释器仍未切换成功
   - 特定依赖未安装到 `.venv`
   - 运行入口需要用 `python -m ...`（模块方式）而不是直接跑单个文件
