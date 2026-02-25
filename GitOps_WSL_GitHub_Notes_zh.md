# WSL 下把他人 clone 的项目推到自己的 GitHub（含 SSH 443 绕过、提交与远程配置）

这份笔记记录了我在 WSL（Linux）里把一个从他人仓库 `git clone` 得到的项目，上传到我自己 GitHub 仓库的全过程。

适用场景：

- 你手上已经有一个带 `.git/` 的项目目录（来自别人仓库）
- 你希望：**保留原仓库历史**，但把远程推送目标改成自己的仓库
- 你在 WSL 网络环境下可能 **SSH 22 端口不可用**（常见于公司/校园网/代理环境）
- 你希望 Python 环境用 `uv` 管理（但 Git 登录/推送仍然是 Git 的事）

> 说明：`uv` 负责 Python 依赖与运行；Git 账号登录/推送依然由 `git`/`ssh` 负责，二者是两套工具。

---

## 0. 先确认当前仓库状态

进入项目目录：

```bash
cd /home/cwoloc/MinivLLM
git status
git remote -v
git branch --show-current
```

你应该能看到类似：

- 当前分支：`main`（也可能是 `master`）
- `origin` 还指向原作者仓库（例如 `https://github.com/Wenyueh/MinivLLM.git`）

---

## 1. VS Code 提示没配 `user.name` / `user.email` 怎么办？

这是 Git 的提交者信息（写进 commit 记录里），不配的话就无法提交。

### 1.1 仅对“当前项目”设置（推荐新手，避免影响别的仓库）

在项目目录执行：

```bash
git config user.name "zowort"
git config user.email "248995571+MDX-0003@users.noreply.github.com"
```

验证：

```bash
git config --local --list | grep -E 'user.name|user.email'
```

> 说明：`users.noreply.github.com` 是 GitHub 的隐私邮箱形式；你也可以用真实邮箱。

### 1.2（可选）全局设置

如果你确定以后所有仓库都用同一身份：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

---

## 2. SSH 测试命令写对了吗？

你可能会不小心写成：

```bash
ssh -T git@zowort@qq.com
```

这条命令会把主机当成 `qq.com`（因为 `ssh` 格式是 `用户@主机`），所以它 **不是在测 GitHub**。

正确测试 GitHub SSH 的命令是：

```bash
ssh -T git@github.com
```

如果成功，会看到：

> You've successfully authenticated, but GitHub does not provide shell access.

（退出码通常是 1，这不代表失败，这是 GitHub 的正常行为：只认证、不提供 shell。）

---

## 3. WSL 网络下 22 端口被拦：用 SSH 443 绕过

如果你执行：

```bash
ssh -T git@github.com
```

出现类似：

- `Connection closed by <某个 IP> port 22`

常见原因是：你的网络对 `github.com:22` 不通。

### 3.1 准备 SSH Key（如果你还没有）

```bash
ssh-keygen -t ed25519 -C "你的邮箱"
```

把公钥内容复制到 GitHub：Settings → SSH and GPG keys

```bash
cat ~/.ssh/id_ed25519.pub
```

### 3.2 配置 GitHub SSH 走 443 端口

编辑（或创建）`~/.ssh/config`：

```sshconfig
Host github.com
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

并确保权限正确：

```bash
chmod 600 ~/.ssh/config
```

### 3.3 首次连接的 known_hosts 处理（避免交互提示）

首次连接可能会提示确认主机指纹（需要输入 `yes`）。
在一些自动化环境里不好交互，可以先写入 known_hosts：

```bash
ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts
```

然后再测试：

```bash
ssh -T git@github.com
```

> 安全提示：更严谨的做法是对照 GitHub 官方公布的 SSH 主机指纹进行核验。

---

## 4. 把“原作者仓库”保留为 upstream，把自己的仓库设为 origin

目标：

- `upstream`：指向原作者仓库（以后方便同步更新）
- `origin`：指向你自己的仓库（你要 push 的地方）

### 4.1 重命名原来的 origin 为 upstream

```bash
git remote rename origin upstream
```

### 4.2 添加你的仓库作为 origin（推荐用 SSH URL）

```bash
git remote add origin git@github.com:MDX-0003/Minivllm-Learn.git
```

检查：

```bash
git remote -v
```

---

## 5. 提交本地修改（Commit）

### 5.1 查看改动

```bash
git status
git diff
```

### 5.2 不要把环境文件误提交

这类目录通常不该提交：`.venv/`、`__pycache__/`、`.pytest_cache/` 等。
本项目的 `.gitignore` 已包含 `.venv` 等常见规则。

另外：如果你不小心在仓库目录里生成了奇怪的文件（例如把命令输出重定向进了文件），建议先删掉再提交。

### 5.3 add + commit

```bash
git add -A
git commit -m "chore: save local changes"
```

---

## 6. 推送到你自己的仓库（Push）

首次推送并建立跟踪关系：

```bash
git push -u origin main
```

之后就可以直接：

```bash
git push
```

---

## 7.（可选）以后如何从原作者同步更新？

如果你保留了 `upstream`：

```bash
git fetch upstream
git merge upstream/main
```

或使用 rebase（更线性，但新手先别急着用）：

```bash
git fetch upstream
git rebase upstream/main
```

---

## 8. Python 环境用 uv 的最小用法（补充）

在项目根目录：

```bash
uv sync
uv run python main.py
```

不需要 `source .venv/bin/activate`，也不需要手动 `pip install`。
