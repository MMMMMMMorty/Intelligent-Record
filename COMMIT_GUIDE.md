# Intelligent-Document 提交规范指南

## 📋 目录

1. [Git 工作流](#git-工作流)
2. [提交信息规范](#提交信息规范)
3. [分支管理](#分支管理)
4. [常用命令](#常用命令)

---

## Git 工作流

### 基本流程

```bash
# 1. 获取最新代码
git pull origin main

# 2. 创建功能分支（推荐）
git checkout -b feature/your-feature-name

# 3. 进行代码修改...

# 4. 查看修改状态
git status

# 5. 添加修改到暂存区
git add .
# 或添加特定文件
git add path/to/file.py

# 6. 提交修改
git commit -m "feat: 添加新功能"

# 7. 推送到远程
git push origin feature/your-feature-name

# 8. 创建 Pull Request 合并到 main 分支
```

---

## 提交信息规范

### 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型 (Type)

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 (feature) |
| `fix` | 修复 bug |
| `docs` | 文档更新 |
| `style` | 代码格式调整（不影响功能）|
| `refactor` | 代码重构 |
| `perf` | 性能优化 |
| `test` | 添加测试 |
| `chore` | 构建过程或辅助工具的变动 |
| `ci` | CI/CD 相关更改 |

### 范围 (Scope)

可选，表示修改影响的范围：

- `backend` - 后端服务
- `frontend` - 前端界面
- `api` - API 接口
- `db` - 数据库
- `docker` - Docker 配置
- `docs` - 文档

### 示例

```bash
# 新功能
git commit -m "feat(backend): 添加语音识别接口"

# Bug 修复
git commit -m "fix(api): 修复笔录保存失败的问题"

# 文档更新
git commit -m "docs: 更新 API 文档"

# 性能优化
git commit -m "perf(asr): 优化语音识别速度"

# 代码重构
git commit -m "refactor(engine): 重构模型加载逻辑"
```

---

## 分支管理

### 分支命名规范

| 分支类型 | 命名格式 | 示例 |
|---------|---------|------|
| 主分支 | `main` | `main` |
| 功能分支 | `feature/<描述>` | `feature/asr-streaming` |
| 修复分支 | `fix/<描述>` | `fix/db-connection` |
| 发布分支 | `release/<版本>` | `release/v1.0.0` |
| 热修复分支 | `hotfix/<描述>` | `hotfix/security-patch` |

### 工作流程

```bash
# 创建并切换到功能分支
git checkout -b feature/new-feature

# 完成开发后，先更新 main 分支
git checkout main
git pull origin main

# 合并 main 到功能分支（解决冲突）
git checkout feature/new-feature
git merge main

# 推送功能分支
git push origin feature/new-feature

# 在 GitHub 上创建 Pull Request
# 代码审查通过后合并到 main

# 删除本地功能分支
git branch -d feature/new-feature

# 删除远程功能分支
git push origin --delete feature/new-feature
```

---

## 常用命令

### 基础命令

```bash
# 克隆仓库
git clone https://github.com/MMMMMMMorty/Intelligent-Document.git

# 查看状态
git status

# 查看提交历史
git log --oneline --graph

# 查看文件修改
git diff

# 查看某文件的修改历史
git log -p path/to/file
```

### 分支操作

```bash
# 查看分支列表
git branch -a

# 创建分支
git branch new-branch

# 切换分支
git checkout branch-name
# 或
git switch branch-name

# 创建并切换分支
git checkout -b new-branch
# 或
git switch -c new-branch

# 删除本地分支
git branch -d branch-name
# 强制删除
git branch -D branch-name

# 删除远程分支
git push origin --delete branch-name
```

### 撤销操作

```bash
# 撤销工作区的修改
git checkout -- file.txt

# 撤销暂存区的文件
git reset HEAD file.txt

# 修改最后一次提交
git commit --amend

# 回退到指定版本
git reset --hard commit-id

# 查看所有操作记录
git reflog
```

### 远程操作

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add origin https://github.com/MMMMMMMorty/Intelligent-Record.git

# 获取远程更新
git fetch origin

# 拉取并合并
git pull origin main

# 推送到远程
git push origin main

# 强制推送（谨慎使用）
git push -f origin main
```

### 标签管理

```bash
# 查看标签
git tag

# 创建标签
git tag -a v1.0.0 -m "版本 1.0.0"

# 推送标签到远程
git push origin v1.0.0

# 推送所有标签
git push origin --tags

# 删除本地标签
git tag -d v1.0.0
```

---

## 📝 提交检查清单

在提交代码前，请检查：

- [ ] 代码可以正常运行
- [ ] 已添加必要的注释
- [ ] 已更新相关文档
- [ ] 提交信息符合规范
- [ ] 已排除敏感信息（密码、密钥等）
- [ ] 大文件已加入 .gitignore

---

## 🔒 安全提醒

1. **永远不要提交敏感信息**：密码、API 密钥、私钥等
2. **使用 .gitignore**：排除日志文件、数据库文件、模型文件等
3. **审查提交内容**：使用 `git diff --cached` 查看即将提交的内容
4. **保护主分支**：建议启用分支保护规则

---

## 📚 参考资源

- [Pro Git 中文文档](https://git-scm.com/book/zh/v2)
- [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/)
- [GitHub Flow](https://docs.github.com/cn/get-started/quickstart/github-flow)
