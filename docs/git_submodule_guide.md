# Git Submodule 完整指南

## 📋 目前狀態

### `lm-evaluation-harness` 的實際情況

經過檢查，`lm-evaluation-harness` **不是標準的 git submodule**，而是：
- ✅ 一個獨立的 git repository（有自己的 `.git/` 目錄）
- ✅ 指向 EleutherAI 的 lm-evaluation-harness
- ⚠️ 但**沒有**在主專案中正式註冊為 submodule（沒有 `.gitmodules` 檔案）

```bash
# 確認狀態
$ cd lm-evaluation-harness
$ git remote -v
origin  https://github.com/EleutherAI/lm-evaluation-harness (fetch)
origin  https://github.com/EleutherAI/lm-evaluation-harness (push)
```

---

## 🎯 兩種管理方式

### 方式 1: 保持現狀（獨立 Git Repo）✅ 推薦

**現況**：`lm-evaluation-harness` 是一個獨立的 git repository，只是放在專案目錄中。

#### 優點：
- ✅ 簡單直觀
- ✅ 可以直接在裡面 commit/push（如果你有權限）
- ✅ 可以隨時 `git pull` 更新
- ✅ 不需要處理 submodule 的複雜性

#### 缺點：
- ⚠️ 主專案不會追蹤 `lm-evaluation-harness` 的版本
- ⚠️ 團隊成員需要手動 clone 這個目錄
- ⚠️ 需要在 `.gitignore` 中排除

#### 如何管理：

```bash
# 1. 確保 .gitignore 排除此目錄
echo "lm-evaluation-harness/" >> .gitignore

# 2. 更新 lm-evaluation-harness
cd lm-evaluation-harness
git pull origin main  # 或其他分支

# 3. 如果你修改了程式碼（不建議）
cd lm-evaluation-harness
git add .
git commit -m "your changes"
# 注意：除非你 fork 了這個 repo，否則無法 push

# 4. 還原修改（你剛剛做的）
cd lm-evaluation-harness
git restore .
git clean -fd  # 刪除未追蹤的檔案
```

#### ⚠️ 重要提醒：

**不要修改 `lm-evaluation-harness` 內的程式碼！**

原因：
1. 這是第三方函式庫，你無法 push 修改
2. 更新時會衝突
3. 正確做法是：
   - 如果需要自訂功能 → 在**你的專案**中繼承/包裝它
   - 如果發現 bug → 提交 PR 給上游
   - 如果需要特定版本 → checkout 特定 commit/tag

---

### 方式 2: 轉換為正式的 Git Submodule

**適用情況**：如果你想讓主專案追蹤 `lm-evaluation-harness` 的版本。

#### 步驟：

```bash
# 1. 刪除現有的目錄（先備份！）
cd /home/frank23026407/TinyLlama
rm -rf lm-evaluation-harness/

# 2. 添加為 submodule
git submodule add https://github.com/EleutherAI/lm-evaluation-harness lm-evaluation-harness

# 3. 提交 submodule 配置
git add .gitmodules lm-evaluation-harness
git commit -m "Add lm-evaluation-harness as submodule"
```

#### 日後如何使用：

```bash
# Clone 專案時（新成員）
git clone <your-repo>
git submodule init
git submodule update

# 或一步完成
git clone --recursive <your-repo>

# 更新 submodule 到最新版本
cd lm-evaluation-harness
git pull origin main
cd ..
git add lm-evaluation-harness
git commit -m "Update lm-evaluation-harness to latest"

# 切換到特定版本（推薦）
cd lm-evaluation-harness
git checkout <commit-hash 或 tag>
cd ..
git add lm-evaluation-harness
git commit -m "Pin lm-evaluation-harness to v0.4.0"

# 還原 submodule 的修改
git submodule update --force
# 或
cd lm-evaluation-harness
git restore .
git clean -fd
```

---

## ⚠️ Git Submodule 的注意事項

### 1. Submodule 是指向特定 Commit 的指標

```bash
# 主專案記錄的是 submodule 的 commit hash，不是分支
$ git ls-tree HEAD lm-evaluation-harness
160000 commit abc123def456...  lm-evaluation-harness
```

**重要概念**：
- Submodule 不會自動更新
- 主專案記錄的是特定的 commit hash
- 即使 submodule 的 remote 更新了，你的專案還是用舊版本（除非手動更新）

### 2. Detached HEAD 狀態

```bash
# 當你 checkout 主專案的某個 commit 時
$ cd lm-evaluation-harness
$ git status
HEAD detached at abc123d  # ⚠️ 這是正常的！
```

**這是正常的！** Submodule 會處於 detached HEAD 狀態，因為它被固定在特定 commit。

如果你想在 submodule 中工作：
```bash
cd lm-evaluation-harness
git checkout main  # 切換到分支
# 做修改...
git add .
git commit -m "changes"
cd ..
git add lm-evaluation-harness  # 告訴主專案使用新 commit
git commit -m "Update submodule"
```

### 3. 修改 Submodule 的工作流程

```bash
# ❌ 錯誤做法：直接在 submodule 中修改
cd lm-evaluation-harness
# 編輯檔案...
git add .
git commit -m "my changes"
# 問題：你無法 push（沒有權限），且主專案不知道

# ✅ 正確做法 1：Fork + 修改
# 1. Fork EleutherAI/lm-evaluation-harness 到你的帳號
# 2. 修改 .gitmodules 指向你的 fork
# 3. 在你的 fork 中開發

# ✅ 正確做法 2：不修改 submodule（推薦）
# 在你的專案中繼承/包裝 lm-evaluation-harness 的類別
# 例如：創建 my_custom_model.py 繼承 HFLM
```

### 4. 更新 Submodule

```bash
# 方法 1：進入 submodule 手動更新
cd lm-evaluation-harness
git fetch origin
git checkout main
git pull origin main
cd ..
git add lm-evaluation-harness
git commit -m "Update lm-evaluation-harness"

# 方法 2：使用 git submodule 命令
git submodule update --remote lm-evaluation-harness
git add lm-evaluation-harness
git commit -m "Update lm-evaluation-harness"

# 方法 3：更新所有 submodules
git submodule update --remote --merge
```

### 5. 團隊協作的挑戰

```bash
# 問題：團隊成員 pull 主專案後，submodule 沒有自動更新
$ git pull origin main
# submodule 還是舊版本！

# 解決：需要手動更新
$ git submodule update --init --recursive

# 或設定自動更新（Git 2.14+）
$ git config submodule.recurse true
# 之後 git pull 會自動更新 submodules
```

---

## 🎯 我的建議

### 針對你的專案：保持現狀 ✅

**理由**：
1. `lm-evaluation-harness` 是第三方函式庫，你不會修改它
2. 你只是使用它的 API（`HFLM` 類別）
3. 不需要追蹤特定版本（直接用最新的即可）
4. 簡單、不需要處理 submodule 複雜性

**操作建議**：

```bash
# 1. 將它加入 .gitignore（如果還沒有）
echo "lm-evaluation-harness/" >> .gitignore
git add .gitignore
git commit -m "Ignore lm-evaluation-harness directory"

# 2. 在 README.md 中說明如何安裝
```

#### 在 README.md 中添加：

```markdown
## Setup

### 1. Clone lm-evaluation-harness

```bash
cd /path/to/TinyLlama
git clone https://github.com/EleutherAI/lm-evaluation-harness
```

### 2. Install dependencies

```bash
cd lm-evaluation-harness
pip install -e .
```
```

---

## 📝 常見問題

### Q1: 我應該 commit lm-evaluation-harness 到我的 repo 嗎？

**A**: ❌ **不應該**

原因：
- 它是第三方函式庫，有自己的 repo
- 會讓你的 repo 變得非常大
- 無法接收上游更新

正確做法：
- 在 `.gitignore` 中排除
- 在文檔中說明如何安裝
- 或使用 git submodule（如果需要版本控制）

### Q2: 如果我需要修改 lm-evaluation-harness 的程式碼怎麼辦？

**A**: 有幾種選擇：

1. **最佳實踐**：在你的專案中繼承/包裝
   ```python
   # my_models.py
   from lm_eval.models.huggingface import HFLM
   
   class MyCustomHFLM(HFLM):
       def __init__(self, *args, custom_param=None, **kwargs):
           super().__init__(*args, **kwargs)
           self.custom_param = custom_param
       
       def _model_call(self, *args, **kwargs):
           # 自訂邏輯
           result = super()._model_call(*args, **kwargs)
           return result
   ```

2. **Fork 方式**：
   - Fork EleutherAI/lm-evaluation-harness 到你的 GitHub
   - Clone 你的 fork
   - 在你的 fork 中開發
   - 可以選擇提交 PR 回上游

3. **Monkey Patch**（不推薦）：
   ```python
   # 在你的程式碼中
   from lm_eval.models import huggingface
   
   original_method = huggingface.HFLM._model_call
   def custom_model_call(self, *args, **kwargs):
       # 自訂邏輯
       return original_method(self, *args, **kwargs)
   
   huggingface.HFLM._model_call = custom_model_call
   ```

### Q3: `git status` 顯示 `modified: lm-evaluation-harness (modified content)` 是什麼意思？

**A**: 這表示：
- 主專案偵測到 `lm-evaluation-harness` 目錄內有 git 變更
- 但由於它不是正式的 submodule，主專案不會追蹤具體改了什麼
- 你剛剛的還原操作已經清除了這些變更

### Q4: 我應該把 `lm-evaluation-harness` 升級到最新版本嗎？

**A**: 看情況：

```bash
# 查看當前版本
cd lm-evaluation-harness
git log -1 --oneline

# 查看有什麼新功能
git fetch origin
git log HEAD..origin/main --oneline

# 如果沒有破壞性變更，可以更新
git pull origin main

# 測試你的程式碼是否還能運行
cd ..
python your_test_script.py

# 如果有問題，回退到舊版本
cd lm-evaluation-harness
git log --oneline  # 找到舊版本的 hash
git checkout <old-commit-hash>
```

**建議**：
- ✅ 定期查看更新日誌
- ✅ 在開發環境測試新版本
- ✅ 如果穩定就更新
- ⚠️ 記錄你使用的版本（在文檔中）

---

## 🔧 實用命令速查表

### 獨立 Git Repo 模式（目前的狀態）

```bash
# 更新
cd lm-evaluation-harness && git pull origin main

# 切換到特定版本
cd lm-evaluation-harness && git checkout v0.4.0

# 查看當前版本
cd lm-evaluation-harness && git describe --tags

# 還原所有修改
cd lm-evaluation-harness && git restore . && git clean -fd

# 查看修改
cd lm-evaluation-harness && git diff

# 查看狀態
cd lm-evaluation-harness && git status
```

### Git Submodule 模式（如果你決定轉換）

```bash
# 初始化和更新
git submodule init
git submodule update

# 或合併為一步
git submodule update --init --recursive

# 更新到最新
git submodule update --remote

# 還原 submodule
git submodule update --force

# 查看 submodule 狀態
git submodule status

# 在 submodule 中執行命令
git submodule foreach 'git status'

# 刪除 submodule
git submodule deinit lm-evaluation-harness
git rm lm-evaluation-harness
rm -rf .git/modules/lm-evaluation-harness
```

---

## 📚 延伸閱讀

- [Git Submodules 官方文檔](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Working with Git Submodules - Atlassian](https://www.atlassian.com/git/tutorials/git-submodule)
- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)

---

## ✅ 最終建議清單

對於你的專案，建議：

1. ✅ **將 `lm-evaluation-harness/` 加入 `.gitignore`**
   ```bash
   echo "lm-evaluation-harness/" >> .gitignore
   ```

2. ✅ **在 README.md 中說明如何設置**
   - Clone lm-evaluation-harness
   - 安裝依賴
   - 建議的版本（optional）

3. ✅ **不要修改 lm-evaluation-harness 的程式碼**
   - 如果需要自訂功能，在你的專案中繼承

4. ✅ **定期檢查更新**
   ```bash
   cd lm-evaluation-harness
   git fetch origin
   git log HEAD..origin/main --oneline
   ```

5. ✅ **記錄你測試過的版本**
   - 在文檔中記錄：「Tested with lm-evaluation-harness v0.4.x」
   - 或記錄 commit hash

這樣可以保持簡單，同時避免 submodule 的複雜性！🎯
