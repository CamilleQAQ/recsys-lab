# 开始之前：装好环境，跑起第一个 lab

这篇只干一件事：让你尽快看到第一个 `PASS`。

装依赖的时间取决于网速（torch 本身有几百 MB）；装好之后，跑第一个 lab 只要十几秒。

原理先不用管，跑通之后再回去看 `docs/01-two-tower`。

## 你需要什么

- **Python 3.10 或更高**。本项目实测环境是 3.10.7。
- **会写 PyTorch**：知道 `nn.Module`、`DataLoader`、一次 `loss.backward()` 在干什么就够了。
- **一块普通 CPU**。这个项目从头到尾没用到 GPU，所有数字都是 CPU 上跑出来的。

不需要先读论文，也不需要懂推荐系统。

## 四步跑起来

```bash
# 1. 虚拟环境
python -m venv .venv

# 2. 激活它（按你的 shell 选一行）
#    Windows PowerShell:
.venv\Scripts\Activate.ps1
#    Windows CMD:
.venv\Scripts\activate.bat
#    macOS / Linux:
source .venv/bin/activate

# 3. 装依赖（torch 本身会下几百 MB，取决于网速）
python -m pip install -e .

# 4. 跑第一个 lab
python labs/verify_labs.py lab1-two-tower-from-scratch.ipynb
```

只有运行 Lab 8 或 FAISS 评估时才需额外执行 `python -m pip install -e ".[faiss]"`。

第 4 步会打印 21 行 `PASS`，最后一行是 `全部通过：1 个 lab`。

**实测（首次运行，含 MovieLens 自动下载）**：12.8 秒，跑完 21 项自测。
测试环境：Windows 11 / Python 3.10.7 / torch 2.14.1+cpu / CPU。你的机器可能更快或更慢。

## 数据会自己下载

不需要手动找数据集。第一次跑某个 lab 时，它会自动下自己需要的那一份：

| 数据集 | 下载体积 | 解压后占用 | 缓存在哪 | 谁需要它 |
|---|---:|---:|---|---|
| MovieLens-1M | 约 6 MB | 约 25 MB（`ratings.dat`） | `data/raw/ml-1m/` | lab1 ~ lab8 |
| KuaiRec 2.0 | 约 431 MB | 约 1.03 GB（`big_matrix.csv`） | `data/kuairec/raw/` | lab9 |

- **MovieLens 很快**：实测首次跑 lab1 一共 12.8 秒，下载只占其中一小部分。
- **KuaiRec 要等**：431 MB 压缩包下载完之后，还要读一遍 1.03 GB 的矩阵。数据准备本身实测约 2 分钟（在压缩包已就位的情况下）；首次还要加上 431 MB 的下载时间，取决于网速。所以 lab9 第一次跑，请留出 5 分钟以上。
- 数据只会下载一次，之后走缓存。想重新下载就把 `data/` 里对应目录删掉。
- `data/` 在 `.gitignore` 里，不会被提交。

**全量校验会连 KuaiRec 和 FAISS 一起触发。** 先安装可选的 `.[faiss]`，再运行 `python labs/verify_labs.py`；lab9 第一次还会下载 KuaiRec 约 431 MB。如果只想快速确认基础环境，先跑单个 lab1 就够了。

## 用 Jupyter 打开（可选）

`verify_labs.py` 不需要 Jupyter，它自己会把每个格子跑一遍。但如果你想在 notebook 里逐格点着看，需要自己装一个：

```bash
python -m pip install jupyter
python -m jupyter notebook
```

然后按顺序打开 `labs/` 里的 notebook。**注意顺序**：lab2 会 import lab1 写出的代码，lab5 依赖 lab4，lab6 依赖 lab5 训练出来的 checkpoint。跳着做会在 `import` 那一步报错。

## Colab（未实测）

理论上可以把仓库 clone 到 Colab 跑，但没有在 Colab 上实际验证过。如果要用，大致是：

```python
!git clone <你的仓库地址>
%cd <仓库目录>
!pip install -e .
!pip install faiss-cpu   # Colab 默认没装 faiss，lab8 需要
!python labs/verify_labs.py lab1-two-tower-from-scratch.ipynb
```

两点提醒：`labkit.py` 会在当前目录往上找 `labs/`，所以只要 `%cd` 进了仓库根目录就能工作；KuaiRec 那 431 MB 在 Colab 会话里能下，但会占用不少时间。

**这一段没有实测过**。如果照着跑失败，开个 issue 告诉我，我把它改成经过验证的版本。

## CPU 还是 GPU

用 CPU 就行，这也是本项目所有结果的来源。

- lab1 ~ lab3 的模型很小，GPU 不会带来可感知的加速。
- **lab5 单这一节实测就要 9.6 分钟**——它要训 3 组 SASRec，每组 30 轮。这是正常现象，不是卡死了。

`src/` 里的训练脚本支持 `--device cuda`，但默认是 `auto`（有 GPU 就用）。

**注意 lab 之间的顺序**：lab5 需要 lab4 先生成 `labs/myrec/sasrec_model.py`。单独跑 lab5 会报"请先运行 Lab 4 生成模型的定义"。同理 lab2 需要 lab1、lab6 需要 lab5。

## 卡住了怎么办

这几条都是本项目里真实会遇到的，不是凑数。

**装依赖时报错，特别是 `faiss-cpu`。**
FAISS 是可选依赖；基础安装 `python -m pip install -e .` 不安装它。只有运行 lab8 或 FAISS 评估时才需要 `python -m pip install -e ".[faiss]"`。不安装时 FAISS 专项测试会跳过，其余测试仍可运行。

**`import myrec` 失败 / `ModuleNotFoundError: No module named 'myrec'`。**
这是 lab 之间的依赖没建立。`myrec/` 是前一个 lab 用 `labkit.harvest` 写出来的，得按顺序先跑完前一个：

| 想跑 | 必须先跑 |
|---|---|
| lab2、lab3 | lab1 |
| lab5 | lab4 |
| lab6 | lab5（它要读 lab5 训出来的 checkpoint） |

单独跑会看到类似"请先运行 Lab 4 生成模型的定义"的报错，这是设计如此，不是坏了。

**数据下载失败 / 卡在下载。**
多半是网络问题。可以手动下载后放到对应目录：MovieLens 的 `ml-1m.zip` 放 `data/raw/`，KuaiRec 的 `KuaiRec.zip` 放 `data/kuairec/raw/`，代码会复用已有文件、不再重下。

**没有 NVIDIA GPU，会不会跑不了？**
不会。这个项目不依赖 GPU，`torch.device("cpu")` 就是默认路径。

**跑得特别慢，是不是卡死了？**
先看是不是 lab5 —— 它真的在训练 3 组 SASRec，**单这一节实测就要 9.6 分钟**。全量 9 个 lab 实测 **26.2 分钟**，这是正常值，不是卡住了（测试环境：Windows 11 / Python 3.10.7 / torch 2.14.1+cpu / CPU）。想快点就只跑单个 lab。

**`verify_labs.py` 报 `FAIL` 而不是 `PASS`。**
说明你的环境算出来的数和期望值不一致。常见原因是 torch 版本差异导致的浮点偏差；先用 `--debug-warnings` 看是哪一格，再用 `--show-output` 看实际数字。如果差得很小（1e-6 量级），是版本差异；差得大，多半是代码被改过。

**`pytest` 报错说找不到 `recsys`。**
`pyproject.toml` 里配了 `pythonpath = ["src"]`，正常情况下不需要额外设置。如果你在别的目录跑 pytest，先 `cd` 到仓库根目录。

## 下一步

- 想直接开始做：`labs/lab1-two-tower-from-scratch.ipynb`
- 想先看懂为什么：`docs/01-two-tower/01-data-and-shared-space.md`
- 查某个英文标识符是什么意思：`docs/术语表.md`
- 想知道整个仓库怎么组织：根目录 `README.md`
