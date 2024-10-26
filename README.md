### 简要介绍

本项目主要基于 [Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer)、[FR-Head](https://github.com/zhysora/FR-Head) 和 [SiT-MLP](https://github.com/zhshj0110/SiT-MLP)，针对 [UAV-Human](https://github.com/sutdcv/UAV-Human) 数据集进行了一定修改。使用两类模型分别进行不同模态数据的训练，随后在 `TestA` 数据集上使用 `ensemble` 方法优化多模型的组合权重，得到最终预测结果。

带有完整训练日志、预处理数据及训练权重的版本可以在百度网盘中获取：

```
【超级会员V5】通过百度网盘分享的文件：ai挑战队 模型…
链接:https://pan.baidu.com/s/1nCUcWishGpPUDstENiwZAA?pwd=8m7r 
提取码:8m7r
复制这段内容打开「百度网盘APP 即可获取」
```

### 环境配置

- 使用 `conda` 安装基本环境

```bash
conda env create -f environment.yaml
```

- Run `pip install -e torchlight` 

### 数据处理

使用 `gen_modal` 为 `Train` `TestA` `TestB` 生成多模态数据（需要使用 `joint` `bone` 两个模态）。

将处理后的数据分别置于 `Skeleton-MixFormer/data` 和 `FR-Head/data` 中，两者内容完全相同，结构如下：

```
- data/
  - uav/
    - B/
      test_joint.npy
      ... # TestB 数据集的多模态数据（label 文件需要使用 gen_0_label.py 生成，是内容全为 0 的 .npy 文件）
    train_joint.npy
    ...
    test_joint.npy # data/uav 下放置的 Test 数据集是由 TestA 生成的，作为训练过程中评价基准，也用于 ensemble 时的组合权重优化
```

压缩文件中已经附带了处理完毕并正确放置的数据集，您也可以自行处理。

### 模型训练

[Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer)、[FR-Head](https://github.com/zhysora/FR-Head) 和 [SiT-MLP](https://github.com/zhshj0110/SiT-MLP) 这三类模型需要分别训练。

#### Skeleton-MixFormer

- 进入 `Skeleton-MixFormer` 目录

- 运行下面代码

  ```bash
  # 示例：训练 joint 模态
  python main.py --config config/uav/j.yaml --work-dir work_dir/uav/skmixf_j --device 0
  # 示例：训练 bone 模态
  python main.py --config config/uav/b.yaml --work-dir work_dir/uav/skmixf_b --device 0
  # 示例：训练 joint-motion 模态
  python main.py --config config/uav/jm.yaml --work-dir work_dir/uav/skmixf_jm --device 0
  # 示例：训练 bone-motion 模态
  python main.py --config config/uav/bm.yaml --work-dir work_dir/uav/skmixf_bm --device 0
  ```
  
  **注意：**可能需要修改配置文件中的训练参数来适配训练设备的具体情况

#### FR-Head

- 进入 `FR-Head` 目录

- 运行下面代码

  ```bash
  # 示例：训练 joint 模态
  python main.py --config config/uav/j.yaml --work-dir results/uav/j --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0
  # 示例：训练 bone 模态
  python main.py --config config/uav/b.yaml --work-dir results/uav/b --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0
  # 示例：训练 joint-motion 模态
  python main.py --config config/uav/j.yaml --work-dir results/uav/jm --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0
  # 示例：训练 bone-motion 模态
  python main.py --config config/uav/b.yaml --work-dir results/uav/bm --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0
  ```
  
  **注意：**可能需要修改配置文件中的训练参数来适配训练设备的具体情况

#### SiT-MLP

- 进入 `Skeleton-MixFormer` 目录

- 运行下面代码

  ```bash
  # 示例：训练 joint 模态
  python main.py --config config/uav/j.yaml --work-dir work_dir/uav/j --device 0
  # 示例：训练 bone 模态
  python main.py --config config/uav/b.yaml --work-dir work_dir/uav/b --device 0
  # 示例：训练 joint-motion 模态
  python main.py --config config/uav/jm.yaml --work-dir work_dir/uav/jm --device 0
  # 示例：训练 bone-motion 模态
  python main.py --config config/uav/bm.yaml --work-dir work_dir/uav/bm --device 0
  ```

  **注意：**可能需要修改配置文件中的训练参数来适配训练设备的具体情况

在训练过程中会自动输出模型在 `TestA` 上的测试结果。

在压缩文件的 `Skeleton-MixFormer/work_dir/uav/`、`FR-Head/results/uav/` 和 `SiT-MLP/results/uav` 这三个路径下存放有预先训练的结果及各轮的权重。其中，训练日志在各文件夹下的 `log.txt` 文件中。

### 生成预测

出于方便考虑，本项目并没有直接实现多模型的组合推理，而选择了先完成各个模型的推理，再将模型的推理结果组合起来的方案。

[Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer)、[FR-Head](https://github.com/zhysora/FR-Head) 和 [SiT-MLP](https://github.com/zhshj0110/SiT-MLP) 这三类模型生成预测的方法是相同的，示例如下（在`Skeleton-MixFormer` 和 `FR-Head` 目录下执行）：

```bash
python main.py --config <config_dir>/test/<config.yaml> --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

具体来说，对于每个模型，我们选择训练过程中在 `TestA` 上准确度最高的一轮训练结果进行预测。

在压缩文件的 `Skeleton-MixFormer/work_dir/test/`、`FR-Head/results/test/` 和 `SiT-MLP/work_dir/test` 这三个路径下存放有我们的训练的模型生成的原始预测结果。

### `ensemble` 得到最终预测结果

#### 配置模型路径

编辑 `ensemble.py` 文件，在 `if __name__ == "__main__":` 下有如下代码：

```python
label = np.load('./Skeleton-MixFormer/data/uav/test_label.npy')

models = [
    "Skeleton-MixFormer/work_dir/uav/skmixf_b",
    "Skeleton-MixFormer/work_dir/uav/skmixf_j",
    "Skeleton-MixFormer/work_dir/uav/skmixf_bm",
    "Skeleton-MixFormer/work_dir/uav/skmixf_jm",
    "FR-Head/results/uav/b",
    "FR-Head/results/uav/j",
    "FR-Head/results/uav/bm",
    "FR-Head/results/uav/jm",
    "SiT-MLP/work_dir/uav/j",
    "SiT-MLP/work_dir/uav/b",
    "SiT-MLP/work_dir/uav/jm",
    "SiT-MLP/work_dir/uav/bm"
]
tests = [
    "Skeleton-MixFormer/work_dir/test/test_b",
    "Skeleton-MixFormer/work_dir/test/test_j",
    "Skeleton-MixFormer/work_dir/test/test_bm",
    "Skeleton-MixFormer/work_dir/test/test_jm",
    "FR-Head/results/test/test_b",
    "FR-Head/results/test/test_j",
    "FR-Head/results/test/test_bm",
    "FR-Head/results/test/test_jm",
    "SiT-MLP/work_dir/test/test_j",
    "SiT-MLP/work_dir/test/test_b",
    "SiT-MLP/work_dir/test/test_jm",
    "SiT-MLP/work_dir/test/test_bm"
]
```

其中：

- `label`：使用 `np.load` 方法加载 `TestA` 的 `label`
- `models`：训练的模型结果，文件夹中需要包含 `epoch1_test_score.pkl` 文件（训练过程中会自动生成）
- `tests`：模型在 `TestB` 上的推理结果，文件夹中需要包含 `epoch1_test_pred.npy` 文件（推理过程中会自动生成）

`ensemble` 组合对模型的类型及训练模态没有特别的要求，可以将任意多的模型结果进行组合。

运行下面代码启动权重优化及组合预测结果生成（在项目根目录下执行）：

```bash
python ensemble.py > logfile.log
```

在执行过程中会为各个模型尝试配置权重组合，最终会输出优化后的模型权重（使用上面的命令可能不会在终端中回显，请在 `logfile.log` 文件中查看结果），并将组合后的预测结果 `pred.npy` 存放在项目根目录下。
