# Byzantine attacks and defenses in federated learning Library

This library contains the implementation of the Byzantine attacks and defenses in federated learning.
original github: https://github.com/CRYPTO-KU/FL-Byzantine-Library.git

`parameters.py` 안에 `buck_rand`를 직접 정의해야했었음

`python main.py --num_client 10 --trials 1 --gpu_id -1 --attack alie --aggr avg` 를 실험으로 돌려봤었는데.. cpu로만해서 느린거지 설정을 잘못해서 느린건지 몰르겠지만 기본 epoch가 100까지인데 4시간 뒤에 18까지 밖에 못가서 결국 강종
<img width="1514" height="868" alt="python main py --num_client 10 --trials 1 --gpu_id -1 --attack alie --aggr avg" src="https://github.com/user-attachments/assets/d885f92d-c1fa-4d03-9ad7-eb4c58bc2f93" />

**Aggregators:** 
- FedAvg(avg): 클라이언트 업데이트의 평균, weight는 데이터셋의 크기에 따라
- Krum(krum): 과반수에 가장 가까운 업데이트를 계산. 가까운 업데이트는 업데이트 사이의 거리를 계산하고 nearest neighbor와의 계산된 거리의 최소합으로 결정
- Bulyan(bulyan): Krum으로 trusted된 업데이트를 선택하고, 좌표계에서 trimmed mean을 계산
- Trimmed mean(tm): 모델  매개변수 차원에 따라 클라이언트의 업데이트를 sort하고 위와 아래 값을 제거(이상치 제거)하고 나머지 평균
- Centered Median(cm): 매개변수에 따라 클라이언트의 중앙값을 가짐 / 업데이트의 거리의 총합을 최소화하는 지점을 찾음
- Centered Clipping(cc): 업데이트 평균이나 중앙에 위치한 ‘τ’ 반경 안에 업데이트가 놓여지도록 함. 악의적인 업데이트가 너무 멀리 위치한 것을 방지
- Sequential Centered Clipping(ccs):  CC를 확장, 좌표계에서 ‘buckets’에 걸쳐 ‘clipping’
- SignSGD(sign): 클라이언트는 그라디언트의 부호만 보내서 서버는 과반수로 aggregate함
- Robust Federated Aggregation RFA (rfa): 업데이트의 기하중앙값 계산
- FL-Trust(fl_trust): validation 데이터셋을 가지고 있음. 클라이언트의 없데이트의 크기는 믿을 수 있는 서버의 그라디언트에 대한 코사인 유사도에 따라 계산
- Gradient Aggregation Sketching GAS (gas): 클라이언트의 업데이트를 ‘sketch’로 압축시키고 Krum이나 Bulyan으로 aggregate

**Attack:**
_데이터셋 공격_
- Label flipping(label_flip): 데이터셋에서 ‘label’을 바꿈
- Bit-flip(bit_flip): ‘weight’ 매개변수를 바꿔서 모델을 손상시킴
_노이즈 공격:_
- Gaussian noise(gaussian):
- Sparse attacks(sparse): 그라디언트의 차원을 일부 손상
- Sparse-optimized(sparse_opt): 차원을 랜덤으로 말고 그라디언트의 크기에 따라 계산해서찾음
_최적화 공격_
- Untargeted C&W(cw): 
- Little is enough(alie): 
- Inner Product Manipulation IPM (ipm): 
**Geometric, 벡터 공간 공격**
- Relocated Orthogonal Perturbation ROP (reloc/rop):
- Min-sum(minsum): 
- Min-max(minmax):

## Aggregators:
- Aggregators can be extended by adding the aggregator in the `aggregators` folder.


- [x] **Bulyan** - The Hidden Vulnerability of Distributed Learning in Byzantium [[ICML 2018]](https://proceedings.mlr.press/v80/mhamdi18a.html)
- [x] **Centered Clipping** - Learning from history for Byzantine robust optimization [[ICML 2021]](http://proceedings.mlr.press/v139/karimireddy21a.html)
- [x] **Centered Median** - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates [[ICML 2018]](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)
- [x] **Krum**  - Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent [[Neurips 2017]](https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)
- [x] **Trimmed Mean** - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates [[ICML 2018]](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)
- [x] **SignSGD** - signSGD with Majority Vote is Communication Efficient and Fault Tolerant [[ICLR 2019]](https://openreview.net/pdf?id=BJxhijAcY7)
- [x] **RFA** - Robust Aggregation for Federated Learning [[IEEE 2022 TSP]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9721118)
- [x] **Sequantial Centered Clipping** -  Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning [[IEEE 2024 TIFS]](https://ieeexplore.ieee.org/document/9636827)
- [x] **FL-Trust** - FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping [[NDSS 2021]](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf)
- [x] **GAS (Krum and Bulyan)** - Byzantine-robust learning on heterogeneous data via gradient splitting} [[ICML 2023]](https://proceedings.mlr.press/v202/liu23d/liu23d.pdf)
- [x] **FedAVG** - [[AISTATS 2016]](http://proceedings.mlr.press/v51/mcmahan16.pdf)


## Byzantine Attacks:
- Attacks can be extended by adding the attack in the `attacks` folder.


- [x] **Label-Flip** - Poisoning Attacks against Support Vector Machines [[ICML 2012]](https://icml.cc/2012/papers/880.pdf)
- [x] **Bit-Flip** - 
- [x] **Gaussian noise** - 
- [x] **Untargeted C&W ()** - Towards evaluating the robustness of neural networks  [[IEEE S&P 2017]](https://ieeexplore.ieee.org/iel7/7957740/7958557/07958570.pdf)
- [x] **Little is enough (ALIE)** - A Little Is Enough: Circumventing Defenses For Distributed Learning [[Neurips]](https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf)
- [x] **Inner product Manipulation (IPM)** - Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation [[UAI 2019]](http://auai.org/uai2019/proceedings/papers/83.pdf)
- [x] **Relocated orthogonal perturbation (ROP)** - Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning [[IEEE 2024 TIFS]](https://ieeexplore.ieee.org/document/9636827)
- [x] **Min-sum** - Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning [[NDSS 2022]] (https://par.nsf.gov/servlets/purl/10286354)
- [x] **Min-max** - Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning [[NDSS 2022]] (https://par.nsf.gov/servlets/purl/10286354)
- [x] **Sparse** - Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning
- [x] **Sparse-Optimized** - Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning


## Datasets:
- [x] **MNIST**
- [x] **CIFAR-10**
- [x] **CIFAR-100**
- [x] **Fashion-MNIST**
- [x] **EMNIST**
- [x] **SVHN**
- [x] **Tiny-ImageNet**

Datasets can be extended by adding the dataset in the `datasets` folder. Any labeled vision classification dataset in https://pytorch.org/vision/main/datasets.html can be used.


### Available data distributions:
- [x] **IID**
- [x] **Non-IID**: 
    - [x] **Dirichlet** lower the alpha, more non-IID the data becomes. value "1" generally realistic for the real FL scenarios.
    - [x] **Sort-and-Partition** Distributes only a few selected classes to each client.

## Models:
- Models can be extended by adding the model in the `models` folder and by modifying the 'nn_classes' accordingly.
- Different Norms and initialization functions are available in 'nn_classes.


### Available models:
- [x] **MLP** Different sizes of MLP models are available for grayscale images.
- [x] **CNN (various sizes)** Different CNN models are available for RGB and grayscale images respectively
- [x] **ResNet** RGB datasets only. Various depts and sizes are available (8-20-9-18).
- [x] **VGG** RGB datasets only. Various depts and sizes are available.
- [x] **MobileNet** RGB datasets only.

## Future models:
- [x] **Visual Transformers** (ViT , DeiT, Swin, Twin, etc.) 


## Installation

You can install the package locally using:

```bash
pip install .
```

## Usage

- [x] **1. Directly with Python (without installing the package)**
From your project directory, run:

```bash
python main.py [arguments]
python main.py --help
python main.py  --trials 1 --num_client 10
```

- [x] **2. As a Command-Line Tool (after installing the package)** 
If you have installed your package using:

After installation, you can use the main script via the command line:

```bash
fl-byzantine --help
fl-byzantine [arguments]
fl-byzantine --trials 1 --num_client 10
```

Or import modules in your Python code:

```python
from fl_byzantine_library import Aggregators, Attacks, Datasets, Models
```

## Citation

If you find this repo useful, please cite our papers.

```
@ARTICLE{ROP,
  author={Ozfatura, Kerem and Ozfatura, Emre and Kupcu, Alptekin and Gunduz, Deniz},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning}, 
  year={2024},
  volume={19},
  number={},
  pages={2010-2022},
  doi={10.1109/TIFS.2023.3345171}}
```

```
@misc{sparseATK,
      title={Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning}, 
      author={Emre Ozfatura and Kerem Ozfatura and Alptekin Kupcu and Deniz Gunduz},
      year={2024},
      eprint={2404.06230},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Kerem Özfatura (aozfatura22@ku.edu.tr)
- Emre Özfatura (m.ozfatura@imperial.ac.uk)
