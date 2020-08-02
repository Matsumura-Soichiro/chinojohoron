# deep graph unets  

## 実装
deep graph u-netsの実装を行った。使用したオープンソースライブラリは以下の二つ。

[rdkit](https://github.com/rdkit/rdkit)

[pytorch geometric](https://github.com/rusty1s/pytorch_geometric)

## ファイルの説明

1. `gen_data.py`は学習に必要なデータを生成している。化合物の文字列表記であるSMILESのデータさえあれば、データを生成することができる。また、このコードには二つ目の工夫としてあげた。原子番号をセグメンテーションの教師ありデータとして用いることやノード(原子)の特徴量ベクトルを作り出すコードである。


2. `net.py`に今回使用したモデルがまとめられている。`Unpool`クラスが工夫その３で述べたものである。また、`execute.py`で実際の学習を行なっている。


3. `clustering.ipynb`でクラスタリングを行っている。レポートで省略してしまった描画が全てまとめられている。また、訓練データ、テストデータに分けてクラスタリングを行っている点に注意が必要である


