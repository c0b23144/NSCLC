# NSCLC
### 使用ファイル
- nsclc.ipynbで実験
  - nsclc2.ipynbに変更(色々書き加えて見づらいため)
- TransAttUnet使用

### 使用データセット
- NSCLC-Radiomics
  - 422患者のCT画像とmask画像
  - trainは各患者から複数のスライスを使用
  - testは最大腫瘍面積1枚のみ

### 学習
学習では以下のファイルを使う
- nsclc2.ipynb
- nsclc_large_train.py
- nsclc_train_fold.py
  - k分割交差検証でつかう

### 評価
- nsclc_size_metrics.py
  - largeとsmallの比較
  - サイズ別bin評価評価
  - 大きい腫瘍と小さい腫瘍の評価のグラフ
- nsclc_size_metrics2.py
  - 5種類のモデルを使用
    - small, large, mix, mix + small, mix + large
  - 折れ線にして表示
  - bin3の詳細棒グラフ

### SegRes系のファイルについて
3Dデータのセグメンテーション結果(TransAttUnetとの比較)  
今は使ってない

### TransAttUnet
論文のgithubから引用  
↓↓URL貼ってないです
> [TransAttUnet github]()
- TransAttUnet.py
- unet_parts.py
- unet_parts_att_multiscale.py
  - マルチスケールスキップ接続のファイル？
- unet_parts_att_transformer.py
  - アテンション機構のファイル？

ローカルにあるファイル。後でのせる
### overlay.py
CT画像とmaskを重ねて表示する。  
CTとmaskが対応しているかの確認をしている

### dicomNif
CT画像(dcmファイル)をNIfTIに変換

### dcmmaskNif.py
mask画像をそれぞれのindexごとに分けてNifTIにする
