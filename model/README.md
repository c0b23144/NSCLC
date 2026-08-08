# NSCLC
### 使用ファイル
- nsclc.ipynbで実験
  - nsclc2.ipynbに変更(色々書き加えて見づらいため)
- TransAttUnet使用

### 使用データセット
- NSCLC-Radiomics
  - 422患者のCT画像とmask画像
### overlay.py
CT画像とmaskを重ねて表示する。  
CTとmaskが対応しているかの確認をしている

### dicomNif
CT画像(dcmファイル)をNIfTIに変換

### dcmmaskNif.py
mask画像をそれぞれのindexごとに分けてNifTIにする
