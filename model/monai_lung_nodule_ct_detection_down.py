from monai.bundle import download

from monai.bundle import download

# モデル名の指定
model_name = "lung_nodule_ct_detection"

# 保存先フォルダ
output_dir = "./lung_nodule_model"

# 新しいAPIでのダウンロード実行
download(
    name=model_name,
    bundle_dir=output_dir,
    progress=True
)

print(f"ダウンロード完了！ 保存先: {output_dir}")