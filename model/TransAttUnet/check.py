import os
import SimpleITK as sitk
import pandas as pd

img_dir = r"E:\NSCLC_NIfTI2\imagesTr"
lab_dir = r"E:\NSCLC_NIfTI\labelsTr"

results = []

# labelsTrにあるファイルを基準にチェック
for lab_name in os.listdir(lab_dir):
    if not lab_name.endswith(".nii.gz"): continue
    
    img_path = os.path.join(img_dir, lab_name)
    lab_path = os.path.join(lab_dir, lab_name)
    
    if os.path.exists(img_path):
        img = sitk.ReadImage(img_path)
        lab = sitk.ReadImage(lab_path)
        
        img_size = img.GetSize()
        lab_size = lab.GetSize()
        
        match = (img_size == lab_size)
        results.append({
            "Patient": lab_name,
            "Img_Size": img_size,
            "Lab_Size": lab_size,
            "Match": "✅ OK" if match else "❌ NG"
        })

# 結果を表示
df = pd.DataFrame(results)
print(df)

# 不一致のものだけ表示
errors = df[df["Match"] == "❌ NG"]
if not errors.empty:
    print("\n⚠️ サイズが合っていない症例:")
    print(errors)
else:
    print("\n✨ すべての症例のサイズが一致しました！")