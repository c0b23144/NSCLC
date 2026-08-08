import os
import SimpleITK as sitk

root_dir = r"E:\manifest-1603198545583\NSCLC-Radiomics"

for root, dirs, files in os.walk(root_dir):

    dcm_files = [
        f for f in files
        if f.endswith(".dcm")
    ]

    if len(dcm_files) > 0:
        print("\nFolder: ")
        print(root)

        print("Number of DICOMs:")
        print(len(dcm_files))

# 出力先
output_dir = r"E:\NSCLC_NIfTI\imagesTr"

# ディレクトリの存在チェックと作成
os.makedirs(output_dir, exist_ok=True)

# 患者一覧
patients = os.listdir(root_dir)

for patient in patients:
    patient_path = os.path.join(root_dir, patient)

    # フォルダ以外を除外
    if not os.path.isdir(patient_path):
        continue

    print(f"\n===== {patient} =====")

    ct_dir = None

    # 患者内部を全部探索
    for root, dirs, files in os.walk(patient_path):

        # Segmentation を除外
        if "Segmentation" in root:
            continue

        dcm_files = [
            f for f in files
            if f.lower().endswith(".dcm")
        ]

        # CT画像はdicom画像がいっぱい
        if len(dcm_files) > 50:

            ct_dir = root

            print("CT found:")
            print(ct_dir)

            break

    if ct_dir is None:

        print("CT not found")
        continue

    try:
        reader = sitk.ImageSeriesReader()
        # DICOMデータセットとシリーズIDを使用してファイル名のシーケンスを生成
        dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        output_path = os.path.join(
            output_dir,
            f"{patient}.nii.gz"
        )

        sitk.WriteImage(image, output_path)

        print("SAVED: ")
        print(output_path)
    
    except Exception as e:
        print("ERROR:")
        print(e)

print("\nFINISHED")