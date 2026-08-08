import os
import glob
import pydicom
import pydicom_seg
import SimpleITK as sitk

# --- 設定（ここを必ず確認してください） ---
DICOM_ROOT = r"E:\manifest-1603198545583\NSCLC-Radiomics"
IMAGES_TR  = r"E:\NSCLC_NIfTI\imagesTr"
LABELS_TR  = r"E:\NSCLC_NIfTI2\labelsTr"

os.makedirs(LABELS_TR, exist_ok=True)

# すでに変換済みのCT（xLUNG1-xxx.nii.gz）を基準にする
ct_files = sorted(glob.glob(os.path.join(IMAGES_TR, "LUNG1-*.nii.gz")))

if not ct_files:
    print(f"❌ エラー: {IMAGES_TR} の中にCTファイルが見つかりません。パスを確認してください。")
    exit()

print(f"🚀 {len(ct_files)}件の処理を開始します...")

for ct_path in ct_files:
    patient_id = os.path.basename(ct_path).replace("x", "").replace(".nii.gz", "")
    print(f"📦 {patient_id} を処理中...", end=" ", flush=True)

    # 1. 基準となるCTを読み込む
    ref_ct = sitk.ReadImage(ct_path)

    # 2. この患者の全DICOMファイルを検索し、Segmentationファイルを探す
    patient_folder = os.path.join(DICOM_ROOT, patient_id)
    seg_file = None
    
    # フォルダ内を再帰的に全探索
    for root, dirs, files in os.walk(patient_folder):
        for f in files:
            if f.endswith(".dcm"):
                full_path = os.path.join(root, f)
                try:
                    # DICOMの中身をチラ見して、Segmentationデータかどうか確認
                    ds = pydicom.dcmread(full_path, stop_before_pixels=True)
                    if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4': # Segmentation Storage UID
                        seg_file = full_path
                        break
                except:
                    continue
        if seg_file: break

    if not seg_file:
        print("⚠️  DICOM-SEGが見つかりませんでした。")
        continue

    try:
        # 3. 変換実行
        dcm = pydicom.dcmread(seg_file)
        reader = pydicom_seg.SegmentReader()
        result = reader.read(dcm)
        
        # 最初のセグメント（腫瘍）を抽出
        segment_id = list(result.available_segments)[0]
        mask_itk = result.segment_image(segment_id) # この時点ではまだ枚数が違う可能性がある

        # 4. 【重要】CTと座標・枚数を完全に一致させる「再サンプリング」
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_ct)             # CT（135枚）を基準にする
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # ラベルなので0か1で補間
        resampler.SetTransform(sitk.Transform())        # 位置合わせのみ
        
        # CTの箱の中に、ラベルの情報を流し込む
        fixed_mask = resampler.Execute(mask_itk)
        
        # 軸の情報を完全に一致させる（念押し）
        fixed_mask.CopyInformation(ref_ct)

        # 5. 保存
        output_path = os.path.join(LABELS_TR, f"{patient_id}.nii.gz")
        sitk.WriteImage(fixed_mask, output_path) # fixed_maskを保存
        print(f"✅ 保存完了: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"❌ エラー発生: {e}")

print("\n✨ すべての処理が終了しました。")
