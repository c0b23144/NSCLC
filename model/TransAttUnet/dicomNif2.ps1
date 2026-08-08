# --- 設定エリア ---
$DICOM_ROOT = "E:\manifest-1603198545583\NSCLC-Radiomics"
$IMAGES_TR  = "E:\NSCLC_NIfTI2\imagesTr"
$LABELS_TR  = "E:\NSCLC_NIfTI2\labelsTr" # 今回は作成のみ

# フォルダがなければ作成
if (!(Test-Path $IMAGES_TR)) { New-Item -ItemType Directory -Path $IMAGES_TR }
if (!(Test-Path $LABELS_TR)) { New-Item -ItemType Directory -Path $LABELS_TR }

Write-Host "🏁 CT画像の一括抽出を開始します..." -ForegroundColor Cyan

$folders = Get-ChildItem -Path $DICOM_ROOT -Filter "LUNG1-*"

foreach ($f in $folders) {
    $patient_id = $f.Name
    
    # 患者フォルダの中にある、実際にDICOMが入っている一番深いフォルダを探す
    # ファイル数が100枚以上あるフォルダをCT画像とみなす
    $target_dir = Get-ChildItem -Path $f.FullName -Recurse | Where-Object { $_.Attributes -eq 'Directory' } | ForEach-Object {
        $fileCount = (Get-ChildItem -Path $_.FullName -Filter *.dcm).Count
        if ($fileCount -gt 50) { $_ }
    } | Select-Object -First 1

    if ($target_dir) {
        Write-Host "🔄 変換中: $patient_id (Folder: $($target_dir.Name))" -ForegroundColor Yellow
        # dcm2niix 実行
        # -f に _0000 をつけるのは nnU-Net 形式に合わせるため（任意）
        # オプションを「文字列のリスト」として定義するとミスが減ります
        $dcm2niix_path = ".\dcm2niix.exe"
        # 強制合体（-i y）を追加した最強のパラメータ設定
        # 出力名をより確実に固定する
        $output_name = $patient_id.Trim()

        # パラメータの渡し方を「配列」ではなく「直接」書いてみる（これだけで直ることも多い）
        & .\dcm2niix.exe -z y -x y -m y -i y -o "$IMAGES_TR" -f "$output_name" "$($target_dir.FullName)"
        
    } else {
        Write-Warning "⚠️ $patient_id にCT画像フォルダが見つかりませんでした。"
    }
}

Write-Host "✅ imagesTr への変換が完了しました！" -ForegroundColor Green