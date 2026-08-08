import pydicom

#dcm_path = r"E:\manifest-1603198545583\NSCLC-Radiomics\LUNG1-083\12-31-2005-StudyID-NA-78482\300.000000-Segmentation-1.880\1-1.dcm"
#dcm_path = r"E:\manifest-1603198545583\NSCLC-Radiomics\LUNG1-004\09-24-2006-StudyID-NA-27873\300.000000-Segmentation-8.760\1-1.dcm"
dcm_path = r"E:\manifest-1603198545583\NSCLC-Radiomics\LUNG1-371\04-12-2010-NA-NA-25607\300.000000-Segmentation-9.158\1-1.dcm"
ds = pydicom.dcmread(dcm_path)

# どんな臓器（セグメント）が入っているか表示
if 'SegmentSequence' in ds:
    for i, segment in enumerate(ds.SegmentSequence):
        label = segment.SegmentLabel if 'SegmentLabel' in segment else "Unknown"
        print(f"Index {i+1}: {label}")