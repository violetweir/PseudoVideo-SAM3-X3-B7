import numpy as np, json
feat = "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features"
for name in ["sam3_base_s1008_patches.npz", "sam3enc_mask_descriptors_s1008.npz", "sam3_base_s1008_features.npz"]:
    p = feat + "/" + name
    d = np.load(p, mmap_mode="r")
    print(name, "->", {k: (d[k].shape, str(d[k].dtype)) for k in d.files})
# also gt-lesion in lesion_knn dir
import os
p2 = "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/features/sam3enc_gt_lesion_descriptors_s1008.npz"
if os.path.exists(p2):
    d = np.load(p2, mmap_mode="r")
    print(p2, "->", {k: (d[k].shape, str(d[k].dtype)) for k in d.files})
rows = [json.loads(l) for l in open("work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl")]
print("merged n =", len(rows))
r = rows[0]
print("sample record keys:", sorted(r.keys()))
print({k: r.get(k) for k in ("merged_id", "split", "image_path", "mask_path")})
print("ids in patches npz first 3:", np.load(feat + "/sam3_base_s1008_patches.npz", mmap_mode="r")["ids"][:3].tolist() if "ids" in np.load(feat + "/sam3_base_s1008_patches.npz", mmap_mode="r").files else "no ids key")
