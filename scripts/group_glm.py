import os
import glob
import json
from pypreprocess.reporting.glm_reporter import group_one_sample_t_test
import nibabel

if __name__ == "__main__":
    masks = []
    effects_maps = []
    conditions = ["LH-RH", "RH-LH", "LF-RF", "RF-LF", "T-AVG"]
    anat = None
    for subject_dir in sorted(glob.glob("/media/Seagate/HCP_clean_preproc/*")):
        mask = os.path.join(subject_dir, "wmask.nii")
        if not os.path.isfile(mask):
            continue

        masks.append(mask)

        skip = False
        subject_effects_maps = {}
        for condition in conditions:
            eff_map = os.path.join(
                subject_dir, "effects_maps/weffects_%s.nii" % condition)
            if not os.path.isfile(eff_map):
                skip = True
                break

            subject_effects_maps[condition] = eff_map

        if skip:
            continue

        effects_maps.append(subject_effects_maps)

        if anat is None:
            tmp = os.path.join(subject_dir, "wT1w_acpc_dc_restore_brain.nii")
            if os.path.isfile(tmp):
                anat = tmp

    contrasts = json.load(open("/tmp/contrasts.json"))
    contrasts = dict((k, v) for k, v in contrasts.iteritems()
                     if k in conditions)

    group_one_sample_t_test(
        masks, effects_maps, contrasts, "/tmp", slicer="ortho",
        threshold=3., anat=nibabel.load(anat).get_data() if anat else None,
        anat_affine=nibabel.load(anat).get_affine() if anat else None,
        title="HCP group GLM (MOTOR protocol, %i subjects)" % len(masks)
        )
