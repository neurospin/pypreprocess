import os
import re
import nibabel
import sys
sys.path.append("..")
import reporting.ica_reporter as ica_reporter

# path of melodic_IC.nii.gz file produced by MELODIC
MELODIC_NII_GZ = "melodic_IC.nii.gz"
if len(sys.argv) > 1:
    MELODIC_NII_GZ = sys.argv[1]

###########
# Reports
###########
output_dir = os.path.dirname(MELODIC_NII_GZ)
mask_filename = os.path.join(output_dir, "mask.nii.gz")

# grab log
methods = "ICA was done by running the following MELODIC command-line:<br/>"
methods += "<i>" + re.search(".*melodic \-i .*",
                             open(
        os.path.join(output_dir,
                     'log.txt')).read()).group() + "</i>"
ica_report_filename = os.path.join(output_dir, "report_stats.html")

# split ICA 4D film into separate 3D vols (one per component)
ica_img = nibabel.load(os.path.join(output_dir, "melodic_IC.nii.gz"))
n_components = ica_img.shape[-1]
ica_maps = {}
for comp in xrange(n_components):
    comp_filename = os.path.join(output_dir, "melodic_IC_%i.nii.gz" % (
            comp + 1))
    nibabel.save(nibabel.Nifti1Image(ica_img.get_data()[..., comp],
                                     ica_img.get_affine()),
                 comp_filename)
    ica_maps[comp + 1] = comp_filename

# generate html report
ica_reporter.generate_ica_report(ica_report_filename,
                                 ica_maps,
                                 mask=mask_filename,
                                 methods=methods,
                                 )

print "Reports written to %s" % ica_report_filename
