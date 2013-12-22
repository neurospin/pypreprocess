import os
import nibabel
import numpy as np

DATA_DIR = "test_tmp_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def create_random_image(shape=None,
                        ndim=3,
                        n_scans=None,
                        affine=np.eye(4),
                        parent_class=nibabel.Nifti1Image):
    """
    Creates a random image of prescribed shape

    """

    rng = np.random.RandomState(0)

    if shape is None:
        shape = np.random.random_integers(20, size=ndim)

    ndim = len(shape)

    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans

    return parent_class(np.random.randn(*shape), affine)


def make_dataset(n_subjects=1, n_scans=10, n_sessions=1,
                 threeD_filenames=False, dataset_name="test_dataset",
                 output_dir=DATA_DIR, ext="nii.gz"):

    output_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = []
    for i in xrange(n_subjects):
        subject_data = {"subject_id": "sub%03i" % (i + 1), "func": [],
                        'anat': '%s/anat.nii.gz' % DATA_DIR}
        nibabel.save(create_random_image(ndim=3),
                     subject_data['anat'])
        subject_data_dir = os.path.join(output_dir, subject_data["subject_id"])
        if not os.path.exists(subject_data_dir):
            os.makedirs(subject_data_dir)

        for j in xrange(n_sessions):
            session_dir = os.path.join(subject_data_dir,
                                       "session%03i" % (j + 1))
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            sfunc = []
            if threeD_filenames:
                for k in xrange(n_scans):
                    func_filename = os.path.join(session_dir,
                                                 "func%03i.%s" % (j + 1, ext))
                    nibabel.save(create_random_image(ndim=3),
                                 func_filename)
                    sfunc.append(func_filename)
                subject_data['func'].append(sfunc)
            else:
                func_filename = os.path.join(session_dir, "func.%s" % ext)
                nibabel.save(create_random_image(ndim=4, n_scans=n_scans),
                             func_filename)
                subject_data['func'].append(func_filename)

        dataset.append(subject_data)

    return dataset
