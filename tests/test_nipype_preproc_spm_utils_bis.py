import nose
import nose.tools
import os
import nibabel
from ._test_utils import create_random_image
from ..pypreprocess.nipype_preproc_spm_utils_bis import (niigz2nii,
                                                         SubjectData,
                                                         do_subject_preproc
                                                         )


def test_niigz2nii_with_filename():
    # create and save .nii.gz image
    img = create_random_image()
    ifilename = '/tmp/toto.nii.gz'
    nibabel.save(img, ifilename)

    # convert img to .nii
    ofilename = niigz2nii(ifilename, output_dir='/tmp/titi')

    # checks
    nose.tools.assert_equal(ofilename, '/tmp/titi/toto.nii')
    nibabel.load(ofilename)


def test_niigz2nii_with_list_of_filenames():
    # creates and save .nii.gz image
    ifilenames = []
    for i in xrange(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii.gz' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii
    ofilenames = niigz2nii(ifilenames, output_dir='/tmp/titi')

    # checks
    nose.tools.assert_equal(len(ifilenames), len(ofilenames))
    for x in xrange(len(ifilenames)):
        nibabel.load(ofilenames[x])


def test_niigz2nii_with_list_of_lists_of_filenames():
    # creates and save .nii.gz image
    ifilenames = []
    for i in xrange(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii.gz' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii
    ofilenames = niigz2nii([ifilenames], output_dir='/tmp/titi')

    # checks
    nose.tools.assert_equal(1, len(ofilenames))
    for x in xrange(len(ofilenames[0])):
        nibabel.load(ofilenames[0][x])


def test_subject_data():
    # create subject data
    sd = SubjectData()
    sd.subject_id = 'sub001'
    sd.output_dir = os.path.join("/tmp/kimbo/", sd.subject_id)
    sd.func = '/tmp/func.nii.gz'
    nibabel.save(create_random_image(ndim=4), sd.func)
    sd.anat = '/tmp/anat.nii.gz'
    nibabel.save(create_random_image(), sd.anat)

    # sanitize subject data
    sd.sanitize()

    # checks
    nose.tools.assert_equal(sd.func, ['/tmp/kimbo/sub001/func.nii'])
    nose.tools.assert_equal(sd.anat, '/tmp/kimbo/sub001/anat.nii')
    nose.tools.assert_equal(sd.session_id, ['session_0'])


# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
