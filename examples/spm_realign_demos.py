import os
import sys
import glob
from collections import namedtuple
import matplotlib.pyplot as plt

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

from algorithms.registration.spm_realign import MRIMotionCorrection
from reporting.check_preprocessing import plot_spm_motion_parameters
from external.nisl.datasets import fetch_nyu_rest
from datasets_extras import fetch_fsl_feeds_data

# datastructure for subject data
SubjectData = namedtuple('SubjectData', 'subject_id func output_dir')


def _demo(subjects, dataset_id):
    """Demo runner.

    Parameters
    ----------
    subjects: iterable for subject data
        each subject data can be anything, with a func (string or list
        of strings; existing file path(s)) and an output_dir (string,
        existing dir path) field
    dataset_id: string
        a short string describing the data being processed (e.g. "HAXBY!")

    Notes
    -----
    Don't invoke this directly!

    """

    # loop over subjects
    for subject_data in subjects:
        print("%sMotion correction for %s (%s)" % ('\t' * 3,
                                                   subject_data.subject_id,
                                                   dataset_id))

        # instantiate realigner
        mc = MRIMotionCorrection(fwhm=5., interp=2)

        # fit realigner
        mc.fit(subject_data.func)

        # write realigned files to disk
        mc.transform(output_dir=subject_data.output_dir)

        # plot results
        rp_filename = glob.glob(os.path.join(subject_data.output_dir,
                                             "rp_*.txt"))[0]
        plot_spm_motion_parameters(
            rp_filename,
            title="Estimated motion for %s of '%s'" % (
                subject_data.subject_id,
                dataset_id))
        plt.show()


def demo_nyu_rest(data_dir="/tmp/nyu_data",
                  output_dir="/tmp/nyu_mrimc_output",
                  ):
    """Demo for FSL Feeds data.

    Parameters
    ----------
    data_dir: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    output_dir: string, optional
        where output will be written to

    """

    # fetch data
    nyu_data = fetch_nyu_rest(data_dir=data_dir)

    # subject data factory
    def subject_factory(session=1):
        session_func = [x for x in nyu_data.func if "session%i" % session in x]

        for subject_id in set([os.path.basename(
                    os.path.dirname
                    (os.path.dirname(x)))
                               for x in session_func]):
            # set func
            func = [
                x for x in session_func if subject_id in x]
            assert len(func) == 1
            func = func[0]

            yield SubjectData(subject_id=subject_id, func=func,
                              output_dir=os.path.join(
                    output_dir,
                    "session%i" % session, subject_id))

    # invoke demon to run de demo
    _demo(subject_factory(), "NYU Resting State")


def demo_fsl_feeds(data_dir="/tmp/fsl_feeds_data",
                  output_dir="/tmp/fsl_feeds_mrimc_output",
                  ):
    """Demo for FSL Feeds data.

    Parameters
    ----------
    data_dir: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    output_dir: string, optional
        where output will be written to

    """

    # fetch data
    fsl_feeds_data = fetch_fsl_feeds_data(data_dir=data_dir)

    # subject data factory
    def subject_factory(session=1):
            subject_id = "sub001"

            yield SubjectData(subject_id=subject_id,
                              func=fsl_feeds_data['func'] + ".gz",
                              output_dir=os.path.join(output_dir, subject_id))

    # invoke demon to run de demo
    _demo(subject_factory(), "FSL FEEDS")


# main
if __name__ == '__main__':
    warning = ("%s: THIS SCRIPT MUST BE RUN FROM ITS PARENT "
               "DIRECTORY!") % sys.argv[0]
    banner = "#" * len(warning)
    separator = "\r\n\t"

    print separator.join(['', banner, warning, banner, ''])

    # run FSL FEEDS demo
    demo_fsl_feeds("/home/elvis/CODE/datasets/fsl-feeds-data")
