"""
:Module: preproc_spm_haxby\
:Synopsis: SPM use-case for preprocessing haxby dataset\
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

"""

# standard imports
import glob
import gzip
import os
import re


def unzip_nii_gz(dirname):
    """
    Helper function for extracting .nii.gz to .nii.

    """

    for filename in glob.glob('%s/*.nii.gz' % dirname):
        if not os.path.exists(re.sub("\.gz", "", filename)):
            f_in = gzip.open(filename, 'rb')
            f_out = open(filename[:-3], 'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            # os.remove(filename)


def fetch_haxby_data_offline(haxby_dir, session_ids=["haxby2001"],
                             subject_ids=["subj1", "subj2", "subj3", "subj4"]):
    """
    Helper function for globbing local haxby directory structure.

    """
    assert os.path.exists(haxby_dir)

    print "Pulling HAXBY data from %s .." % haxby_dir
    sessions = dict()
    for session_id in session_ids:
        session_dir = os.path.join(haxby_dir, "%s" % session_id)

        if not os.path.exists(session_dir):
            print "Warning: %s doesn't exist" % session_dir
            continue

        sessions[session_id] = dict()

        for subject_id in subject_ids:
            subject_dir = os.path.join(session_dir, "%s" % subject_id)
            if not os.path.exists(subject_dir):
                print "Warning: %s doesn't exist" % subject_dir
                continue

            # Because SPM doesn't understand .gz
            print '\t\tDoing .nii.gz -> .nii extraction in %s' % subject_dir
            unzip_nii_gz(subject_dir)
            print '\t\tDone.'

            try:
                anat_image = glob.glob("%s/anat.nii" % subject_dir)[0]
                fmri_image = glob.glob("%s/bold.nii" % subject_dir)[0]
            except IndexError:
                continue

            sessions[session_id][subject_id] = dict()
            sessions[session_id][subject_id]['anat'] = anat_image
            sessions[session_id][subject_id]['func'] = fmri_image

        if len(sessions[session_id]) == 0:
            del sessions[session_id]

    print "Done. Pulled data for %d sesssion(s)." % len(sessions)
    return sessions


def fetch_nyu_data_offline(nyu_rest_dir, session_ids=["session1"],
                           subject_ids=None):
    """
    Helper function for globbing local NYU directory structure.

    """
    assert os.path.exists(nyu_rest_dir)

    if subject_ids is None:
        subject_ids = [os.path.basename(x) for x in glob.glob(
                os.path.join(nyu_rest_dir,
                             os.path.join(session_ids[0], "sub*")))]

    print "Pulling NYU (rest) data from %s .." % nyu_rest_dir
    sessions = dict()
    for session_id in session_ids:
        session_dir = os.path.join(nyu_rest_dir, "%s" % session_id)

        if not os.path.exists(session_dir):
            print "Warning: %s doesn't exist" % session_dir
            continue

        sessions[session_id] = dict()

        for subject_id in subject_ids:
            subject_dir = os.path.join(session_dir, "%s" % subject_id)
            if not os.path.exists(subject_dir):
                print "Warning: %s doesn't exist" % subject_dir
                continue

            # Because SPM doesn't understand .gz
            try:
                unzip_nii_gz(os.path.join(subject_dir, "anat"))
                unzip_nii_gz(os.path.join(subject_dir, "func"))
            except IOError:
                print "Warning: %s contains corrupt .nii.gz archives!"  \
                    % subject_dir
                continue

            try:
                anonymized_image = glob.glob("%s/anat/mprage_anonymized.nii" %\
                                                 subject_dir)[0]
                skullstripped_image = glob.glob(
                    "%s/anat/mprage_skullstripped.nii" % subject_dir)[0]
                fmri_image = glob.glob("%s/func/lfo.nii" % subject_dir)[0]
            except IndexError:
                continue

            sessions[session_id][subject_id] = dict()
            sessions[session_id][subject_id]['skullstripped_anat'] = \
                skullstripped_image
            sessions[session_id][subject_id]['anonymized_anat'] = \
                anonymized_image
            sessions[session_id][subject_id]['func'] = fmri_image

        if len(sessions[session_id]) == 0:
            del sessions[session_id]

    print "Done. Pulled data for %d sesssion(s)" % len(sessions)
    return sessions
