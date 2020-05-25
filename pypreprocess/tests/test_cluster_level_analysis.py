""" Test the cluster level thresholding utilities
"""
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import nibabel as nib
from ..cluster_level_analysis import (empirical_p_value, fdr_threshold,
                                      fdr_p_values, cluster_stats)


def test_empirical_p_value():
    ref = np.arange(100)
    np.random.shuffle(ref)
    z_score = np.array([-1, 18.6, 98.9, 100])
    pvals_ = empirical_p_value(z_score, ref)
    pvals = np.array([1., .81, .01, .0])
    assert_array_almost_equal(pvals, pvals_)


def test_fdr():
    n = 100
    x = np.linspace(.5 / n, 1. - .5 / n, n)
    x[:10] = .0005
    x = norm.isf(x)
    np.random.shuffle(x)
    assert_almost_equal(fdr_threshold(x, .1), norm.isf(.0005))
    assert fdr_threshold(x, .001) == np.infty


def test_fdr_p_values():
    n = 100
    x = np.linspace(.5 / n, 1. - .5 / n, n)
    x[:10] = .0005
    x = norm.isf(x)
    fdr = fdr_p_values(x)
    assert_array_almost_equal(fdr[:10], .005)
    assert np.all(fdr[10:] > .95)
    assert fdr.max() <= 1


def test_cluster_stats():
    shape = (9, 10, 11)
    data = np.random.randn(*shape)
    threshold = norm.sf(data.max() + 1)
    data[2:4, 5:7, 6:8] = np.maximum(10, data.max() + 2)
    stat_img = nib.Nifti1Image(data, np.eye(4))
    mask_img = nib.Nifti1Image(np.ones(shape), np.eye(4))

    # test 1
    clusters, _ = cluster_stats(
        stat_img, mask_img, threshold, height_control='fpr',
        cluster_th=0)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster['size'] == 8
    assert_array_almost_equal(cluster['z_score'], 10 * np.ones(8))
    assert cluster['maxima'].shape == (8, 3)

    # test 2:excessive size threshold
    clusters, _ = cluster_stats(
        stat_img, mask_img, threshold, height_control='fpr',
        cluster_th=10)
    assert clusters == []

    # test 3: excessive cluster forming threshold
    clusters, _ = cluster_stats(
        stat_img, mask_img, 100, height_control='fpr',
        cluster_th=0)
    assert clusters == []

    # test 4: fdr threshold
    clusters, _ = cluster_stats(
        stat_img, mask_img, .05, height_control='fdr',
        cluster_th=5)
    assert len(clusters) == 1
    cluster_ = clusters[0]
    assert_array_almost_equal(cluster['maxima'], cluster_['maxima'])

    # test 5: fdr threshold
    clusters, _ = cluster_stats(
        stat_img, mask_img, .05, height_control='bonferroni',
        cluster_th=5)
    assert len(clusters) == 1
    cluster_ = clusters[0]
    assert_array_almost_equal(cluster['maxima'], cluster_['maxima'])

    # test 5: direct threshold
    clusters, _ = cluster_stats(
        stat_img, mask_img, 5., height_control=None,
        cluster_th=5)
    assert len(clusters) == 1
    cluster_ = clusters[0]
    assert_array_almost_equal(cluster['maxima'], cluster_['maxima'])


def test_multi_cluster_stats():
    shape = (9, 10, 11)
    data = np.random.randn(*shape)
    threshold = norm.sf(data.max() + 1)
    data[2:4, 5:7, 6:8] = np.maximum(10, data.max() + 2)
    data[6:7, 8:9, 9:10] = np.maximum(11, data.max() + 1)
    stat_img = nib.Nifti1Image(data, np.eye(4))
    mask_img = nib.Nifti1Image(np.ones(shape), np.eye(4))

    # test 1
    clusters, _ = cluster_stats(
        stat_img, mask_img, threshold, height_control='fpr',
        cluster_th=0)
    assert len(clusters) == 2
    cluster = clusters[1]
    assert cluster['size'] == 1
    assert_array_almost_equal(cluster['z_score'], 11)
    assert_array_almost_equal(cluster['maxima'], np.array([[6, 8, 9]]))
