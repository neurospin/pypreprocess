""" Utilities to describe the result of cluster-level analysis of statistical
maps.

Author: Bertrand Thirion, 2015
"""
import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.stats import norm
from nilearn.image.resampling import coord_transform
from nilearn._utils.niimg_conversions import check_niimg, _check_same_fov


def fdr_threshold(z_vals, alpha):
    """ return the BH fdr for the input z_vals"""
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    pos = p_vals < alpha * np.linspace(
        .5 / n_samples, 1 - .5 / n_samples, n_samples)
    if pos.any():
        return (z_vals_[pos][-1] - 1.e-8)
    else:
        return np.infty


def fdr_p_values(z_vals):
    """ return the fdr p_values for the z-variate"""
    order = np.argsort(- z_vals)
    p_vals = norm.sf(z_vals[order])
    n_samples = len(z_vals)
    fdr = np.minimum(1, p_vals / np.linspace(1. / n_samples, 1., n_samples))
    for i in range(n_samples - 1, 0, -1):
        fdr[i - 1] = min(fdr[i - 1], fdr[i])

    inv_order = np.empty(n_samples, 'int')
    inv_order[order] = np.arange(n_samples)
    return fdr[inv_order]


def empirical_p_value(z_score, ref):
    """ retrun the percentile """
    ranks = np.searchsorted(np.sort(ref), z_score)
    return 1 - ranks * 1. / ref.size


def cluster_stats(stat_img, mask_img, threshold, height_control='fpr',
                  cluster_th=0, nulls=None):
    """
    Return a list of clusters, each cluster being represented by a
    dictionary. Clusters are sorted by descending size order. Within
    each cluster, local maxima are sorted by descending statical value

    Parameters
    ----------
    stat_img: Niimg-like object,
       statsitical image (presumably in z scale)
    mask_img: Niimg-like object,
        mask image
    threshold: float,
        cluster forming threshold (either a p-value or z-scale value)
    height_control: string
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|'none'
    cluster_th: int or float,
        cluster size threshold
    nulls: dictionary,
        statistics of the null distribution

    Notes
    -----
    If there is no cluster, an empty list is returned
    """
    if nulls is None: nulls = {}

    # Masking
    mask_img, stat_img = check_niimg(mask_img), check_niimg(stat_img)
    if not _check_same_fov(mask_img, stat_img):
        raise ValueError('mask_img and stat_img do not have the same fov')
    mask = mask_img.get_data().astype(np.bool)
    affine = mask_img.get_affine()
    stat_map = stat_img.get_data() * mask
    n_voxels = mask.sum()

    # Thresholding
    if height_control == 'fpr':
        z_th = norm.isf(threshold)
    elif height_control == 'fdr':
        z_th = fdr_threshold(stat_map[mask], threshold)
    elif height_control == 'bonferroni':
        z_th = norm.isf(threshold / n_voxels)
    else:  # Brute-force thresholding
        z_th = threshold

    p_th = norm.sf(z_th)
    # General info
    info = {'n_voxels': n_voxels,
            'threshold_z': z_th,
            'threshold_p': p_th,
            'threshold_pcorr': np.minimum(1, p_th * n_voxels)}

    above_th = stat_map > z_th
    above_values = stat_map * above_th
    if (above_th == 0).all():
        return [], info

    # Extract connected components above threshold
    labels, n_labels = label(above_th)

    # Extract the local maxima anove the threshold
    maxima_mask = (above_values ==
                   np.maximum(z_th, maximum_filter(above_values, 3)))
    x, y, z = np.array(np.where(maxima_mask))
    maxima_coords = np.array(coord_transform(x, y, z, affine)).T
    maxima_labels = labels[maxima_mask]
    maxima_values = above_values[maxima_mask]

    # FDR-corrected p-values
    max_fdr_p_values = fdr_p_values(stat_map[mask])[maxima_mask[mask]]

    # Default "nulls"
    if not 'zmax' in nulls:
        nulls['zmax'] = 'bonferroni'
    if not 'smax' in nulls:
        nulls['smax'] = None
    if not 's' in nulls:
        nulls['s'] = None

    # Make list of clusters, each cluster being a dictionary
    clusters = []
    for k in range(n_labels):
        cluster_size = np.sum(labels == k + 1)
        if cluster_size >= cluster_th:

            # get the position of the maxima that belong to that cluster
            in_cluster = maxima_labels == k + 1

            # sort the maxima by decreasing statistical value
            max_vals = maxima_values[in_cluster]
            sorted_ = max_vals.argsort()[::-1]

            # Report significance levels in each cluster
            z_score = max_vals[sorted_]
            p_values = norm.sf(z_score)

            # Voxel-level corrected p-values
            fwer_p_value = None
            if nulls['zmax'] == 'bonferroni':
                fwer_p_value = np.minimum(1, p_values * n_voxels)
            elif isinstance(nulls['zmax'], np.ndarray):
                fwer_p_value = empirical_p_value(
                    clusters['z_score'], nulls['zmax'])

            # Cluster-level p-values (corrected)
            cluster_fwer_p_value = None
            if isinstance(nulls['smax'], np.ndarray):
                cluster_fwer_p_value = empirical_p_value(
                    cluster_size, nulls['smax'])

            # Cluster-level p-values (uncorrected)
            cluster_p_value = None
            if isinstance(nulls['s'], np.ndarray):
                cluster_p_value = empirical_p_value(
                    cluster_size, nulls['s'])

            # write all this into the cluster structure
            clusters.append({
                    'size': cluster_size,
                    'maxima': maxima_coords[in_cluster][sorted_],
                    'z_score': z_score,
                    'fdr_p_value': max_fdr_p_values[in_cluster][sorted_],
                    'p_value': p_values,
                    'fwer_p_value': fwer_p_value,
                    'cluster_fwer_p_value': cluster_fwer_p_value,
                    'cluster_p_value': cluster_p_value
                    })

    # Sort clusters by descending size order
    order = np.argsort(- np.array([cluster['size'] for cluster in clusters]))
    clusters = [clusters[i] for i in order]

    return clusters, info
