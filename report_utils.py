def sidebyside(report, img1, img2):
    """
    Inserts two columns of images in a n html report

    """
    if not type(img1) is list:
        img1 = [img1]
    if not type(img2) is list:
        img2 = [img2]

    if min(len(img1), len(img2)) > 0:
        report.center()
        report.table(border="0", width="100%", cellspacing="1")
        report.tr()
        report.td()
        report.img(src=img1, width="100%")
        report.td.close()
        report.td()
        report.img(src=img2, width="100%")
        report.td.close()
        report.tr.close()
        report.table.close()
        report.center.close()
    else:
        report.img(src=img1 + img2, width="50%")
