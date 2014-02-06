"""
:Module: ica_reporter
:Synopsis: Report generation after ICA
:Author: dohmatob elvis dopgima

"""

import os
import sys
import base_reporter

# root dir
PYPREPROCESS_DIR = os.path.dirname(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

import external.pytries.trie as trie

# def is_img(trie):
#     return trie.label.endswith(".png") and "outline" in trie.label \
#         and not "axial" in trie.label


# def trie2html(trie):
#     children = trie.children
#     images = [child for child in children if child.is_leaf(
#             ) and is_img(child)]
#     children = [child for child in children if not is_img(child)
#                 and not child.label.endswith(".png")]

#     html = str(trie.label)

#     if images:
#         thumbnails = []
#         for image in images:
#             thumbnails.append(base_reporter.Thumbnail(
#                     a=base_reporter.a(href='https://github.com/dohmatob'),
#                     img=base_reporter.img(src=image._full_label_as_str(),
#                                           height="250px"
#                                           ),
#                     description=image.label
#                     ))

#         gallery_markup = base_reporter.get_gallery_html_markup().substitute(
#             thumbnails=thumbnails)

#         html += "<br>%s<br clear='all'>" % gallery_markup

#     html += "<ul>" + "".join(["<li>%s</li>" % trie2html(child) + "</ul>"
#                               for child in children])

#     return html


def generate_hierarchical_report(
    root_dir,
    report_filename,
    report_title,
    depth=None
    ):
    """
    Generates hierarchical report.

    """

    # def handle_filename(x):

    #     y = trie._handle_filename(x)
    #     if not y is None:
    #         return y

    #     for ext in ['.png']:
    #         if x.endswith(ext) and "z_map" in x:
    #             thumbnail = base_reporter.Thumbnail(
    #                 a=base_reporter.a(href='#'),
    #                 img=base_reporter.img(src=x,
    #                                       height="150px"
    #                                       ),
    #                 description=""
    #                 )

    #             return os.path.basename(x) + "<br clear='all'/>" + \
    #                 base_reporter.get_thumbnail_html_markup(
    #                 ).substitute(thumbnail=thumbnail
    #                              ) + "<br clear='all'/>"
    #         else:
    #             return -1

    output_dir = os.path.dirname(report_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # copy css and js stuff to output dir
    base_reporter.copy_web_conf_files(output_dir)

    html_markup = base_reporter.get_hirarchical_html_template(
        ).substitute(
        title=report_title,
        body=trie.linux_tree(root_dir, depth=depth,
                             ignore_if_endswith=['.html', ".pyc", "#", "~"],
                             # handle_filename=handle_filename,
                             display=False).as_html()
        )

    with open(report_filename, 'w') as fd:
        fd.write(str(html_markup))
        fd.close()

    print html_markup

if __name__ == '__main__':
    generate_hierarchical_report(
        ".",
        "/tmp/hierarchical/report.html",
        "Hierarchical report demo")
