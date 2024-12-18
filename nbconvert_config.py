"""Custom nbconvert config for converting Jupyter notebooks to Markdown.

**Why This File Exists**:

The versioned, git-tracked version of the publication is generated via `make pub`, but
we also manually create a PubPub publication and this is most easily done by first
converting the notebook to markdown.

When converting to markdown, by default, `nbconvert` inlines all HTML output directly
into the Markdown file when converting a Jupyter notebook. This can lead to excessively
large Markdown files, which are slow to load, render, and scroll through.

This config defines enables a custom preprocessor, `ExtractHTMLPreprocessor`, that
modifies the nbconvert conversion process to handle HTML outputs specially when
converting notebooks to Markdown: instead of the HTML cells being inline, they are saved
as HTML files and referenced in iframe tags.

For usage, see the `markdown` rule in `Makefile`.
"""

c = get_config()  # noqa: F821 - get_config() is defined by nbconvert at runtime
c.Exporter.preprocessors = ["nbconvert_preprocessor.ExtractHTMLPreprocessor"]
