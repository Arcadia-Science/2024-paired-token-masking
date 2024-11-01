Additional Code
===============

This section contains documentation of all the supplementary code used in the analysis but not directly implemented in the publicized notebook.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}
