"""See nbconvert_config.py for details."""

import os

from nbconvert.preprocessors import Preprocessor
from traitlets import List


class ExtractHTMLPreprocessor(Preprocessor):
    output_mimetypes = List(
        ["text/html"], help="The list of MIME types to extract and save to files."
    ).tag(config=True)

    def preprocess_cell(self, cell, resources, index):
        output_dir = resources.get("output_files_dir", "notebook_files")
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(cell, "outputs"):
            for output_index, output in enumerate(cell.outputs):
                if output.output_type not in ("display_data", "execute_result"):
                    continue

                for mime in self.output_mimetypes:
                    if mime in output.data:
                        data = output.data[mime]

                        # Generate a unique filename for each HTML output
                        filename = (
                            f'output_{index}_{output_index}_{mime.replace("/", "_")}.html'
                        )
                        filepath = os.path.join(output_dir, filename)

                        # Save the HTML content to a file
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(data)

                        # Replace output in the cell with an iframe referencing the saved file
                        relative_path = os.path.join(output_dir, filename).replace("\\", "/")
                        iframe_html = (
                            f'<iframe src="{relative_path}" width="100%" height="600" '
                            f'frameborder="0" allowfullscreen></iframe>'
                        )

                        # Replace the data in the output
                        output.data = {"text/html": iframe_html}
        return cell, resources
