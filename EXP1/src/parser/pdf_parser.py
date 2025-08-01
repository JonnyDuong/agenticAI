import fitz  # PyMuPDF
from pathlib import Path


class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.suffix.lower() == ".pdf":
            raise ValueError("Only .pdf files are supported")

    def extract_text(self) -> str:
        """Extracts full text from all pages using a more reliable block-based method."""
        full_text = []

        with fitz.open(str(self.file_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("blocks")
                page_text_sorted = sorted(page_text, key=lambda b: (b[1], b[0]))  # sort by y, then x
                content = "\n".join([b[4] for b in page_text_sorted if b[4].strip()])
                full_text.append(f"\n--- Page {page_num} ---\n{content}")

        return "\n".join(full_text)

    def extract_markdown(self) -> str:
        """Extracts and returns text as markdown (experimental formatting)."""
        markdown_output = []

        with fitz.open(str(self.file_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                markdown = page.get_text("markdown")
                markdown_output.append(f"\n## Page {page_num}\n{markdown}")

        return "\n".join(markdown_output)
