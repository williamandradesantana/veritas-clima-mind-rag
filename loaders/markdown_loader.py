from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def to_text_list(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        headers_to_split_on = [("#", "Título"), ("##", "Seção"), ("###", "Subseção")]

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs = splitter.split_text(markdown_text)

        return [doc.page_content for doc in docs]
