from langchain_docling.loader import DoclingLoader
from langchain_community.document_loaders import PyMuPDFLoader
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

class PDFReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_pdf(self):
        # Configure the pipeline to ignore the broken text layer
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.allow_external_plugins = True 
        pipeline_options.ocr_options.force_full_page_ocr = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        print(f"Loading PDF from: {self.file_path}")
        loader = DoclingLoader(
            file_path=self.file_path, 
            converter=converter
        )
        data = loader.load()
        print("PDF loaded successfully")
        return data