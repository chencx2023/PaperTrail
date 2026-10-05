"""T10: locate saved original text in a fixed PDF without guessing pages."""

from io import BytesIO
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.app.evidence import pdf as evidence_pdf


def pdf_with_pages(*texts: str) -> bytes:
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class PageLocatorTest(unittest.TestCase):
    def test_unique_full_text_match_reports_physical_page_span(self):
        first = "A fixed benchmark contains carefully reviewed questions and stable browser interaction tasks"
        second = "The evaluation reports agent results and discusses limits of the collected evidence"
        pdf = pdf_with_pages(first, second)
        locator = getattr(evidence_pdf, "locate_original_text_pages", None)
        self.assertIsNotNone(locator)

        location = locator(pdf, f"{first}\n{second}")
        self.assertEqual(location["status"], "unique")
        self.assertEqual(location["matches"], [{"start_page": 1, "end_page": 2}])

    def test_repeated_full_text_preserves_all_pdf_occurrences(self):
        passage = "The same detailed description of this benchmark appears in two separate parts of the saved paper"
        pdf = pdf_with_pages(passage, "An unrelated middle page", passage)
        locator = getattr(evidence_pdf, "locate_original_text_pages", None)
        self.assertIsNotNone(locator)

        location = locator(pdf, passage)
        self.assertEqual(location["status"], "multiple")
        self.assertEqual(location["matches"], [
            {"start_page": 1, "end_page": 1},
            {"start_page": 3, "end_page": 3},
        ])

    def test_changed_original_text_has_unknown_page(self):
        passage = "The benchmark contains carefully reviewed questions and stable browser interaction tasks"
        pdf = pdf_with_pages(passage)
        locator = getattr(evidence_pdf, "locate_original_text_pages", None)
        self.assertIsNotNone(locator)

        location = locator(pdf, passage.replace("stable", "changed"))
        self.assertEqual(location["status"], "unknown")
        self.assertEqual(location["matches"], [])


if __name__ == "__main__":
    unittest.main()
