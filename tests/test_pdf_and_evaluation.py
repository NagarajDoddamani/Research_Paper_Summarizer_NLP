import fitz
import pytest

from modules.evaluation import calculate_metrics, calculate_rouge
from modules.pdf_extractor import PDFExtractionError, extract_pdf


def make_pdf(text: str, title: str = "Sample Paper") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_metadata_and_page_text():
    result = extract_pdf(make_pdf("A reproducible sample paper with several words."))
    assert result.title == "Sample Paper"
    assert result.page_count == 1
    assert result.word_count >= 7
    assert "sample paper" in result.text


@pytest.mark.parametrize("data", [b"", b"not a pdf"])
def test_invalid_pdf_files(data):
    with pytest.raises(PDFExtractionError):
        extract_pdf(data)


def test_image_only_pdf_reports_missing_text():
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    with pytest.raises(PDFExtractionError, match="OCR is not included"):
        extract_pdf(data)


def test_encrypted_pdf_reports_unlock_message():
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    with pytest.raises(PDFExtractionError, match="encrypted"):
        extract_pdf(data)


def test_metrics_and_rouge_use_supplied_reference():
    metrics = calculate_metrics("one two three four", "one two", 0.25)
    assert metrics.word_count == 2
    assert metrics.sentence_count == 1
    assert metrics.compression_ratio == 0.5
    assert calculate_rouge("", "reference") == {}
    scores = calculate_rouge("A useful summary.", "A useful reference summary.")
    assert set(scores) == {"rouge1", "rouge2", "rougeL"}
    assert all(0 <= item["f1"] <= 1 for item in scores.values())
