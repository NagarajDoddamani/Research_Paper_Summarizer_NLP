from modules.preprocessing import preprocess, split_sentences
from modules.tfidf_summarizer import summarize_tfidf
from modules.textrank_summarizer import summarize_textrank


TEXT = ("Neural networks learn patterns from data. They are widely used in image recognition. "
        "Support vector machines find separating boundaries. Neural networks can model complex patterns.")


def test_preprocessing_retains_sentence_order_and_text():
    sentences = preprocess(TEXT)
    assert len(sentences) == 4
    assert sentences[0].original.startswith("Neural networks")
    assert sentences[0].normalized_tokens


def test_sentence_split_and_empty_input():
    assert split_sentences("") == []
    assert split_sentences("One sentence.") == ["One sentence."]
    assert summarize_tfidf("") == []
    assert summarize_textrank("   ") == []


def test_both_summarizers_return_requested_original_order():
    for summarize in (summarize_tfidf, summarize_textrank):
        summary = summarize(TEXT, 2)
        assert len(summary) == 2
        assert all(sentence in TEXT for sentence in summary)
        assert TEXT.index(summary[0]) < TEXT.index(summary[1])


def test_short_and_zero_vector_documents():
    assert summarize_tfidf("A single useful sentence.", 5) == ["A single useful sentence."]
    assert summarize_textrank("!!! ...", 1)


def test_percentage_bounds_and_selection():
    assert len(summarize_tfidf(TEXT, percentage=0.5)) == 2
    assert len(summarize_textrank(TEXT, percentage=2.0)) == 4
