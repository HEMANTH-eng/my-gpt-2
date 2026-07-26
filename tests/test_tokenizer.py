import pytest
from tokenizer.bpe_tokenizer import BPETokenizer


def test_bpe_training_from_text():
    sample_text = "lower lowest newer newest hello world " * 10
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(sample_text)

    # Initial 256 byte tokens + merges created
    assert len(tokenizer.vocab) > 256
    assert len(tokenizer.merges) > 0


def test_bpe_training_from_file(tmp_path):
    corpus_file = tmp_path / "train_corpus.txt"
    corpus_content = "The quick brown fox jumps over the lazy dog. " * 15
    corpus_file.write_text(corpus_content, encoding="utf-8")

    tokenizer = BPETokenizer(vocab_size=310)
    tokenizer.train(str(corpus_file))

    assert len(tokenizer.vocab) <= 310
    ids = tokenizer.encode(corpus_content)
    decoded = tokenizer.decode(ids)
    assert decoded == corpus_content


def test_encode_decode_roundtrip():
    text = "Hello world! PyTorch & Python large language model BPE test 🚀."
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_special_tokens():
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    sample_text = "Hello world! <bos> This is a test. <eos> <pad>"

    tokenizer = BPETokenizer(vocab_size=280, special_tokens=special_tokens)
    tokenizer.train("Hello world! This is a test.", special_tokens=special_tokens)

    ids = tokenizer.encode(sample_text)
    assert tokenizer.special_tokens["<bos>"] in ids
    assert tokenizer.special_tokens["<eos>"] in ids
    assert tokenizer.special_tokens["<pad>"] in ids

    decoded = tokenizer.decode(ids)
    assert decoded == sample_text


def test_save_and_load(tmp_path):
    save_dir = tmp_path / "bpe_model"
    text = "Machine learning and deep learning models require proper tokenization."

    tokenizer1 = BPETokenizer(vocab_size=320, special_tokens=["<pad>", "<eos>"])
    tokenizer1.train(text)
    tokenizer1.save(save_dir)

    tokenizer2 = BPETokenizer()
    tokenizer2.load(save_dir)

    assert tokenizer1.vocab == tokenizer2.vocab
    assert tokenizer1.merges == tokenizer2.merges
    assert tokenizer1.special_tokens == tokenizer2.special_tokens

    ids1 = tokenizer1.encode(text)
    ids2 = tokenizer2.encode(text)
    assert ids1 == ids2

    decoded2 = tokenizer2.decode(ids2)
    assert decoded2 == text


def test_empty_string():
    tokenizer = BPETokenizer(vocab_size=260)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

