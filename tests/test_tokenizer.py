"""Unit test for ByteLevelBPETokenizer."""

import os

from core.tokenizer import ByteLevelBPETokenizer, SpecialToken


def test_tokenizer_train_and_roundtrip() -> None:
    """Tests training, encode, decode, and persistence of the tokenizer."""
    sample_corpus = [
        "Xin chào thế giới! Đây là đồ án xây dựng mô hình ngôn ngữ lớn từ số 0.",
        "DeepSeek Multi-Head Latent Attention (MLA) và BPE Tokenizer 🚀.",
        "Trí tuệ nhân tạo đang thay đổi thế giới nhanh chóng.",
        "<|think|> Đây là chuỗi suy luận mẫu </|think|> Câu trả lời là 42.",
    ]

    tokenizer = ByteLevelBPETokenizer()
    target_vocab_size = 350
    tokenizer.train(sample_corpus, vocab_size=target_vocab_size, min_frequency=1)

    print(f"Vocabulary size trained: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size <= target_vocab_size

    # Test encode and decode roundtrip
    test_text = "Xin chào DeepSeek MLA! 🚀 <|think|> Tư duy logic </|think|>"
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)

    print(f"Original: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded:  {decoded_text}")
    assert test_text == decoded_text, f"Mismatch: '{test_text}' != '{decoded_text}'"

    # Test save and load
    save_path = "data/tokenizer_test.json"
    tokenizer.save(save_path)
    assert os.path.exists(save_path)

    loaded_tokenizer = ByteLevelBPETokenizer.load(save_path)
    loaded_decoded = loaded_tokenizer.decode(loaded_tokenizer.encode(test_text))
    assert loaded_decoded == test_text
    print("Tokenizer test passed successfully!")


if __name__ == "__main__":
    test_tokenizer_train_and_roundtrip()

