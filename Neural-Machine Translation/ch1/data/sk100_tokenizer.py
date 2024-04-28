import sentencepiece as spm

spm.SentencePieceTrainer.Train(sentence_iterator=iter(src),             # Load dataset
            vocab_size=100,                                             # Vocabulary Size
            model_prefix="../trained_models/tokenizer/modern_en_vs100", # Where to save the model
            pad_id=0, bos_id=1, eos_id=2, unk_id=3                      # Reserve preprocessing tokens
            ) 