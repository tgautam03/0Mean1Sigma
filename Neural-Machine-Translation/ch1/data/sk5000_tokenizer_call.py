import sentencepiece as spm

modern = spm.SentencePieceProcessor(model_file="../trained_models/tokenizer/modern_en.model")
src_tok = modern.EncodeAsIds(["Every man you meet these days is frowing."])