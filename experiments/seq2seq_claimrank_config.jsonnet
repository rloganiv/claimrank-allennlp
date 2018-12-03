{
    "vocabulary": {
        "max_vocab_size": { "tokens": 50000 }
    },
    "dataset_reader": {
        "type": "sequence",
        "tokenizer": {
            "type": "word",
            "word_splitter": "just_spaces",
            "start_tokens": ["<s>"],
            "end_tokens":["</s>"]
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false
            }
        }
    },
    "train_data_path": "/home/rlogan/projects/PostModifier/dataset/train.jsonl",
    "validation_data_path": "/home/rlogan/projects/PostModifier/dataset/valid.jsonl",
    "model": {
        "type": "seq2seq-claimrank",
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 100
            }
        },
        "sentence_encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 256,
            "num_layers": 1,
            "dropout": 0.50,
            "bidirectional": true
        },
        "claim_encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 256,
            "num_layers": 1,
            "dropout": 0.50,
            "bidirectional": true
        },
        "attention": {
            # Dimensions are dependent on encoder hidden sizes.
            "vector_dim": 512,
            "matrix_dim": 512,
            "type": "bilinear",
            "normalize": false
        },
        "decoder_embedding_dim": 200
    },
    "iterator": {
        "type": "basic",
        "batch_size": 16
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 3e-4
        },
        "num_epochs": 50,
        "patience": 10,
        "cuda_device": 0,
        "validation_metric": "+BLEU"
    }
}
