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
        },
        "prev": true
    },
    "train_data_path": "/home/rlogan/projects/PostModifier/dataset/train.jsonl",
    "validation_data_path": "/home/rlogan/projects/PostModifier/dataset/valid.jsonl",
    "model": {
        "type": "seq2seq-claimrank",
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 500
            }
        },
        "sentence_encoder": {
            "type": "lstm",
            "input_size": 500,
            "hidden_size": 250,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": true
        },
        "claim_encoder": {
            "type": "lstm",
            "input_size": 500,
            "hidden_size": 250,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": true
        },
        "attention": {
            # Dimensions are dependent on encoder hidden sizes.
            "vector_dim": 500,
            "matrix_dim": 500,
            "type": "bilinear",
            "normalize": false
        },
        "beta": 1.0
    },
    "iterator": {
        "type": "basic",
        "batch_size": 12
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 1e-3,
        },
        "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [7, 9, 10, 12, 13, 15],
            "gamma": 0.5,
        },
        "grad_clipping": 1.0,
        "num_epochs": 15,
        "patience": 15,
        "cuda_device": 0,
        "validation_metric": "+BLEU",
        "should_log_learning_rate": true
    }
}
