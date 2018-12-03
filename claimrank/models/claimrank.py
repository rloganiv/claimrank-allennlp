from typing import Dict
import logging

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import F1Measure
from overrides import overrides
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("claimrank")
class ClaimRank(Model):
    """
    ``ClaimRank`` is a model for ranking the relevance of claims for generating
    post-modifiers.

    Currently supports fully-supervised training.
    """
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super(ClaimRank, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.hidden2score = torch.nn.Linear(in_features=3*encoder.get_output_dim(),
                                            out_features=vocab.get_vocab_size('labels'))
        self.f1 = F1Measure(positive_label=1)

    @overrides
    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                properties: Dict[str, torch.LongTensor],
                values: Dict[str, torch.LongTensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        batch_size = labels.shape[0]

        # Encode sentences
        sentence_mask = get_text_field_mask(sentence)
        sentence_word_embeddings = self.text_field_embedder(sentence)
        sentence_encodings = self.encoder(sentence_word_embeddings, sentence_mask)

        # Encode properties - since there are two sequence dimensions we
        # collapse the embeddings along the property sequence dim (e.g. the one
        # that counts the maxium number of properties out of all of the
        # instances in the batch)
        property_word_mask = get_text_field_mask(properties, num_wrapping_dims=1)
        property_word_embeddings = self.text_field_embedder(properties)

        property_dim = property_word_embeddings.shape
        property_word_embeddings = property_word_embeddings.view(-1, *property_dim[2:])
        property_word_mask = property_word_mask.view(-1, *property_dim[2:-1])
        property_encodings = self.encoder(property_word_embeddings, property_word_mask)
        property_encodings = property_encodings.view(*property_dim[:2], -1)

        # Encode values - same issue applies
        value_word_mask = get_text_field_mask(values, num_wrapping_dims=1)
        value_word_embeddings = self.text_field_embedder(values)

        value_dim = value_word_embeddings.shape
        value_word_embeddings = value_word_embeddings.view(-1, *value_dim[2:])
        value_word_mask = value_word_mask.view(-1, *value_dim[2:-1])
        value_encodings = self.encoder(value_word_embeddings, value_word_mask)
        value_encodings = value_encodings.view(*value_dim[:2], -1)

        # Concatenate all encodings together. Requires tiling the sentence encodings
        sentence_encodings.unsqueeze_(1)
        sentence_encodings = sentence_encodings.repeat(1, property_dim[1], 1)

        concat = torch.cat((sentence_encodings, property_encodings, value_encodings),
                           dim=-1)
        logits = self.hidden2score(concat)
        out = {'score_logits': logits,
               'scores': F.softmax(logits, dim=-1)}

        if labels is not None:
            mask = get_text_field_mask(properties)
            self.f1(logits, labels, mask)
            out['loss'] = sequence_cross_entropy_with_logits(logits, labels, mask)

        return out

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.f1.get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

