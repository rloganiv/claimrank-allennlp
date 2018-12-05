from typing import Dict, Tuple
import logging

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.attention import Attention
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Average, BLEU
import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: If attention over claims needs to be done at word-level, then switch out ``Seq2VecEncoder``
# for ``Seq2SeqEncoder`` and make necessary changes to attention implementation.
@Model.register("seq2seq-claimrank")
class Seq2SeqClaimRank(Model):
    """
    A ``Seq2SeqClaimRank`` model. This model is intended to be trained with a multi-instance
    learning objective that simultaneously tries to:
        - Decode the given post modifier (e.g. the ``target`` sequence).
        - Ensure that the model is attending to the proper claims during decoding (which are
        identified by the ``labels`` variable).
    The basic architecture is a seq2seq model with attention where the input sequence is the source
    sentence (without post-modifier), and the output sequence is the post-modifier. The main
    difference is that instead of performing attention over the input sequence, attention is
    performed over a collection of claims.

    Parameters
    ==========
    text_field_embedder : ``TextFieldEmbedder``
        Embeds words in the source sentence / claims.
    sentence_encoder : ``Seq2VecEncoder``
        Encodes the entire source sentence into a single vector.
    claim_encoder : ``Seq2SeqEncoder``
        Encodes each claim into a single vector.
    attention : ``Attention``
        Type of attention mechanism used.
        WARNING: Do not normalize attention scores, and make sure to use a
        sigmoid activation. Otherwise the claim ranking loss will not work
        properly!
    max_steps : ``int``
        Maximum number of decoding steps. Default: 100 (same as ONMT).
    beam_size: ``int``
        Beam size used during evaluation. Default: 5 (same as ONMT).
    beta: ``float``
        Weight of attention loss term.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 claim_encoder: Seq2SeqEncoder,
                 attention: Attention,
                 max_steps: int = 100,
                 beam_size: int = 5,
                 beta: float = 1.0) -> None:
        super(Seq2SeqClaimRank, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.sentence_encoder = sentence_encoder
        self.claim_encoder = TimeDistributed(claim_encoder)  # Handles additional sequence dim
        self.claim_encoder_dim = claim_encoder.get_output_dim()
        self.attention = attention
        self.decoder_embedding_dim = text_field_embedder.get_output_dim()
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.beta = beta

        # self.target_embedder = torch.nn.Embedding(vocab.get_vocab_size(), decoder_embedding_dim)

        # Since we are using the sentence encoding as the initial hidden state to the decoder, the
        # decoder hidden dim must match the sentence encoder hidden dim.
        self.decoder_output_dim = sentence_encoder.get_output_dim()
        self.decoder_cell = torch.nn.LSTMCell(self.decoder_embedding_dim + self.claim_encoder_dim,
                                              self.decoder_output_dim)

        # When projecting out we will use attention to combine claim embeddings into a single
        # context embedding, this will be concatenated with the decoder cell output before being
        # fed to the projection layer. Hence the expected input size is:
        #   decoder output dim + claim encoder output dim
        projection_input_dim = self.decoder_output_dim + self.claim_encoder_dim
        self.output_projection_layer = torch.nn.Linear(projection_input_dim,
                                                       vocab.get_vocab_size())

        self._start_index = self.vocab.get_token_index('<s>')
        self._end_index = self.vocab.get_token_index('</s>')

        self.beam_search = BeamSearch(self._end_index, max_steps=max_steps, beam_size=beam_size)
        pad_index = vocab.get_token_index(vocab._padding_token)
        self.bleu = BLEU(exclude_indices={pad_index, self._start_index, self._end_index})
        self.avg_reconstruction_loss = Average()
        self.avg_claim_scoring_loss = Average()

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        output_projections, _, state = self._prepare_output_projections(last_predictions, state)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        return class_log_probabilities, state

    @overrides
    def forward(self,
                inputs: Dict[str, torch.LongTensor],
                claims: Dict[str, torch.LongTensor],
                targets: Dict[str, torch.LongTensor] = None,
                labels: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the model + decoder logic.

        Parameters
        ----------
        inputs : ``Dict[str, torch.LongTensor]``
            Output of `TextField.as_array()` from the `input` field.
        claims : ``Dict[str, torch.LongTensor]``
            Output of `ListField.as_array()` from the `claims` field.
        targets : ``Dict[str, torch.LongTensor]``
            Output of `TextField.as_array()` from the `target` field.
            Only expected during training and validation.
        labels : ``torch.Tensor``
            Output of `LabelField.as_array()` from the `labels` field, indicating which claims were
            used.
            Only expected during training and validation.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing loss tensor and decoder outputs.
        """

        # Obtain an encoding for each input sentence (e.g. the contexts)
        input_mask = util.get_text_field_mask(inputs)
        input_word_embeddings = self.text_field_embedder(inputs)
        input_encodings = self.sentence_encoder(input_word_embeddings, input_mask)

        # Next we encode claims. Note that here we have two additional sequence dimensions (since
        # there are multiple claims per instance, and we want to apply attention at the word
        # level). To deal with this we need to set `num_wrapping_dims=1` for the embedder, and make
        # the claim encoder TimeDistributed.
        claim_mask = util.get_text_field_mask(claims, num_wrapping_dims=1)
        claim_word_embeddings = self.text_field_embedder(claims, num_wrapping_dims=1)
        claim_encodings = self.claim_encoder(claim_word_embeddings, claim_mask)

        # Package the encoder outputs into a state dictionary.
        state = {
            'input_mask': input_mask,
            'input_encodings': input_encodings,
            'claim_mask': claim_mask,
            'claim_encodings': claim_encodings
        }

        # If ``target`` (the post-modifier) and ``labels`` (indicator of which claims are used) are
        # provided then we use them to compute loss.
        if (targets is not None) and (labels is not None):
            state = self._init_decoder_state(state)
            output_dict = self._forward_loop(state, targets, labels)
        else:
            output_dict = {}

        # If model is not training, then we perform beam search for decoding to obtain higher
        # quality outputs.
        if not self.training:
            # Perform beam search
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            # Compute BLEU
            top_k_predictions = output_dict['predictions']
            best_predictions = top_k_predictions[:, 0, :]
            self.bleu(best_predictions, targets['tokens'])

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        predicted_indices = output_dict['predictions']
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x) for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _init_decoder_state(self,
                            state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds fields to the state required to initialize the decoder."""
        # Initialize LSTM hidden state (e.g. h_0) with output of the sentence encoder.
        state['decoder_h'] = state['input_encodings']
        # Initialize LSTM context state (e.g. c_0) with zeros.
        batch_size = state['input_mask'].shape[0]
        state['decoder_c'] = state['input_encodings'].new_zeros(batch_size, self.decoder_output_dim)
        # Initialize previous context.
        state['prev_context'] = state['input_encodings'].new_zeros(batch_size, self.claim_encoder_dim)
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor],
                      labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss using greedy decoding."""
        batch_size = state['input_mask'].shape[0]
        target_tokens = targets['tokens']
        num_decoding_steps = target_tokens.shape[1] - 1

        # Greedy decoding phase
        output_logit_list = []
        attention_logit_list = []
        for timestep in range(num_decoding_steps):
            # Feed target sequence as input
            decoder_input = target_tokens[:, timestep]
            output_logits, attention_logits, state = self._prepare_output_projections(decoder_input, state)
            # Store output and attention logits
            output_logit_list.append(output_logits.unsqueeze(1))
            attention_logit_list.append(attention_logits.unsqueeze(1))

        # Compute reconstruction loss
        output_logit_tensor = torch.cat(output_logit_list, dim=1)
        relevant_target_tokens = target_tokens[:, 1:].contiguous()
        target_mask = util.get_text_field_mask(targets)[:, 1:].contiguous()
        reconstruction_loss = util.sequence_cross_entropy_with_logits(output_logit_tensor,
                                                                      relevant_target_tokens,
                                                                      target_mask)

        # Compute claim scoring loss. A loss is computed between **each** attention vector and the
        # true label. In order for that to work we need to:
        #   a. Tile the source labels (so that they are copied for each word)
        #   b. Mask out padding tokens - this requires taking the outer-product of the target mask
        #       and the claim mask
        attention_logit_tensor = torch.cat(attention_logit_list, dim=1)
        claim_level_mask = (state['claim_mask'].sum(-1) > 0).long()
        attention_mask = target_mask.unsqueeze(-1) * claim_level_mask.unsqueeze(1)
        labels = labels.unsqueeze(1).repeat(1, num_decoding_steps, 1).float()
        claim_scoring_loss = F.binary_cross_entropy_with_logits(attention_logit_tensor, labels, reduction='none')
        claim_scoring_loss *= attention_mask.float()  # Apply mask

        # We want to apply 'batch' reduction (as is done in `sequence_cross_entropy...` which
        # entails averaging over each dimension.
        denom = attention_mask
        for i in range(3):
            denom = denom.sum(-1)
            claim_scoring_loss =  claim_scoring_loss.sum(-1) / (denom.float() + 1e-13)
            denom = (denom > 0)

        total_loss = reconstruction_loss + self.beta * claim_scoring_loss

        # Update metrics
        self.avg_reconstruction_loss(reconstruction_loss)
        self.avg_claim_scoring_loss(claim_scoring_loss)

        output_dict =  {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "claim_scoring_loss": claim_scoring_loss,
            "attention_logits": attention_logit_tensor
        }

        return output_dict

    def _prepare_output_projections(self,
                                    decoder_input: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Embed decoder input
        decoder_word_embeddings = self.text_field_embedder({'tokens': decoder_input})

        # Concat with previous context
        concat = torch.cat((decoder_word_embeddings, state['prev_context']), dim=-1)

        # Run forward pass of decoder RNN
        decoder_h, decoder_c = self.decoder_cell(concat, (state['decoder_h'], state['decoder_c']))
        state['decoder_h'] = decoder_h
        state['decoder_c'] = decoder_h

        # Compute attention and get context embedding. We get an attention score for each word in
        # each claim. Then we sum up scores to get a claim level score (so we can use overlap as
        # supervision).
        claim_encodings = state['claim_encodings']
        claim_mask = state['claim_mask']
        batch_size, n_claims, claim_length, dim = claim_encodings.shape

        flattened_claim_encodings = claim_encodings.view(batch_size, -1, dim)
        flattened_claim_mask = claim_mask.view(batch_size, -1)
        flattened_attention_logits = self.attention(decoder_h, flattened_claim_encodings, flattened_claim_mask)
        attention_logits = flattened_attention_logits.view(batch_size, n_claims, claim_length)

        # Now get claim level encodings by summing word level attention.
        word_level_attention = util.masked_softmax(attention_logits, claim_mask)
        claim_encodings = util.weighted_sum(claim_encodings, word_level_attention)

        # We compute our context directly from the claim word embeddings
        claim_mask = (claim_mask.sum(-1) > 0).float()
        attention_logits = attention_logits.sum(-1)
        attention_weights = torch.sigmoid(attention_logits) * claim_mask
        normalized_attention_weights = attention_weights / (attention_weights.sum(-1, True) + 1e-13)
        context_embedding = util.weighted_sum(claim_encodings, normalized_attention_weights)
        state['prev_context'] = context_embedding

        # Concatenate RNN output w/ context vector and feed through final hidden layer
        projection_input = torch.cat((decoder_h, context_embedding), dim=-1)
        output_logits = self.output_projection_layer(projection_input)

        return output_logits, attention_logits, state

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state['input_mask'].size()[0]
        start_predictions = state['input_mask'].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self.beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {
            'recon': self.avg_reconstruction_loss.get_metric(reset=reset).data.item(),
            'claim': self.avg_claim_scoring_loss.get_metric(reset=reset).data.item()
        }
        # Only update BLEU score during validation and evaluation
        if not self.training:
            all_metrics.update(self.bleu.get_metric(reset=reset))
        return all_metrics

