from typing import Tuple

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('claimrank')
class ClaimrankPredictor(Predictor):
    """Predictor wrapper for Claimrank"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict)
        return instance

@Predictor.register('seq2seq-claimrank')
class ClaimrankPredictor(Predictor):
    """Predictor wrapper for Seq2Seq Claimrank"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict)
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return ' '.join(outputs['predicted_tokens'])+'\n'
