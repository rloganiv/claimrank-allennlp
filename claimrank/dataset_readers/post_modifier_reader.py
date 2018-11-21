from typing import Any, Dict, Iterator
import json
import logging

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ListField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("post_modifier")
class PostModifierDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 use_prev_sent: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self._use_prev_sent = use_prev_sent
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, data: Dict[str, Any]) -> Instance:
        # Tokenize sentence and make a TextField
        sentence = data['input']
        if self._use_prev_sent:  # Concat previous sentence if using
            previous_sentence = data['previous_sentence']
            sentence = ' '.join((previous_sentence, sentence))
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)

        # Tokenize each property; make properties a ListField of TextFields
        properties = data['properties']
        property_list = []
        for property in properties:
            tokenized_property = self._tokenizer.tokenize(property)
            property_field = TextField(tokenized_property,
                                       self._token_indexers)
            property_list.append(property_field)
        properties_field = ListField(property_list)

        # Tokenize each value; make values a ListField of TextFields
        values = data['values']
        value_list = []
        for value in values:
            tokenized_value = self._tokenizer.tokenize(value)
            value_field = TextField(tokenized_value,
                                       self._token_indexers)
            value_list.append(value_field)
        values_field = ListField(value_list)

        # Stuff everything in a dict
        fields = {
            "sentence": sentence_field,
            "properties": properties_field,
            "values": values_field,
        }

        # If target labels are provided add as SequenceLabelField
        if 'used' in data:
            labels = ["used" if x else "not used" for x in data['used']]
            label_field = SequenceLabelField(labels=labels,
                                             sequence_field=properties_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

