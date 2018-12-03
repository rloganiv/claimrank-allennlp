"""
Dataset reader to use for the end-to-end sequence-to-sequence with claim
ranking baseline.

ROB: I opted to use a separate reader to make the claim encoding (hopefully)
a little simpler.
"""
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


@DatasetReader.register("sequence")
class SequenceReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SindleIdTokenIndexer()}

    def text_to_instance(self, data: Dict[str, Any]) -> Instance:
        # Tokenize input sentence
        input = data['input']
        tokenized_input = self._tokenizer.tokenize(input)
        input_field = TextField(tokenized_input, self._token_indexers)

        # Combine and tokenize claims
        properties = data['properties']
        values = data['values']
        qualifiers = data['qualifiers']
        claims_list = []
        for prop, val, quals in zip(properties, values, qualifiers):
            substrings = []
            substrings.extend(['<prop>', prop, '</prop>'])
            substrings.extend(['<val>', val, '</val>'])
            if len(quals) > 0:
                for qp, qv in quals:
                    substrings.extend(['<qual_prop>', qp, '</qual_prop>'])
                    substrings.extend(['<qual_val>', qv, '</qual_val>'])
            claim_string = ' '.join(substrings)
            tokenized_claim = self._tokenizer.tokenize(claim_string)
            claim_field = TextField(tokenized_claim, self._token_indexers)
            claims_list.append(claim_field)
        claims_field = ListField(claims_list)

        # Stuff everything in a dict
        fields = {
            'inputs': input_field,
            'claims': claims_field,
        }

        # If target labels are provided add as SequenceLabelField
        if 'used' in data:
            labels = ['used' if x else 'not used' for x in data['used']]
            label_field = SequenceLabelField(labels=labels,
                                             sequence_field=claims_field)
            fields['labels'] = label_field

        # If target output sequence is provided add as TextField
        if 'target' in data:
            target = data['target']
            tokenized_target = self._tokenizer.tokenize(target)
            fields['targets'] = TextField(tokenized_target,
                                          self._token_indexers)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

