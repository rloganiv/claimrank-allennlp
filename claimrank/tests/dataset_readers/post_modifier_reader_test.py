from allennlp.common.util import ensure_list
from allennlp.data.vocabulary import Vocabulary
import pytest

from claimrank.dataset_readers import PostModifierDatasetReader


class TestPostModifierDatasetReader:

    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("use_prev_sent", (True, False))
    def test_read_from_file(self, lazy, use_prev_sent):
        reader = PostModifierDatasetReader(lazy=lazy,
                                           use_prev_sent=use_prev_sent)
        fixture_path = 'claimrank/tests/fixtures/pm_data.jsonl'
        instances = ensure_list(reader.read(fixture_path))

        instance1 = {
            "sentence": "`` They 're trying to figure out our business and "
                        "our business models , '' said Larry Augustin , "
                        "<postmod> .".split(),
            "previous_sentence": "The brisk revenue growth has not been "
                                 "enough to convince more than a couple "
                                 "dozen fund managers that the company "
                                 "is worth a $ 1.7 billion market "
                                 "capitalization .".split(),
            "properties": [
                "sex or gender", "educated at", "educated at", "educated at",
                "educated at", "occupation", "occupation", "occupation",
                "given name", "family name", "country of citizenship",
                "position held", "date of birth"
            ],
            "values": [
                "male", "University of Notre Dame", "Stanford University",
                "Stanford University",
                "Stanford University School of Engineering", "engineer",
                "computer scientist", "businessperson", "Larry", "Augustin",
                "United States of America", "chief executive officer",
                "+1962-10-10T00:00:00Z"
            ],
            "used": [
                False, False, False, True, True, True, False, False, False,
                False, False, True, True, False, False, False, True, False,
                False, False, False, False, True, True
            ]
        }

        assert len(instances) == 3
        fields = instances[0].fields

        if use_prev_sent:
            expected = instance1['previous_sentence'] + instance1['sentence']
        else:
            expected = instance1['sentence']

        # TODO: Check property, value and used fields



