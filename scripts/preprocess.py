"""
Unifies dataset files into JSONL objects
"""
import argparse
from collections import namedtuple
import csv
import json
import logging
import sys

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


WIKI_FIELDS = ['id', 'name', 'aliases', 'descriptions', 'claims']
PM_FIELDS = ['input', 'name', 'target', 'sentence', 'id', 'previous_sentence',
             'next_sentence', 'file']


def insert_postmod_token(sentence, post_modifier):
    """
    Given a source sentence and a post modifier (which appears in the source
    sentence), replace the post modifier with a <postmod> token.
    """
    if post_modifier not in sentence:
        msg = 'Post-modifier "%s" does not occur in sentence "%s"'
        msg = msg % (post_modifier, sentence)
        raise ValueError(msg)
    output = sentence.replace(post_modifier, '<postmod>')
    return output


def process_wiki_row(row):
    """Processes row from a .wiki file"""
    out = row.copy()
    # Split comma separated fields
    out['aliases'] = row['aliases'].split(',')
    out['descriptions'] = row['descriptions'].split(',')
    # Convert JSON string to dict
    out['claims'] = json.loads(row['claims'])
    return out


def process_pm_row(row):
    """Processes row from a .pm file"""
    out = row.copy()
    new_input = insert_postmod_token(row['sentence'], row['target'])
    out['input'] = new_input
    del out['sentence']
    return out


def combine_rows(pm_row, wiki_row):
    """Combines processed pm and wiki rows. Most importantly, determines which
    claims were used."""
    # Determine which claims were used by checking if pm source file appears in
    # claim's ``used`` list.
    file = pm_row['file']
    claims = wiki_row['claims']
    properties = []
    qualifiers = []
    values = []
    used = []
    for claim in claims:
        property, value = claim['property']
        properties.append(property)
        values.append(value)
        qualifiers.append(claim['qualifiers'])
        if file in claim['used']:
            used.append(True)
        else:
            used.append(False)
    assert sum(used) != 0, print(file, claims)

    # Combine the two rows, replacing the ``claims`` field with the field
    # derived above.
    out = pm_row.copy()
    out.update(wiki_row)
    del out['claims']
    out['properties'] = properties
    out['values'] = values
    out['used'] = used
    out['qualifiers'] = qualifiers

    return out


def row_generator(fname, fieldnames):
    """Generates row objects from a file."""
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, fieldnames, delimiter='\t')
        for i, row in enumerate(reader):
            logger.debug('On row %i', i)
            yield row


def main(_):
    # Load all wiki data into memory
    wiki = dict()
    for wiki_row in row_generator(args.wiki, WIKI_FIELDS):
        wiki[wiki_row['id']] = process_wiki_row(wiki_row)

    # Merge pm data with wiki data, then print merged data as a JSON object
    for pm_row in row_generator(args.pm, PM_FIELDS):
        pm_row = process_pm_row(pm_row)
        wiki_row = wiki[pm_row['id']]
        out = combine_rows(pm_row, wiki_row)
        print(json.dumps(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pm', type=str, help='post-modifier file')
    parser.add_argument('wiki', type=str, help='wiki file')
    parser.add_argument('--debug', action='store_true',
                        help='set logging level to DEBUG')
    args, _ = parser.parse_known_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(_)

