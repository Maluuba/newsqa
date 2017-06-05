import json
import re
import string
from collections import Counter, namedtuple

import numpy as np
import six

TAG_B = "BBBBBB"
TAG_E = "EEEEEE"

USER_DELIMITER = "|"
SPAN_DELIMITER = ","
EDGE_DELIMITER = ":"

Span = namedtuple('Span', ['s', 'e'])


def span_to_string(span):
    return "%d%s%d" % (span.s, EDGE_DELIMITER, span.e)


def span_from_string(span_string):
    s, e = map(int, span_string.split(EDGE_DELIMITER))
    return Span(s, e)


def span_array_from_string(spans_string):
    result = map(span_from_string, spans_string.split(SPAN_DELIMITER))
    if not six.PY2:
        result = list(result)
    return result


def span_rack_from_string(spans_string):
    span_rack = []
    for user in spans_string.split(USER_DELIMITER):
        if user != "None" and user:
            spans = span_array_from_string(user)
            spans = sorted(spans, key=lambda span: span[0])
            span_rack.append(spans)
    return span_rack


def span_rack_to_string(span_rack):
    return USER_DELIMITER.join([
                                   SPAN_DELIMITER.join([span_to_string(span) for span in user])
                                   for user in span_rack])


def rebase_span_array(span_array):
    """Rebase ranges to match the untagged, tokenized text positions.
    """
    shift = 0
    rebased_array = []
    for span in span_array:
        begin = span.s - shift
        shift += len(TAG_B) + 2 + len(TAG_E)
        end = span.e - shift
        rebased_array.append(Span(s=begin, e=end))
    return rebased_array


def extract_spans_from_text(spans, text):
    splitted_text = text.split()
    return [' '.join(splitted_text[span.s:span.e]) for span in spans]


def remove_tags(tagged_text):
    """
    Remove tags from tokenized tagged text.
    """
    return " ".join(
        filter(lambda a: a != TAG_B and a != TAG_E, tagged_text.split()))


def tag_text_from_span_rack(span_rack, untokenized_text):
    """Given the untokenized text, it inserts special tags in the positions
       specified by ranges.
    """
    tagged_texts = []
    if len(span_rack) > 0:
        for span_array in span_rack:
            tag_shift = 0
            tag_list = list(untokenized_text)
            for span in span_array:
                tag_list[span.s + tag_shift: span.e + tag_shift] = \
                    ' %s %s %s ' % (
                        TAG_B, untokenized_text[span.s:span.e], TAG_E)
                tag_shift += len(TAG_B) + len(TAG_E) + 4
            tagged_texts.append("".join(tag_list))
    else:
        tagged_texts.append(untokenized_text)
    return tagged_texts


def char_to_word_index(spans, text):
    return [Span(s=text[:span.s].count(" "), e=text[:span.e].count(" ") + 1)
            for span in spans]


regex = re.compile('%s (.*?) %s' % (TAG_B, TAG_E))


def span_rack_from_tag_text(tagged_text, untagged_text):
    """Get spans in the tagged, tokenized text and convert.
    """
    span_rack = []
    for num, tt in enumerate(tagged_text):
        matches = regex.finditer(tt)
        span_array = [Span(s=match.start(), e=match.end()) for match in matches]
        span_array = char_to_word_index(rebase_span_array(span_array), untagged_text)
        span_rack.append(span_array)
    return span_rack


def has_overlap(span_array_1, span_array_2):
    for s1 in span_array_1:
        for s2 in span_array_2:
            if s1.s <= s2.s and s1.e >= s2.s:
                return True
            if s1.s <= s2.e and s1.e >= s2.e:
                return True
    return False


def get_most_overlap(span_rack):
    if len(span_rack) <= 1:
        return 0
    overlap_counts = [0] * len(span_rack)
    for i in six.moves.range(len(span_rack) - 1):
        for j in six.moves.range(i + 1, len(span_rack)):
            if has_overlap(span_rack[i], span_rack[j]):
                overlap_counts[i] += 1
                overlap_counts[j] += 1
    return np.argmax(overlap_counts)


def refine_answers(span_rack, untokenized_text):
    if span_rack:
        for i, span_array in enumerate(span_rack):
            tl = untokenized_text
            for j, _span in enumerate(span_array):
                head, tail = _span.s, _span.e

                while head < len(tl) and head >= 0 and (tl[head] in string.punctuation or tl[head] in string.whitespace):
                    head += 1
                while head >= 1 and head < len(tl) and tl[head - 1] in string.letters:
                    head -= 1

                while tail >= 1 and tail <= len(tl) and (tl[tail - 1] in string.punctuation or tl[tail - 1] in string.whitespace):
                    tail -= 1
                while tail >= 1 and tail < len(tl) and tl[tail] in string.letters:
                    tail += 1

                if head >= len(tl):
                    head = _span.s
                if tail < 1:
                    tail = _span.e
                if head < tail:
                    span_array[j] = Span(s=head, e=tail)
            span_rack[i] = span_array
        which = get_most_overlap(span_rack)
        return [span_rack[which]]
    return []


def valid_span_rack_from_string(validated_answers, untokenized_text):
    span_rack = []
    if not validated_answers:
        return []
    validated_answers = json.loads(validated_answers)
    if len(validated_answers) == 0:
        return []
    best_answer, count = Counter(validated_answers).most_common(1)[0]
    if best_answer == 'none' or best_answer == 'bad_question':
        return []
    if count < 2:
        return []
    answer = best_answer
    _span = span_from_string(answer)

    tl = untokenized_text

    head, tail = _span.s, _span.e

    while head < len(tl) and head >= 0 and (tl[head] in string.punctuation or tl[head] in string.whitespace):
        head += 1
    while head >= 1 and head < len(tl) and tl[head - 1] in string.letters:
        head -= 1
    while tail >= 1 and tail <= len(tl) and (tl[tail - 1] in string.punctuation or tl[tail - 1] in string.whitespace):
        tail -= 1
    while tail < len(tl) and tail >= 1 and tl[tail] in string.letters:
        tail += 1
    if head >= len(tl):
        head = _span.s
    if tail < 1:
        tail = _span.e
    if head < tail:
        _span = Span(s=head, e=tail)

    span_rack.append([_span])
    return span_rack


def nearby_range_merge(span_rack, threshold=0):
    # When threshold equals to 0, only concat continuous spans.
    if len(span_rack) == 0:
        return span_rack
    new_span_rack = []
    for span_array in span_rack:
        if len(span_array) <= 1:
            new_span_rack.append(span_array)
            continue
        new_span_array = [span_array[0]]
        for i in six.moves.range(len(span_array) - 1):
            _r = span_array[i + 1]
            if _r.e > new_span_array[-1].e and _r.s - new_span_array[-1].e <= threshold:
                new_span_array[-1] = Span(s=new_span_array[-1].s, e=_r.e)
            else:
                new_span_array.append(_r)
        new_span_rack.append(new_span_array)
    return new_span_rack
