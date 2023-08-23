import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def partition_uniform_with_embed(num_items, pipeline_parallel_size, num_chunks):
    """
    When partitioning, additional consideration will be given to the need for embedding in the first part
    and the last part, so one less layer will be allocated.
    """

    num_items += 2  # Additional consideration of the two layers of embedding and head
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        # left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        end = partition_items % pipeline_parallel_size
        start = -1
        # If the remaining is less than the number of middle layers, try to allocate to the middle layer
        if end <= pipeline_parallel_size - 2:
            start = 0
            end += 1

        if chunk_size == 0:
            from utils.logger import LLM_LOGGER as logger

            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (start < p < end)
            parts[p].append((st, base_idx))

    # The whole must shift
    real_parts = []
    for idx, _parts in enumerate(parts):
        _real_parts = []
        for _, (s, e) in enumerate(_parts):
            s -= 1  # All forward shift an embedding
            e -= 1
            s = max(s, 0)
            if e == num_items - 1:  # The last head needs to subtract one more bit
                e -= 1
            if e - s > 0:
                _real_parts.append((s, e))

        real_parts.append(_real_parts)

    # num_chunks=1 [[(star, end)], [(start, end)]...] front closed back open
    # num_chunks=2 [[(star, end), (start, end)], [(start, end), (start, end)]...] front closed back open
    # to make sure not wrong, add an assert
    indexes = []
    for _parts in real_parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items - 2))), (
        indexes,
        num_items,
    )  # should have the same indexes as expected
    return real_parts


def partition_uniform(num_items, pipeline_parallel_size, num_chunks):
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            raise ValueError("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected
    return parts


def partition_uniform_with_embed2(num_items, pipeline_parallel_size, num_chunks):
    """
    When partitioning, additional consideration will be given to the need for embedding in the first part
    and the last part, so one less layer will be allocated.
    It seems that the first layer of embedding takes up a lot, so let the first pipe reduce the number of
    layers when splitting.
    """

    assert num_chunks == 1, "Only num_chunks=1 supported."
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"
    # num_items=6, pipeline_parallel_size=4, num_chunks=2

    num_shift_in_pp0 = 2
    if pipeline_parallel_size == 1:
        return [[(0, num_items)]]

    parts = [[] for _ in range(pipeline_parallel_size)]
    item_per_pp_in0 = num_items // pipeline_parallel_size - num_shift_in_pp0
    item_per_pp_inother = (num_items - item_per_pp_in0) // (pipeline_parallel_size - 1)
    left_layer = (num_items - item_per_pp_in0) % (pipeline_parallel_size - 1)
    for p in range(pipeline_parallel_size):
        if p == 0:
            start = 0
            end = item_per_pp_in0
        else:
            start = end
            end = start + item_per_pp_inother
            # should be arranged as far back as possible
            if (pipeline_parallel_size - 1 - (p - 1)) <= left_layer:
                end += 1

        parts[p].append((start, end))

    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected
    return parts


def partition_without_last_head(num_items, pipeline_parallel_size, num_chunks):
    """
    When partitioning, additional consideration will be given to the need for embedding in the first part
    and the last part, so one less layer will be allocated.
    """

    pipeline_parallel_size -= 1  # The last pipeline stage is only for the head
    assert num_chunks == 1, "Currently only 1 chunk is allowed, whether it supports more chunks should be considered."
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"
    # num_items=6, pipeline_parallel_size=4, num_chunks=2

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        # left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        end = partition_items % pipeline_parallel_size
        start = -1
        # If the remaining is less than the number of middle layers, try to allocate it to the middle layer
        if end <= pipeline_parallel_size - 2:
            start = 0
            end += 1

        if chunk_size == 0:
            from utils.logger import LLM_LOGGER as logger

            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (start < p < end)
            parts[p].append((st, base_idx))

    # num_chunks=1 [[(star, end)], [(start, end)]...] front closed back open
    # num_chunks=2 [[(star, end), (start, end)], [(start, end), (start, end)]...] front closed back open
    # to make sure not wrong, add an assert
    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected

    parts.append([])

    return parts