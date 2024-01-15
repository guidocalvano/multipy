import numpy as np

def count_ones(v):

    # A good way to understand this algoritm is by seeing it as a step wise parallel add of increasingly higher bit
    # unsigned integers.


    total_uint_array = v

    ODD_UINT1_MASK = 0b0101_0101_0101_0101
    odd_uint1_array_as_uint2_array = total_uint_array & ODD_UINT1_MASK
    even_uint1_array_as_uint2_array = (total_uint_array >> 1) & ODD_UINT1_MASK

    total_uint_array = odd_uint1_array_as_uint2_array + even_uint1_array_as_uint2_array

    ODD_UINT2_MASK = 0b0011_0011_0011_0011
    odd_uint2_array_as_uint4_array = total_uint_array & ODD_UINT2_MASK
    even_uint2_array_as_uint4_array = (total_uint_array >> 2) & ODD_UINT2_MASK

    total_uint_array = odd_uint2_array_as_uint4_array + even_uint2_array_as_uint4_array

    ODD_UINT4_MASK = 0b0000_1111_0000_1111
    odd_uint4_array_as_uint8_array = total_uint_array & ODD_UINT4_MASK
    even_uint4_array_as_uint8_array = (total_uint_array >> 4) & ODD_UINT4_MASK

    total_uint_array = odd_uint4_array_as_uint8_array + even_uint4_array_as_uint8_array

    ODD_UINT8_MASK = 0b0000_0000_1111_1111
    odd_uint8_array_as_uint16_array = total_uint_array & ODD_UINT8_MASK
    even_uint8_array_as_uint16_array = (total_uint_array >> 8) & ODD_UINT8_MASK

    total_uint_array = odd_uint8_array_as_uint16_array + even_uint8_array_as_uint16_array

    return total_uint_array



def test_simple_count_ones(test_array):

    two_exponents = np.power(2, np.arange(16))

    bit_count_array = np.sum((test_array[:, np.newaxis] & two_exponents[np.newaxis, :]) != 0, axis=1)

    return bit_count_array

def test_count_ones():
    test_array = np.arange(1024)

    correct_counts = test_simple_count_ones(test_array)
    counts = count_ones(test_array)

    assert np.all(correct_counts == counts)

def most_significant_bit_index(array):

    # float32 has the following format
    # s = sign (1 bit long)
    # t = exponent sign bit
    # e = exponent bit (8 bits long)
    # m = mantissa bit (23 bits long)
    #                 s teee eeee mmmmm mmmmm mmmmm mmmmm mmm
    EXPONENT_MASK = 0b0_1111_1111_00000_00000_00000_00000_000
    MANTISSA_BIT_COUNT = 23
    most_significant_bit_index = ((array.astype(np.float32).view(np.uint32) & EXPONENT_MASK) >> MANTISSA_BIT_COUNT) - 126
    most_significant_bit_index[0] = 0
    return most_significant_bit_index

def test_most_signficant_bit():
    test_array = np.arange(1024)
    correct = np.ceil(np.log2(test_array + 1))[:17]

    actual = most_significant_bit_index(test_array)

    assert np.all(correct == actual)

# test_most_signficant_bit()

def reduce_precalculation_mask(node_ids: np.array, depth: int):

    least_significant_bit = node_ids & -node_ids  # this is some bit voodoo probably based on two complements representation of negative integers

    current_least_significant_cutoff_bit = 0b1 << (depth)

    mask = least_significant_bit >= current_least_significant_cutoff_bit

    return mask


def test_reduce_precalculation():
    node_ids = np.arange(1024) + 1

    for depth in range(1, 10):
        actual = node_ids[reduce_precalculation_mask(node_ids, depth)]
        correct = node_ids[2**(depth) -1::2**depth]
        assert np.all( actual == correct)

test_reduce_precalculation()

def final_calculation_depth(node_ids):
    depth = most_significant_bit_index(node_ids) + count_ones(node_ids) - 2
    # depth[node_ids == 0] = 0

    return depth

def current_node_mask(node_ids, depth):
     return final_calculation_depth(node_ids) == depth


def final_prior_node_id(node_ids):
    least_significant_bit = node_ids & -node_ids
    prior_node_id = node_ids - least_significant_bit

    return prior_node_id

def final_prior_node_idx(node_ids):
    return final_prior_node_id(node_ids) - 1

def reduce_precalculation_prior_node_idx(node_id, depth):
    node_idx = (node_id - 1) - (1 << (depth - 1))
    return node_idx


def scan_step_opportunity(width, depth):
    depth_bit = int(2**depth)
    indices = np.arange(width)
    least_significant_bits = np.log2(indices[1:] & -indices[1:]).astype(np.int32)

    first_order_opportunity = (least_significant_bits | depth_bit) == indices[1:]

    return first_order_opportunity


calculation_depth_lookup = final_calculation_depth(np.arange(1024, dtype=np.int32) + 1)
def scan_kernel(values):
    #@TODO: deal with odd values.shape[0]
    node_ids = np.arange(1024) + 1
    # scan_depth = int(np.ceil(np.log2(values.shape[0])))
    scan_depth = 19

    partial_scan_sum = values.copy()


    for depth in range(1, scan_depth):
        current_reduce_precalculation_mask = reduce_precalculation_mask(node_ids, depth)
        # partial_scan_sum += partial_scan_sum[reduce_precalculation_prior_node_id(node_ids, depth)] * current_reduce_precalculation_mask  # this works on gpu because the out of bounds memory access is multiplied by 0
        reduce_prior_node_idx = reduce_precalculation_prior_node_idx(node_ids, depth)

        partial_scan_sum[current_reduce_precalculation_mask] += partial_scan_sum[reduce_prior_node_idx[current_reduce_precalculation_mask]]
        current_final_precalculation_mask = np.logical_and(final_calculation_depth(node_ids) == depth, ~current_reduce_precalculation_mask)
        partial_scan_sum += partial_scan_sum[final_prior_node_idx(node_ids)] * (1 - current_reduce_precalculation_mask) * current_final_precalculation_mask

    return partial_scan_sum

def segmented_scan_kernel(values, segment_heads):
    #@TODO: deal with odd values.shape[0]
    node_ids = np.arange(1024) + 1
    # scan_depth = int(np.ceil(np.log2(values.shape[0])))
    scan_depth = 19

    partial_scan_sum = values.copy()
    is_completed = segment_heads.copy()

    for depth in range(1, scan_depth):
        current_reduce_precalculation_mask = reduce_precalculation_mask(node_ids, depth)
        # partial_scan_sum += partial_scan_sum[reduce_precalculation_prior_node_id(node_ids, depth)] * current_reduce_precalculation_mask  # this works on gpu because the out of bounds memory access is multiplied by 0
        reduce_prior_node_idx = reduce_precalculation_prior_node_idx(node_ids, depth)

        partial_scan_sum[current_reduce_precalculation_mask] += partial_scan_sum[reduce_prior_node_idx[current_reduce_precalculation_mask]] * (1 - is_completed[current_reduce_precalculation_mask])

        is_completed[current_reduce_precalculation_mask] = np.maximum(is_completed[reduce_prior_node_idx[current_reduce_precalculation_mask]], is_completed[current_reduce_precalculation_mask])
        final_calculation_nodes = final_calculation_depth(node_ids) == depth
        final_prior_nodes = final_prior_node_idx(node_ids)
        current_final_precalculation_mask = np.logical_and(final_calculation_nodes, ~current_reduce_precalculation_mask)
        partial_scan_sum += partial_scan_sum[final_prior_nodes] * (1 - current_reduce_precalculation_mask) * current_final_precalculation_mask * (1 - is_completed)

        is_completed[current_reduce_precalculation_mask] = np.maximum(is_completed[reduce_prior_node_idx[current_reduce_precalculation_mask]], is_completed[current_reduce_precalculation_mask])

        is_completed[final_calculation_nodes] = np.maximum(is_completed[final_prior_nodes[final_calculation_nodes]], is_completed[final_calculation_nodes])

    return partial_scan_sum, is_completed


if __name__ == '__main__':
    print(scan_kernel(np.arange(1024))[:16])
    print(np.cumsum(np.arange(16)))
    print(np.all(scan_kernel(np.arange(1024)) == np.cumsum(np.arange(1024))))

    segments = np.zeros([1024], dtype=np.int32)
    segments[[4, 6, 14]] = 1
    values, is_completed = segmented_scan_kernel(np.arange(1024), segments)
    print(segments[:16])
    print(values[:16])
    print(is_completed[:16])

    # print(final_calculation_depth(np.arange(32, dtype=np.int32)))
    #
    # print(np.unique(final_prior_node_id(np.arange(1024, dtype=np.int32)), return_counts=True
    #                 ))
