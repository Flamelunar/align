#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Zhenghua Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import sys
import math
import numpy as np


# ***************************************************************
class Bucketing(object):
    """
    A simple bucketing method.
    The goals are:
        The resulting bucket number can be different from the given k
        Different buckets contain roughly the same number of words (no less than 0.6 of the average)
        The gap between the max and min sentence lengths are as small as possible (7 can be threshold)
    """

    # =============================================================
    def __init__(self, assumed_bucket_num, len_cntr):
        """"""
        # Error checking
        if len(len_cntr) < assumed_bucket_num:
            raise ValueError('Trying to sort %d data points into %d buckets' % (len(len_cntr), assumed_bucket_num))

        uniq_lengths = sorted(len_cntr.keys(), reverse=True) # from large to small
        total_token_num = 0
        for length in uniq_lengths:
            total_token_num += length * len_cntr[length]
        average_token_num = math.ceil(total_token_num / assumed_bucket_num)
        '''
        lengths = []
        for length, count in len_cntr.items():
            lengths.extend([length] * count)
        lengths.sort()
        print(lengths)
        '''

        self._max_len_in_buckets = []
        uniq_len_num = len(uniq_lengths)
        total_token_num_this_bucket = 0
        for i_len in range(uniq_len_num):
            length = uniq_lengths[i_len]
            i_bucket = len(self._max_len_in_buckets) - 1
            if -1 == i_bucket or total_token_num_this_bucket >= average_token_num or \
                    (total_token_num_this_bucket >= 0.6 * average_token_num and
                     self._max_len_in_buckets[i_bucket] - uniq_lengths[i_len] > 7):
                self._max_len_in_buckets.append(length)
                total_token_num_this_bucket = 0
            else:
                pass
            total_token_num_this_bucket += length * len_cntr[length]

            # if the last bucket is too small, merge it into the second-last
            if i_len == (uniq_len_num - 1) and total_token_num_this_bucket < 0.4 * average_token_num:
                assert 0 < i_bucket
                self._max_len_in_buckets.pop()

        self._max_len_in_buckets.reverse()

        self._len2bucket_idx = {}
        last_split = -1
        for split_idx, split in enumerate(self._max_len_in_buckets):
            self._len2bucket_idx.update(
                dict(zip(range(last_split+1, split+1), [split_idx] * (split - last_split))))
            last_split = split  # add this line by zhenghua (a bug)

    # =============================================================
    @property
    def max_len_in_buckets(self):
        return self._max_len_in_buckets

    @property
    def len2bucket_idx(self):
        return self._len2bucket_idx


# ***************************************************************
if __name__ == '__main__':
    np.random.seed(123)
    len_cntr = Counter()
    for i in range(100):
        # len_cntr[1+int(10**(1+np.random.randn()))] += 1
        len_cntr[1 + np.random.randint(1, 10)] += 1
    print(len_cntr)
    buckets = Bucketing(4, len_cntr)
    print('splits: ', buckets.max_len_in_buckets)
    print('len2split_idx: ', buckets.len2bucket_idx)
