# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.schedulers.rollout.stepped import Stepped

@pytest.mark.parametrize(
    "minimum, maximum, every_n, increment, epoch_test, expected_value",
    [
        # Increment of 1 and every_n of 1
        (1, 10, 1, 1, 0, 1),
        (1, 10, 1, 1, 1, 2),
        (1, 10, 1, 1, 5, 6),
        (1, 10, 1, 1, 6, 7),
        (1, 10, 1, 1, 8, 9),
        (1, 10, 1, 1, 9, 10),
        (1, 10, 1, 1, 10, 10),
        (1, 10, 1, 1, 11, 10),
        (1, 10, 1, 1, 1000, 10),

        # Increment of 2 and every_n of 1
        (1, 10, 1, 2, 1, 3),
        (1, 10, 1, 2, 2, 5),
        (1, 10, 1, 2, 4, 9),
        (1, 10, 1, 2, 5, 10),

        # Increment of 1 and every_n of 2
        (1, 10, 2, 1, 0, 1),
        (1, 10, 2, 1, 1, 1),
        (1, 10, 2, 1, 2, 2),

    ]
)
def test_stepped(minimum: int, maximum: int, every_n: int, increment: int, epoch_test: int, expected_value: int):
    sched = Stepped(minimum, maximum, every_n, increment=increment)

    sched.sync(epoch = epoch_test)
    assert sched.rollout == expected_value
    assert sched.current_maximum == expected_value



@pytest.mark.parametrize(
    "minimum, maximum, every_n, increment, epoch_test, expected_value",
    [
        (1, 10, 1, {0:0, 2:1, 4:2,}, 0, 1),
        (1, 10, 1, {0:0, 2:1, 4:2,}, 1, 1),
        (1, 10, 1, {0:0, 2:1, 4:2,}, 2, 2),
        (1, 10, 1, {0:0, 2:1, 4:2,}, 3, 3),
        (1, 10, 1, {0:0, 2:1, 4:2,}, 4, 5),
        (1, 10, 1, {0:0, 2:1, 4:2,}, 5, 7),
        (1, 10, 1, {0:0, 2:1, 3:0, 4:2,}, 4, 4),
        (1, 10, 1, {0:0, 2:1, 3:0, 4:2,}, 5, 6),
        (1, 10, 1, {0:0, 2:1, 3:0, 4:2,}, 1000, 10),
    ]
)
def test_stepped_complex_increment(minimum: int, maximum: int, every_n: int, increment: dict[int, int], epoch_test: int, expected_value: int):

    sched = Stepped(minimum, maximum, every_n, increment=increment)

    sched.sync(epoch = epoch_test)
    assert sched.rollout == expected_value
    assert sched.current_maximum == expected_value
