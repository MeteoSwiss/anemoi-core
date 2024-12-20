# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import pytest
from unittest.mock import patch

from anemoi.training.schedulers.rollout.randomise import RandomList, RandomRange, IncreasingRandom, BaseRandom


def test_determism():
    sched = RandomList([1, 2, 3])
    sched_1 = RandomList([1, 2, 3])

    sched.rollout # Force a retrieval to try and break the determinism

    for i in range(100):
        sched.sync(epoch = i)
        sched_1.sync(epoch = i)

        assert sched.rollout == sched_1.rollout

@pytest.mark.parametrize(
    "rollouts",
    [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [16, 2, 3, 4, 5],
    ]
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps = RandomList([0])._randomly_pick)
def test_random_list(pick_mock: Any, rollouts: list[int]):
    sched = RandomList(rollouts)
    assert sched.rollout in rollouts
    assert sched.maximum_rollout == max(rollouts)

    pick_mock.assert_called_once_with(rollouts)

@pytest.mark.parametrize(
    "minimum, maximum, step",
    [
        (1, 10, 1),
        (1, 10, 2),
        (1, 10, 3),
    ]
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps = RandomList([0])._randomly_pick)
def test_random_range(pick_mock: Any, minimum: int, maximum: int, step: int):
    sched = RandomRange(minimum, maximum, step)
    assert sched.rollout in range(minimum, maximum + 1, step)
    assert sched.maximum_rollout == max(range(minimum, maximum + 1, step))

    pick_mock.assert_called_once_with(range(minimum, maximum + 1, step))


@pytest.mark.parametrize(
    "minimum, maximum, step, every_n, epoch_test, expected_max",
    [
        (1, 10, 1, 1, 0, 1),
        (1, 10, 1, 1, 1, 2),
        (1, 10, 1, 1, 2, 3),
        (1, 10, 1, 1, 10, 10),
        (1, 10, 1, 1, 100, 10),
        (1, 10, 1, 2, 2, 2),
        (1, 10, 1, 2, 4, 3),
        (1, 10, 2, 2, 4, 3),

    ]
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps = RandomList([0])._randomly_pick)
def test_increasing_random_increment(pick_mock: Any, minimum: int, maximum: int, step: int, every_n: int, epoch_test: int, expected_max: int):
    sched = IncreasingRandom(minimum, maximum, step, every_n, 1)

    sched.sync(epoch = epoch_test)

    assert sched.current_maximum == expected_max
    assert sched.rollout in list(range(minimum, expected_max + 1, step))
    assert sched.maximum_rollout == maximum
    
    pick_mock.assert_called_once_with(range(minimum, expected_max + 1, step))


@pytest.mark.parametrize(
    "minimum, maximum, step, every_n, increment, epoch_test, expected_max",
    [
        (1, 10, 1, 1, {0:0, 2:1, 4:2,}, 0, 1),
        (1, 10, 1, 1, {0:0, 2:1, 4:2,}, 1, 1),
        (1, 10, 1, 1, {0:0, 2:1, 4:2,}, 2, 2),
        (1, 10, 1, 1, {0:0, 2:1, 4:2,}, 3, 3),
        (1, 10, 1, 1, {0:0, 2:1, 4:2,}, 4, 5),
        (1, 10, 1, 1, {0:0, 2:1, 4:2,}, 5, 7),
        (1, 10, 1, 1, {0:0, 2:1, 3:0, 4:2,}, 4, 4),
        (1, 10, 1, 1, {0:0, 2:1, 3:0, 4:2,}, 5, 6),
        (1, 10, 1, 1, {0:0, 2:1, 3:0, 4:2,}, 1000, 10),
        (1, 10, 2, 1, {0:0, 2:1, 3:0, 4:2,}, 1000, 10),
    ]
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps = RandomList([0])._randomly_pick)
def test_increasing_random_complex_increment(pick_mock: Any, minimum: int, maximum: int, step: int, every_n: int, increment: dict[int, int], epoch_test: int, expected_max: int):

    sched = IncreasingRandom(minimum, maximum, step, every_n, increment=increment)

    sched.sync(epoch = epoch_test)
    assert sched.rollout in list(range(minimum, expected_max + 1, step))
    assert sched.current_maximum == expected_max
    pick_mock.assert_called_with(range(minimum, expected_max + 1, step))


@pytest.mark.parametrize(
    "minimum, maximum, step, every_n, increment, epoch_test, expected_max",
    [
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 0, 1),
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 1, 1),
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 2, 2),
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 3, 2),
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 4, 4),
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 5, 4),
        (1, 10, 1, 2, {0:0, 2:1, 4:2,}, 6, 6),
    ]
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps = RandomList([0])._randomly_pick)
def test_increasing_random_complex_increment_every_not_1(pick_mock: Any, minimum: int, maximum: int, step: int, every_n: int, increment: dict[int, int], epoch_test: int, expected_max: int):

    sched = IncreasingRandom(minimum, maximum, step, every_n, increment=increment)

    sched.sync(epoch = epoch_test)
    assert sched.rollout in list(range(minimum, expected_max + 1, step))
    assert sched.current_maximum == expected_max
    pick_mock.assert_called_with(range(minimum, expected_max + 1, step))

