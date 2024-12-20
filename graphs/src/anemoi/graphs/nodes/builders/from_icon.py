# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC

import torch

from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.utils.config import DotDict


class ICONBaseNodeBuilder(BaseNodeBuilder, ABC):
    """Base class for building ICON nodes."""

    def __init__(self, name: str, grid_filename: str, max_level: int) -> None:
        super().__init__(name)
        self.grid_filename = grid_filename
        self.max_level = max_level
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"icon_nodes", "max_level"}

    def get_coordinates(self) -> torch.Tensor:
        return torch.from_numpy(self.icon_nodes.node_coordinates()).fliplr()


class ICONMultimeshNodes(ICONBaseNodeBuilder):
    """Processor mesh based on an ICON grid."""

    def __init__(self, name: str, grid_filename: str, max_level: int) -> None:
        super().__init__(name, grid_filename, max_level)
        self.icon_nodes = ICONMultiMesh(self.grid_filename, max_level=self.max_level)


class ICONCellGridNodes(ICONBaseNodeBuilder):
    """Data mesh based on an ICON grid."""

    def __init__(self, name: str, grid_filename: str, max_level: int) -> None:
        super().__init__(name, grid_filename, max_level)
        self.icon_nodes = ICONCellDataGrid(self.grid_filename, max_level=self.max_level)
