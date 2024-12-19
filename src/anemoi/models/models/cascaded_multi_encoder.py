# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import torch
from anemoi.utils.config import DotDict
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.models import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiModelCascadedEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with double encoder, multiple encoders are used to map the features to a common dimention, then the actual ancoder is used."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """

        self.target_space_dims = model_config.multi_encoder.target_space_dims
        self.encode_global = model_config.multi_encoder.encode_global

        # Get indices
        assert hasattr(graph_data["data"], "cutout")
        self.lam_indices = getattr(graph_data["data"], "cutout")
        self.global_shape = not self.lam_indices
        self.lam_features = graph_data["data"].x.numpy()[self.lam_indices].shape[1]
        self.global_features = graph_data["data"].x.numpy()[self.global_shape].shape[1]

        # Define cascaded encoders
        self.lam_encoders = [
            nn.Sequential(
                nn.Linear(self.lam_features[i], self.target_space_dims),
                act_func=getattr(nn, model_config.model.activation),
            )
            for i in range(len(self.lam_features))
        ]
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_features, self.target_space_dims), act_func=getattr(nn, model_config.model.activation)
        )

        # Define cascaded decoders
        self.lam_decoder = [
            nn.Sequential(
                nn.Linear(self.target_space_dims, self.lam_features[i]),
                act_func=getattr(nn, model_config.model.activation),
            )
            for i in range(len(self.lam_features))
        ]
        self.global_decoder = nn.Sequential(
            nn.Linear(self.target_space_dims, self.global_features), act_func=getattr(nn, model_config.model.activation)
        )

        # TODO: overwrite this:
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        super().__init__(model_config=model_config, data_indices=data_indices, graph_data=graph_data)

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:

        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # Map lam vars to target vars
        mapped_lams = []
        prev_idx = 0
        for i, lam_index in enumerate(self.lam_indices):
            lam_data = x[:, :, :, prev_idx:lam_index, : self.lam_features[i]]
            lam_data = einops.rearrange(lam_data, "batch time ensemble grid vars -> batch (time ensemble grid) vars")
            mapped_lam_data = self.lam_encoders[i](lam_data)
            mapped_lam_data = einops.rearrange(
                mapped_lam_data,
                " batch (time ensemble grid) vars -> batch time ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )

            # Append and increment idx
            mapped_lams.append(mapped_lam_data)
            prev_idx = lam_index

        # Map global vars to target vars
        global_data = x[:, :, :, lam_index:, : self.global_features]
        global_data = einops.rearrange(global_data, "batch time ensemble grid vars -> batch (time ensemble grid) vars")
        mapped_global_data = self.global_encoder(global_data)
        mapped_global_data = einops.rearrange(
            mapped_global_data,
            " batch (time ensemble grid) vars -> batch time ensemble grid vars",
            batch=batch_size,
            ensemble=ensemble_size,
        )

        # stack all mapped vars on the grid axis
        x = torch.concatenate(mapped_lams + [mapped_global_data], axis=3)

        # Normal iter after the first cascaded mapping encoder
        super().forward(x, model_comm_group=model_comm_group)
