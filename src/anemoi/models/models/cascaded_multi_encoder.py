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
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.models import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiModelCascadedEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with double encoder, one for model levels and another for pressur levels"""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
        lam_index: int,
        global_shape: int,
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
        lam_index : int
            Index that separates lam data and global data in the input 
        global_shape: int
        """
        super().__init__(model_config, data_indices, graph_data)

        # Model level encoder
        self.lam_index = lam_index
        self.global_shape = global_shape

        # Define first encoder
        self.ml_encoder = nn.Sequential(nn.Linear(config.model.ml_features, config.model.pl_features), act_func = getattr(nn, config.model.activation))

        #Define decoder
        self.ml_decoder = nn.Sequential(nn.Linear(config.model.pl_features, config.model.ml_features), act_func = getattr(nn, config.model.activation))


    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        grid_size = x.shape[3]
        vars_size = x.shape[4]

        # Map ml to pl
        lam_data = x[:,:,:,:self.lam_index, :]
        lam_data = einops.rearrange(lam_data, "batch time ensemble grid vars -> batch (time ensemble grid) vars")
        mapped_lam_data = self.ml_encoder(lam_data)
        mapped_lam_data = einops.rearrange(mapped_lam_data, 
                                           " batch (time ensemble grid) vars -> batch time ensemble grid vars", 
                                           batch=batch_size,
                                           ensemble=ensemble_size
                                           )
        # Remove padding on original data
        x = x[:,:,:,:,:self.global_shape]
        # Insert mapped features
        x[:,:,:,:self.lam_index, :] = mapped_lam_data

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x.dtype)
            .clone()
        )

        # Map pl back to ml
        lam_data = x_out[:,:,:self.lam_index, :]
        lam_data = einops.rearrange(lam_data, "batch ensemble grid vars -> batch (ensemble grid) vars")
        mapped_lam_data = self.ml_decoder(lam_data)
        mapped_lam_data = einops.rearrange(mapped_lam_data, 
                                           "batch (ensemble grid) vars -> batch ensemble grid vars", 
                                           batch=batch_size,
                                           ensemble=ensemble_size
                                           )
        # Add padding on output global data
        padded_out = torch.zeros([batch_size, ensemble_size, grid_size, vars_size], torch.float32).to(x_out.device)
        padded_out[:,:,:,:,:self.global_shape] = x_out
        # Insert mapped features
        padded_out[:,:,:,:self.lam_index, :] = mapped_lam_data

        # residual connection (just for the prognostic variables)
        padded_out[..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]
        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            padded_out = bounding(padded_out)
        return padded_out
