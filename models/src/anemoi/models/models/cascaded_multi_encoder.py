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
import torch.nn.functional as F
from anemoi.utils.config import DotDict
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
import torch.distributed as dist
from torch_geometric.data import HeteroData
from torch.utils.checkpoint import checkpoint
from anemoi.models.distributed.shapes import get_shape_shards

from .encoder_processor_decoder import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


def get_shard(tensor: Tensor, dim: int, model_comm_group: Optional[ProcessGroup] = None):
    """Get the current process shard of the tensor."""
    # Get the rank of the current process in the model_comm_group
    rank = dist.get_rank(group=model_comm_group)
    
    # Get the shard shapes (list of tensors)
    shards = torch.tensor_split(tensor, dist.get_world_size(group=model_comm_group), dim=dim)
    
    # Index the shard corresponding to the current rank
    return shards[rank]

def batch_standardize(x):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    x_norm = F.batch_norm(x, running_mean=None, running_var=None, training=True, momentum=0, eps=1e-5)
    return x_norm, mean, var

def batch_destandardize(x, mean, var):
    return x * torch.sqrt(var + 1e-5) + mean

class AnemoiModelCascadedEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with double encoder, multiple encoders are used to map the features to a common dimention, then the actual ancoder is used."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
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

        self.target_space_dims = model_config.model.multi_encoder.target_space_dims
        self.encode_global = model_config.model.multi_encoder.encode_global

        super().__init__(model_config=model_config, data_indices=data_indices, statistics=statistics, graph_data=graph_data)
  
        # Get indices
        assert hasattr(graph_data["data"], "cutout")
        # TODO: extend for multiple lams
        # self.lam_indices = [getattr(graph_data["data"], f"cutout_{i}") for i in range(len(model_config.model.multi_encoder.lam_variables))]
        # self.global_shape = torch.all(~torch.stack(self.lam_indices), dim=0)
        self.lam_indices = [getattr(graph_data["data"], "cutout")]
        self.global_indices = ~self.lam_indices[0]

        self.lam_features = model_config.model.multi_encoder.lam_variables
        self.global_features = model_config.model.multi_encoder.global_variables

        # Define cascaded encoders
        self.lam_encoders = [
            nn.Sequential(
                nn.Linear(self.lam_features[i], self.target_space_dims),
                getattr(nn, model_config.model.activation)() 
            )
            for i in range(len(self.lam_features))
        ]

        # Define cascaded decoders
        self.lam_decoders = [
            nn.Sequential(
                nn.Linear(self.target_space_dims, self.lam_features[i]),
                getattr(nn, model_config.model.activation)() 
            )
            for i in range(len(self.lam_features))
        ]

        if self.encode_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.global_features, self.target_space_dims), 
                getattr(nn, model_config.model.activation)() 
            )
            self.global_decoder = nn.Sequential(
                nn.Linear(self.target_space_dims, self.global_features),
                getattr(nn, model_config.model.activation)() 
            )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = self.target_space_dims
        self.num_output_channels = self.target_space_dims
        # self._internal_input_idx = data_indices.internal_model.input.prognostic
        # self._internal_output_idx = data_indices.internal_model.output.prognostic

        self._internal_input_idx = list(range(self.target_space_dims))
        self._internal_output_idx = list(range(self.target_space_dims))
        print("Prognostic input", self._internal_input_idx, len(self._internal_input_idx))
        print("Prognostic outut", self._internal_output_idx, len(self._internal_output_idx))

    def _assert_matching_indices(self, data_indices: dict) -> None:
        pass

    def cascade_encode(self, x, index, features, encoder):
        batch_size = x.shape[0]
        time_size = x.shape[1]
        ensemble_size = x.shape[2]
        
        data = x[:, :, :, index.flatten(), :features]
        data = einops.rearrange(data, "batch time ensemble grid vars -> batch (time ensemble grid) vars")
        mapped_data = checkpoint(
            encoder,
            data,
        )
        mapped_data = einops.rearrange(
            mapped_data,
            " batch (time ensemble grid) vars -> batch time ensemble grid vars",
            batch=batch_size,
            time=time_size,
            ensemble=ensemble_size,
        )
        return mapped_data
    
    def cascade_decode(self, x, out, index, decoder):
        batch_size = x.shape[0]
        time_size = x.shape[1]
        ensemble_size = x.shape[2]

        data = out[:, :, index.flatten(), :]
        data = einops.rearrange(data, "batch time ensemble grid vars -> batch (time ensemble grid) vars")
        mapped_data = checkpoint(
            decoder,
            data,
        )
        mapped_data = einops.rearrange(
            mapped_data,
            " batch (time ensemble grid) vars -> batch time ensemble grid vars",
            batch=batch_size,
            time=time_size,
            ensemble=ensemble_size,
        )
        return mapped_data
    
    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:

        print("Initial shape: ", x.shape)
        print(model_comm_group)
        x, mean, var = batch_standardize(x)


        # Get shard
        global_indices = get_shard(self.global_indices, 0, model_comm_group)

        # Map lam vars to target vars
        mapped_lams = []
        for i, lam_index in enumerate(self.lam_indices):
            # Get shard
            lam_index = get_shard(lam_index, 0, model_comm_group)
            mapped_lam_data = self.cascade_encode(x, lam_index, self.lam_features[i], self.lam_encoders[i])
            mapped_lams.append(mapped_lam_data)
        
        if self.encode_global:
            mapped_global_data = self.cascade_encode(x, global_indices, self.global_features, self.global_encoder)
        
        else:
            mapped_global_data = x[:, :, :, global_indices.flatten(), :]

        # stack all mapped vars on the grid axis
        x = torch.concatenate(mapped_lams + [mapped_global_data], axis=3)

        print("Shape before EncProcDec: ", x.shape)

        # Normal iter after the first cascaded mapping encoder

        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        print(x_data_latent.shape)

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        print("running Encoder")
        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )
        print("running Procesor")
        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        print("running Decoder")
        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
        )

        # residual connection (just for the prognostic variables)
        print("Shape after EncProcDec: ", x.shape)

        # Decoded
        decoded_lams = []
        for i, lam_index in enumerate(self.lam_indices):
            # Get shard
            lam_index = get_shard(lam_index, 0, model_comm_group)

            mapped_lam_out = self.cascade_decode(x, out, lam_index, self.lam_decoders[i])
            decoded_lams.append(mapped_lam_out)

        if self.encode_global:
            mapped_global_out = self.cascade_decode(x, out, global_indices, self.global_decoder)
        else:
            mapped_global_out = x[:, :, :, global_indices.flatten(), :]

        # stack all mapped vars on the grid axis
        print("Decoded LAM shape: ", decoded_lams[0].shape)
        print("Decoded Global shape: ", mapped_global_out.shape)

        out = torch.concatenate(decoded_lams + [mapped_global_out], axis=3)

        out = batch_destandardize(out, mean, var).to(dtype=x.dtype).clone()

        out[..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        return out
