# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""


from .initialize import is_unitialized
from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_embedding_group
from .initialize import get_model_parallel_group
from .initialize import get_tensor_model_parallel_group
from .initialize import get_pipeline_model_parallel_group
from .initialize import get_tensor_model_parallel_rank, set_tensor_model_parallel_rank
from .initialize import get_pipeline_model_parallel_rank, set_pipeline_model_parallel_rank
from .initialize import is_pipeline_first_stage, is_pipeline_last_stage
from .initialize import get_tensor_model_parallel_src_rank
from .initialize import get_pipeline_model_parallel_first_rank
from .initialize import get_pipeline_model_parallel_last_rank
from .initialize import get_pipeline_model_parallel_next_rank
from .initialize import get_pipeline_model_parallel_prev_rank
from .initialize import get_tensor_model_parallel_world_size, set_tensor_model_parallel_world_size
from .initialize import get_pipeline_model_parallel_world_size, set_pipeline_model_parallel_world_size
from .initialize import get_virtual_pipeline_model_parallel_rank, set_virtual_pipeline_model_parallel_rank
from .initialize import initialize_model_parallel
from .initialize import model_parallel_is_initialized
from .initialize import get_model_parallel_world_size, get_model_parallel_rank
from .random import reset_checkpointed_activations_memory_buffer