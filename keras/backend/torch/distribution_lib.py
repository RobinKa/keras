"""!!!DO NOT USE!!!

Distribution related class for JAX backend.

This is just a prototype and we might want to unify it
with other backends in the future.
"""

import numpy as np
import torch.distributed as dist
import torch.distributed._tensor as dist_tensor
import torch.distributed.device_mesh as torch_mesh


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    return list(range(dist.get_world_size()))


def distribute_variable(value, layout):
    """Create a distributed variable for JAX.

    Since JAX doesn't have a variable class, this will just return a `jax.Array`
    with the corresponding layout/sharding specified.

    Note that this function should be used in eager context, not in jitted
    function.

    Args:
        value: the initial value of the variable.
        layout: `TensorLayout` for the created variable, or a
            `jax.sharding.Sharding` instance.

    Returns:
        jax.Array which is the distributed variable.
    """

    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Note that this function can be used both in eager context, or within a
    jitted function.

    Args:
        tensor: `jax.Array` that need to be distributed.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        Distributed value.
    """
    if not (
        isinstance(layout, list)
        and all(
            isinstance(layout_part, dist_tensor.Placement) for layout_part in layout
        )
    ):
        placements = _to_torch_placements(layout)
    mesh = _to_torch_mesh(layout.device_mesh)
    return dist_tensor.distribute_tensor(
        tensor=tensor, device_mesh=mesh, placements=placements
    )


def distribute_data_input(inputs, layout):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to the each of the devices.

    Args:
        inputs: `jax.Array` that is already sharded to a local process size.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        Distributed inputs thats been properly put to local devices.
    """
    raise NotImplementedError()
    # if not isinstance(layout, jax.sharding.Sharding):
    #     layout = _to_torch_placements(layout)
    # if layout.is_fully_addressable:
    #     return jax.device_put(inputs, layout)

    # # We need the jax mesh information to determine how to place the data
    # # on to each of the worker.
    # jax_mesh = layout.mesh
    # mesh_rank = len(jax_mesh.shape)
    # per_process_batch_size = inputs.shape[0]
    # if mesh_rank == 1:
    #     # This is data parallel mesh only. We will split the full data
    #     # across the batch dim.
    #     num_split = jax.local_device_count()
    #     per_replica_batch_size = per_process_batch_size // num_split
    #     if per_process_batch_size % per_replica_batch_size != 0:
    #         raise ValueError(
    #             f"The local batch size {per_process_batch_size} is not"
    #             "divisible by the number of local replicas "
    #             f"{num_split}"
    #         )
    #     global_batch_size = per_process_batch_size * jax.process_count()
    #     per_replica_batches = jax.numpy.split(inputs, num_split, axis=0)
    # elif mesh_rank == 2:
    #     # Data+Model parallel
    #     # In this case, we need to check if the mesh batch dim shape is large
    #     # than number of local devices, so that we can decide whether a split
    #     # is needed for the data, or a repeat/copy of the data is needed for
    #     # each of the device.
    #     # TODO(scottzhu): The mesh batch dim name is not available here, since
    #     # we only have jax Mesh. We assume the first dim is for batch, and
    #     # second dim is for model for now.
    #     mesh_batch_dim_size = list(jax_mesh.shape.values())[0]
    #     local_device_count = jax.local_device_count()
    #     if mesh_batch_dim_size < local_device_count:
    #         # No split needed, we only need to repeat here.
    #         global_batch_size = per_process_batch_size
    #         per_replica_batches = [inputs for _ in range(local_device_count)]
    #     else:
    #         # Note that global batch size is not simply per_process_batch_size *
    #         # num_process. It actually depends on the model dim size.
    #         global_batch_size = per_process_batch_size * (
    #             mesh_batch_dim_size // local_device_count
    #         )
    #         per_replica_batches = jax.numpy.split(inputs, local_device_count, axis=0)
    # else:
    #     raise ValueError(
    #         "Only 1D or 2D mesh is supported at the moment. "
    #         f"Received mesh shape = {jax_mesh.shape}"
    #     )

    # global_shape = (global_batch_size,) + inputs.shape[1:]
    # global_batch_array = jax.make_array_from_single_device_arrays(
    #     global_shape,
    #     layout,
    #     arrays=[
    #         jax.device_put(batch, device)
    #         for batch, device in zip(per_replica_batches, layout.addressable_devices)
    #     ],
    # )
    # return global_batch_array


def initialize(job_addresses, num_processes, process_id):
    if job_addresses and "," in job_addresses:
        # When user provide all the job addresses, we will split and get the
        # first one, which is the coordinator.
        job_addresses = job_addresses.split(",")
        # Do a sanity check to make sure the number of addresses also match
        # the num_processes.
        if num_processes is not None and num_processes != len(job_addresses):
            raise ValueError(
                f"The provided job_addresses {job_addresses} has "
                f"{len(job_addresses)} jobs, but num_processes is "
                f"{num_processes}"
            )
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{coordinator_address}",
        world_size=num_processes,
        rank=process_id,
    )


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return dist.get_world_size()


def process_id():
    """Return the current process ID for the distribution setting."""
    return dist.get_rank()


def _to_torch_device(device_id):
    if isinstance(device_id, int):
        return device_id
    return int(device_id.split(":")[-1])


def _to_torch_mesh(device_mesh):
    """Convert the DeviceMesh to JAX backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `jax.sharding.Mesh` instance.
    """
    shape = device_mesh.devices.shape
    devices = [_to_torch_device(d) for d in device_mesh.devices.flatten()]
    devices = np.array(devices).reshape(shape)
    return torch_mesh.DeviceMesh(
        device_type="cuda", mesh=devices, mesh_dim_names=device_mesh.axis_names
    )


def _to_torch_placements(tensor_layout):
    """Convert the TensorLayout to JAX backend specific Sharding.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `jax.sharding.NamedSharding` instance.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set " "for TensorLayout."
        )

    assert len(set(tensor_layout.axes)) == len(
        tensor_layout.axes
    ), "Torch does not support sharding over the same axis multiple times."

    placements = [dist_tensor.Replicate()] * len(tensor_layout.device_mesh.axis_names)
    for i, axis in enumerate(tensor_layout.axes):
        if axis is None:
            continue

        if axis not in tensor_layout.device_mesh.axis_names:
            raise ValueError(
                f"Axis {axis} is not in the device mesh axis names: "
                f"{tensor_layout.device_mesh.axis_names}"
            )

        axis_index = tensor_layout.device_mesh.axis_names.index(axis)
        placements[axis_index] = dist_tensor.Shard(i)

    return placements
