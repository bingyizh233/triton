from __future__ import annotations

from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr, int32, tensor

__all__ = ["arrive", "wait", "barrier", "cluster_cta_id"]


@builtin
def arrive(relaxed: bool = False, _semantic=None):
    """
    Arrive at a barrier that synchronizes across the CTA cluster.

    Args:
        relaxed (bool): Whether to use relaxed semantics. Defaults to False.
    """
    relaxed = _unwrap_if_constexpr(relaxed)
    _semantic.builder.create_cluster_arrive(relaxed)


@builtin
def wait(_semantic=None):
    """
    Wait for all CTAs in the cluster to arrive at the cluster barrier.
    """
    _semantic.builder.create_cluster_wait()


@builtin
def barrier(relaxed: bool = False, _semantic=None):
    """
    Barrier that synchronizes across the CTA cluster.

    Args:
        relaxed (bool): Whether to use relaxed arrival semantics. Defaults to
            False.
    """
    relaxed = _unwrap_if_constexpr(relaxed)
    _semantic.builder.create_cluster_barrier(relaxed)


@builtin
def cluster_cta_id(_semantic=None):
    """
    Return the index of this CTA within the cluster (0 .. num_ctas - 1).

    For single-CTA kernels this is always 0. When ``num_ctas > 1``,
    ``program_id`` identifies the cluster, not the CTA; use this to shard work
    or shared memory per CTA inside the cluster.
    """
    handle = _semantic.builder.create_cluster_cta_id()
    return tensor(handle, int32)
