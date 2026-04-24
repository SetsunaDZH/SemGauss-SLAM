"""Run SemGauss-SLAM with an object-centric Gaussian scene graph layer.

This wrapper keeps the original ``sem_gauss.py`` untouched.  It monkey-patches the
loss function before calling the original dense SLAM entry point.  The object
layer is deliberately optional: if ``config['object_graph']['enabled']`` is
False or missing, the behavior is the same as the original SemGauss-SLAM.

Usage:
    python sem_gauss_og.py configs/replica/replica_og.py
"""

from __future__ import annotations

import argparse
import json
import os
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, Set

import torch

import sem_gauss as base_slam
from utils.common_utils import seed_everything
from utils.object_gaussian_graph import ObjectGaussianGraph


_ORIGINAL_GET_LOSS = base_slam.get_loss
_OBJ_GRAPH: ObjectGaussianGraph | None = None
_CONFIG: Dict[str, Any] | None = None
_UPDATED_FRAME_IDS: Set[int] = set()


def _get_graph() -> ObjectGaussianGraph | None:
    global _OBJ_GRAPH, _CONFIG
    if _CONFIG is None:
        return None
    graph_cfg = _CONFIG.get("object_graph", {})
    if not graph_cfg.get("enabled", False):
        return None
    if _OBJ_GRAPH is None:
        _OBJ_GRAPH = ObjectGaussianGraph(graph_cfg, device=_CONFIG.get("primary_device", "cuda"))
    return _OBJ_GRAPH


def _maybe_update_graph(params, curr_data, mapping: bool, BA: bool) -> None:
    """Update object nodes once per frame from the current semantic observation."""
    graph = _get_graph()
    if graph is None or not (mapping or BA):
        return
    frame_id = int(curr_data.get("id", -1))
    if frame_id in _UPDATED_FRAME_IDS:
        return
    if "se" not in curr_data:
        return
    graph.update_from_semantic(params, curr_data, curr_data["se"])
    _UPDATED_FRAME_IDS.add(frame_id)


def get_loss_with_object_graph(
    params,
    curr_data,
    variables,
    iter_time_idx,
    loss_weights,
    use_sil_for_loss,
    sil_thres,
    use_l1,
    ignore_outlier_depth_loss,
    seg_net,
    tracking=False,
    mapping=False,
    BA=False,
    plot_dir=None,
    visualize_tracking_loss=False,
):
    """Wrapper around the original loss to add weak object/graph regularizers."""
    _maybe_update_graph(params, curr_data, mapping=mapping, BA=BA)

    result = _ORIGINAL_GET_LOSS(
        params,
        curr_data,
        variables,
        iter_time_idx,
        loss_weights,
        use_sil_for_loss,
        sil_thres,
        use_l1,
        ignore_outlier_depth_loss,
        seg_net,
        tracking=tracking,
        mapping=mapping,
        BA=BA,
        plot_dir=plot_dir,
        visualize_tracking_loss=visualize_tracking_loss,
    )

    # Some BA calls in the original code return rendered tensors directly.
    if not isinstance(result, tuple) or len(result) != 3:
        return result
    loss, variables, weighted_losses = result
    if not (mapping or BA):
        return result

    graph = _get_graph()
    if graph is None:
        return result

    graph_losses = graph.compute_losses(params)
    obj_anchor = graph_losses["obj_anchor"]
    graph_rel = graph_losses["graph"]
    loss = loss + obj_anchor + graph_rel
    weighted_losses["obj_anchor"] = obj_anchor
    weighted_losses["graph"] = graph_rel
    weighted_losses["loss"] = loss
    return loss, variables, weighted_losses


def run(config: Dict[str, Any]) -> None:
    global _CONFIG, _OBJ_GRAPH, _UPDATED_FRAME_IDS
    _CONFIG = config
    _OBJ_GRAPH = None
    _UPDATED_FRAME_IDS = set()

    base_slam.get_loss = get_loss_with_object_graph
    seed_everything(seed=config["seed"])
    base_slam.dense_semantic_slam(config)

    graph = _get_graph()
    if graph is not None:
        output_dir = os.path.join(config["workdir"], config["run_name"])
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "object_graph_summary.json"), "w", encoding="utf-8") as f:
            json.dump(graph.state_dict(), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="SemGauss-SLAM with object Gaussian scene graph regularization")
    parser.add_argument("experiment", type=str, help="Path to a config file, e.g. configs/replica/replica_og.py")
    args = parser.parse_args()

    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    config = experiment.config
    run(config)


if __name__ == "__main__":
    main()
