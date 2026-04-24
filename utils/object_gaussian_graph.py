"""Object-centric Gaussian scene graph utilities.

This module is intentionally lightweight.  It does not depend on SAM, CLIP, VLMs,
or heavyweight instance-segmentation backbones.  The first implementation uses
semantic logits/probabilities already produced by SemGauss-SLAM and converts them
into connected-component object candidates.  These candidates are then fused into
3D Gaussian object nodes through multi-frame geometric consistency.

The module is designed as an auxiliary layer on top of the original
per-Gaussian semantic representation.  It can be disabled without changing the
original SLAM behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ObjectCandidate:
    """A lightweight 2D object observation extracted from a semantic map."""

    class_id: int
    mask: torch.Tensor  # bool tensor, shape [H, W], CUDA or CPU
    score: float
    area: int
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class ObjectNode:
    """Object-level node maintained above a set of 3D Gaussians."""

    object_id: int
    class_id: int
    class_prob: Dict[int, float] = field(default_factory=dict)
    gaussian_indices: Optional[torch.Tensor] = None
    centroid: Optional[torch.Tensor] = None
    aabb_min: Optional[torch.Tensor] = None
    aabb_max: Optional[torch.Tensor] = None
    confidence: float = 0.0
    num_observations: int = 0


@dataclass
class RelationEdge:
    """Topological or structural relation between two object nodes."""

    src_id: int
    dst_id: int
    relation_type: str
    confidence: float


class ObjectGaussianGraph:
    """Maintain object nodes, Gaussian memberships, and scene-graph losses.

    The class follows a deliberately conservative design:
    - Gaussian-to-object assignments are updated outside autograd.
    - Object graph losses are used only as weak regularizers.
    - Relation factors are activated only after object nodes are stable enough.
    """

    def __init__(self, config: Optional[dict] = None, device: str | torch.device = "cuda") -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.device = torch.device(device)

        self.max_objects = int(cfg.get("max_objects", 256))
        self.assign_threshold = float(cfg.get("assign_threshold", 0.55))
        self.min_component_area = int(cfg.get("min_component_area", 80))
        self.min_gaussians_per_object = int(cfg.get("min_gaussians_per_object", 30))
        self.keyframe_only = bool(cfg.get("keyframe_only", True))

        self.lambda_mask = float(cfg.get("lambda_mask", 1.0))
        self.lambda_depth = float(cfg.get("lambda_depth", 0.1))
        self.lambda_semantic = float(cfg.get("lambda_semantic", 0.5))
        self.depth_sigma = float(cfg.get("depth_sigma", 0.05))
        self.match_iou_threshold = float(cfg.get("match_iou_threshold", 0.10))
        self.match_class_bonus = float(cfg.get("match_class_bonus", 0.25))
        self.forget = float(cfg.get("forget", 0.01))

        self.anchor_weight = float(cfg.get("anchor_weight", 1.0))
        self.graph_weight = float(cfg.get("graph_weight", 1.0))
        self.relation_min_confidence = float(cfg.get("relation_min_confidence", 0.5))
        self.adjacent_distance = float(cfg.get("adjacent_distance", 0.05))
        self.support_vertical_gap = float(cfg.get("support_vertical_gap", 0.08))
        self.support_min_overlap = float(cfg.get("support_min_overlap", 0.05))
        self.huber_delta = float(cfg.get("huber_delta", 0.05))

        # Optional class filtering.  Empty means all non-zero labels are considered.
        self.ignore_class_ids = set(int(x) for x in cfg.get("ignore_class_ids", [0, 255]))

        self.objects: List[ObjectNode] = []
        self.edges: List[RelationEdge] = []
        self.assignment_probs: Optional[torch.Tensor] = None  # [N, max_objects]
        self.local_means: Optional[torch.Tensor] = None  # per-Gaussian anchor coordinate in object frame approximation
        self.initialized = False

    @property
    def num_gaussians(self) -> int:
        if self.assignment_probs is None:
            return 0
        return int(self.assignment_probs.shape[0])

    def initialize(self, num_gaussians: int) -> None:
        """Initialize assignment storage after the first Gaussian map is created."""
        if not self.enabled:
            return
        self.assignment_probs = torch.zeros(num_gaussians, self.max_objects, device=self.device)
        # Column 0 is a conservative unknown/background bin until object nodes are created.
        self.assignment_probs[:, 0] = 1.0
        self.local_means = torch.zeros(num_gaussians, 3, device=self.device)
        self.objects = []
        self.edges = []
        self.initialized = True

    def expand_gaussians(self, new_num_gaussians: int) -> None:
        """Synchronize assignments when the SLAM system adds new Gaussians."""
        if not self.enabled:
            return
        if self.assignment_probs is None:
            self.initialize(new_num_gaussians)
            return
        old_n = self.assignment_probs.shape[0]
        if new_num_gaussians <= old_n:
            return
        add_n = new_num_gaussians - old_n
        new_probs = torch.zeros(add_n, self.max_objects, device=self.device)
        new_probs[:, 0] = 1.0
        self.assignment_probs = torch.cat([self.assignment_probs, new_probs], dim=0)
        if self.local_means is None:
            self.local_means = torch.zeros(new_num_gaussians, 3, device=self.device)
        else:
            self.local_means = torch.cat([self.local_means, torch.zeros(add_n, 3, device=self.device)], dim=0)

    @torch.no_grad()
    def extract_candidates_from_semantic(
        self,
        sem_out: torch.Tensor,
        min_area: Optional[int] = None,
    ) -> List[ObjectCandidate]:
        """Convert semantic logits/probabilities into connected-component candidates.

        Args:
            sem_out: tensor with shape [1, C, H, W], [C, H, W], or [H, W].
            min_area: optional area threshold overriding the configured value.
        """
        if not self.enabled:
            return []
        area_thres = int(min_area or self.min_component_area)

        if sem_out.dim() == 4:
            sem_prob = torch.softmax(sem_out.detach(), dim=1)
            sem_pred = torch.argmax(sem_prob, dim=1).squeeze(0)
            conf = torch.max(sem_prob, dim=1).values.squeeze(0)
        elif sem_out.dim() == 3:
            sem_prob = torch.softmax(sem_out.detach().unsqueeze(0), dim=1).squeeze(0)
            sem_pred = torch.argmax(sem_prob, dim=0)
            conf = torch.max(sem_prob, dim=0).values
        elif sem_out.dim() == 2:
            sem_pred = sem_out.detach().long()
            conf = torch.ones_like(sem_pred, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported semantic tensor shape: {tuple(sem_out.shape)}")

        sem_pred_cpu = sem_pred.detach().cpu().numpy().astype(np.int32)
        conf_cpu = conf.detach().cpu().numpy().astype(np.float32)
        candidates: List[ObjectCandidate] = []

        for class_id in np.unique(sem_pred_cpu):
            class_id_int = int(class_id)
            if class_id_int in self.ignore_class_ids:
                continue
            class_mask = (sem_pred_cpu == class_id_int).astype(np.uint8)
            if int(class_mask.sum()) < area_thres:
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < area_thres:
                    continue
                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                w = int(stats[label_id, cv2.CC_STAT_WIDTH])
                h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                comp_np = labels == label_id
                score = float(conf_cpu[comp_np].mean()) if area > 0 else 0.0
                mask = torch.from_numpy(comp_np).to(self.device)
                candidates.append(
                    ObjectCandidate(
                        class_id=class_id_int,
                        mask=mask.bool(),
                        score=score,
                        area=area,
                        bbox_xyxy=(x, y, x + w, y + h),
                    )
                )
        return candidates

    @staticmethod
    def _bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _create_object(self, candidate: ObjectCandidate) -> int:
        obj_id = len(self.objects) + 1  # object id 0 is reserved for unknown/background
        if obj_id >= self.max_objects:
            return 0
        node = ObjectNode(object_id=obj_id, class_id=candidate.class_id)
        node.class_prob[candidate.class_id] = candidate.score
        node.confidence = candidate.score
        node.num_observations = 1
        self.objects.append(node)
        return obj_id

    def _match_or_create_object(self, candidate: ObjectCandidate) -> int:
        best_score = -1.0
        best_id = 0
        # Lightweight first version: class consistency dominates because current nodes
        # do not store 2D masks across views.  Geometry will refine after assignment.
        for obj in self.objects:
            score = 0.0
            if obj.class_id == candidate.class_id:
                score += self.match_class_bonus
            score += 0.01 * min(obj.num_observations, 10)
            if score > best_score:
                best_score = score
                best_id = obj.object_id
        if best_score < self.match_class_bonus:
            best_id = self._create_object(candidate)
        else:
            obj = self.objects[best_id - 1]
            obj.num_observations += 1
            old = obj.class_prob.get(candidate.class_id, 0.0)
            obj.class_prob[candidate.class_id] = max(old, candidate.score)
            obj.confidence = min(1.0, obj.confidence + 0.05 * candidate.score)
        return best_id

    @staticmethod
    def project_gaussians(
        means3d: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Gaussian centers into the current image."""
        ones = torch.ones((means3d.shape[0], 1), device=means3d.device, dtype=means3d.dtype)
        pts_h = torch.cat([means3d, ones], dim=1)
        pts_cam = (w2c.to(means3d.device) @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2]
        z_safe = z.clamp_min(1e-6)
        fx, fy = intrinsics[0, 0].to(means3d.device), intrinsics[1, 1].to(means3d.device)
        cx, cy = intrinsics[0, 2].to(means3d.device), intrinsics[1, 2].to(means3d.device)
        u = fx * pts_cam[:, 0] / z_safe + cx
        v = fy * pts_cam[:, 1] / z_safe + cy
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        return u.long(), v.long(), z, valid

    @torch.no_grad()
    def update_from_semantic(
        self,
        params: Dict[str, torch.Tensor],
        curr_data: Dict[str, torch.Tensor],
        sem_out: torch.Tensor,
    ) -> Dict[str, int]:
        """Update object candidates and Gaussian memberships from a keyframe."""
        if not self.enabled:
            return {"num_candidates": 0, "num_objects": 0}
        n = int(params["means3D"].shape[0])
        if not self.initialized or self.assignment_probs is None:
            self.initialize(n)
        else:
            self.expand_gaussians(n)

        candidates = self.extract_candidates_from_semantic(sem_out)
        if len(candidates) == 0:
            return {"num_candidates": 0, "num_objects": len(self.objects)}

        depth = curr_data["depth"]
        if depth.dim() == 3:
            depth_img = depth[0]
        else:
            depth_img = depth
        height, width = int(depth_img.shape[-2]), int(depth_img.shape[-1])
        u, v, z, valid = self.project_gaussians(
            params["means3D"].detach(), curr_data["w2c"].detach(), curr_data["intrinsics"].detach(), height, width
        )

        # Mild forgetting prevents early wrong assignments from becoming irreversible.
        if self.forget > 0:
            self.assignment_probs.mul_(1.0 - self.forget)
            self.assignment_probs[:, 0].add_(self.forget)

        energy = torch.full_like(self.assignment_probs, fill_value=8.0)
        energy[:, 0] = 2.0  # unknown bin remains possible but not dominant

        for cand in candidates:
            obj_id = self._match_or_create_object(cand)
            if obj_id <= 0 or obj_id >= self.max_objects:
                continue
            inside = torch.zeros(n, device=self.device, dtype=torch.bool)
            valid_idx = torch.where(valid)[0]
            if valid_idx.numel() == 0:
                continue
            uu = u[valid_idx].clamp(0, width - 1)
            vv = v[valid_idx].clamp(0, height - 1)
            cand_mask_vals = cand.mask[vv, uu]
            inside[valid_idx] = cand_mask_vals
            if inside.sum() == 0:
                continue
            obs_depth = depth_img[v[inside].clamp(0, height - 1), u[inside].clamp(0, width - 1)].to(self.device)
            depth_err = torch.abs(obs_depth - z[inside]) / max(self.depth_sigma, 1e-6)
            e_val = self.lambda_depth * torch.clamp(depth_err, max=10.0)
            energy[inside, obj_id] = torch.minimum(energy[inside, obj_id], e_val)

        posterior = self.assignment_probs * torch.exp(-energy)
        posterior = posterior / (posterior.sum(dim=1, keepdim=True) + 1e-8)
        self.assignment_probs = posterior.detach()
        self.update_object_geometry(params)
        self.update_relations()
        return {"num_candidates": len(candidates), "num_objects": len(self.objects)}

    @torch.no_grad()
    def update_object_geometry(self, params: Dict[str, torch.Tensor]) -> None:
        if not self.enabled or self.assignment_probs is None:
            return
        xyz = params["means3D"].detach()
        for obj in self.objects:
            probs = self.assignment_probs[:, obj.object_id]
            mask = probs > self.assign_threshold
            if int(mask.sum()) < self.min_gaussians_per_object:
                obj.gaussian_indices = None
                continue
            pts = xyz[mask]
            weights = probs[mask].unsqueeze(1)
            centroid = (weights * pts).sum(dim=0) / (weights.sum() + 1e-8)
            obj.gaussian_indices = torch.where(mask)[0]
            obj.centroid = centroid.detach()
            obj.aabb_min = pts.min(dim=0).values.detach()
            obj.aabb_max = pts.max(dim=0).values.detach()
            obj.confidence = min(1.0, 0.5 * obj.confidence + 0.5 * float(mask.float().mean().item()))
            if self.local_means is not None:
                self.local_means[mask] = (pts - centroid).detach()

    @staticmethod
    def _aabb_distance(a_min: torch.Tensor, a_max: torch.Tensor, b_min: torch.Tensor, b_max: torch.Tensor) -> torch.Tensor:
        gap = torch.maximum(torch.maximum(a_min - b_max, b_min - a_max), torch.zeros_like(a_min))
        return torch.linalg.norm(gap)

    @staticmethod
    def _overlap_1d(a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
        inter = torch.clamp(torch.minimum(a1, b1) - torch.maximum(a0, b0), min=0.0)
        denom = torch.clamp(torch.minimum(a1 - a0, b1 - b0), min=1e-6)
        return inter / denom

    @torch.no_grad()
    def update_relations(self) -> None:
        """Generate lightweight adjacent/support relation candidates from AABBs."""
        self.edges = []
        valid_objects = [o for o in self.objects if o.aabb_min is not None and o.aabb_max is not None]
        for i, obj_a in enumerate(valid_objects):
            for obj_b in valid_objects[i + 1 :]:
                d = self._aabb_distance(obj_a.aabb_min, obj_a.aabb_max, obj_b.aabb_min, obj_b.aabb_max)
                if float(d.item()) < self.adjacent_distance:
                    self.edges.append(RelationEdge(obj_a.object_id, obj_b.object_id, "adjacent", 0.5))

                # z-axis is assumed to be gravity-opposite for Replica/ScanNet style coordinates.
                gap_ab = obj_b.aabb_min[2] - obj_a.aabb_max[2]
                ov_x = self._overlap_1d(obj_a.aabb_min[0], obj_a.aabb_max[0], obj_b.aabb_min[0], obj_b.aabb_max[0])
                ov_y = self._overlap_1d(obj_a.aabb_min[1], obj_a.aabb_max[1], obj_b.aabb_min[1], obj_b.aabb_max[1])
                ov = ov_x * ov_y
                if abs(float(gap_ab.item())) < self.support_vertical_gap and float(ov.item()) > self.support_min_overlap:
                    if float(obj_a.centroid[2].item()) < float(obj_b.centroid[2].item()):
                        self.edges.append(RelationEdge(obj_a.object_id, obj_b.object_id, "support", 0.6))
                    else:
                        self.edges.append(RelationEdge(obj_b.object_id, obj_a.object_id, "support", 0.6))

    def _huber(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.tensor(self.huber_delta, device=x.device, dtype=x.dtype)
        abs_x = torch.abs(x)
        return torch.where(abs_x <= delta, 0.5 * x * x, delta * (abs_x - 0.5 * delta))

    def compute_object_anchor_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.enabled or self.assignment_probs is None or self.local_means is None:
            return params["means3D"].sum() * 0.0
        xyz = params["means3D"]
        loss = xyz.sum() * 0.0
        count = 0
        for obj in self.objects:
            if obj.centroid is None or obj.gaussian_indices is None:
                continue
            idx = obj.gaussian_indices
            if idx.numel() < self.min_gaussians_per_object:
                continue
            target = obj.centroid.to(xyz.device) + self.local_means[idx].to(xyz.device)
            weights = self.assignment_probs[idx, obj.object_id].to(xyz.device).detach().unsqueeze(1)
            loss = loss + (weights * (xyz[idx] - target).pow(2)).mean()
            count += 1
        if count == 0:
            return xyz.sum() * 0.0
        return self.anchor_weight * loss / count

    def _get_object(self, object_id: int) -> Optional[ObjectNode]:
        if object_id <= 0 or object_id > len(self.objects):
            return None
        return self.objects[object_id - 1]

    def compute_graph_relation_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.enabled:
            return params["means3D"].sum() * 0.0
        loss = params["means3D"].sum() * 0.0
        count = 0
        for edge in self.edges:
            if edge.confidence < self.relation_min_confidence:
                continue
            obj_a = self._get_object(edge.src_id)
            obj_b = self._get_object(edge.dst_id)
            if obj_a is None or obj_b is None:
                continue
            if obj_a.aabb_min is None or obj_b.aabb_min is None:
                continue
            a_min, a_max = obj_a.aabb_min.to(params["means3D"].device), obj_a.aabb_max.to(params["means3D"].device)
            b_min, b_max = obj_b.aabb_min.to(params["means3D"].device), obj_b.aabb_max.to(params["means3D"].device)
            if edge.relation_type == "adjacent":
                d = self._aabb_distance(a_min, a_max, b_min, b_max)
                loss = loss + edge.confidence * self._huber(d)
                count += 1
            elif edge.relation_type == "support":
                gap_z = b_min[2] - a_max[2]
                ov_x = self._overlap_1d(a_min[0], a_max[0], b_min[0], b_max[0])
                ov_y = self._overlap_1d(a_min[1], a_max[1], b_min[1], b_max[1])
                overlap = ov_x * ov_y
                loss = loss + edge.confidence * (self._huber(gap_z) + self._huber(1.0 - overlap))
                count += 1
        if count == 0:
            return params["means3D"].sum() * 0.0
        return self.graph_weight * loss / count

    def compute_losses(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "obj_anchor": self.compute_object_anchor_loss(params),
            "graph": self.compute_graph_relation_loss(params),
        }

    def state_dict(self) -> Dict[str, object]:
        """Return a lightweight serializable graph summary for debugging."""
        return {
            "enabled": self.enabled,
            "num_objects": len(self.objects),
            "num_edges": len(self.edges),
            "objects": [
                {
                    "object_id": o.object_id,
                    "class_id": o.class_id,
                    "num_observations": o.num_observations,
                    "confidence": o.confidence,
                    "num_gaussians": 0 if o.gaussian_indices is None else int(o.gaussian_indices.numel()),
                }
                for o in self.objects
            ],
            "edges": [edge.__dict__ for edge in self.edges],
        }
