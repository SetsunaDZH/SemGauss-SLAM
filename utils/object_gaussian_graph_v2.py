"""Object-centric Gaussian scene graph layer using current SLAM camera poses.

Compared with the first auxiliary implementation, this version avoids using the
canonical ``curr_data['w2c']`` for Gaussian-object association.  SemGauss-SLAM
stores the current estimated camera pose in ``params['cam_unnorm_rots']`` and
``params['cam_trans']``; therefore object binding must project Gaussians with the
pose of the current frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation


@dataclass
class ObjectCandidate:
    class_id: int
    mask: torch.Tensor
    score: float
    area: int
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class ObjectNode:
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
    src_id: int
    dst_id: int
    relation_type: str
    confidence: float


class ObjectGaussianGraph:
    """Lightweight object-node and relation layer for SemGauss-SLAM."""

    def __init__(self, config: Optional[dict] = None, device: str | torch.device = "cuda") -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.device = torch.device(device)

        self.max_objects = int(cfg.get("max_objects", 256))
        self.assign_threshold = float(cfg.get("assign_threshold", 0.30))
        self.min_component_area = int(cfg.get("min_component_area", 80))
        self.min_gaussians_per_object = int(cfg.get("min_gaussians_per_object", 10))
        self.ignore_class_ids = set(int(x) for x in cfg.get("ignore_class_ids", [0, 255]))

        self.depth_sigma = float(cfg.get("depth_sigma", 0.10))
        self.lambda_depth = float(cfg.get("lambda_depth", 0.05))
        self.unknown_energy = float(cfg.get("unknown_energy", 3.0))
        self.default_energy = float(cfg.get("default_energy", 8.0))
        self.forget = float(cfg.get("forget", 0.01))
        self.match_class_bonus = float(cfg.get("match_class_bonus", 0.25))

        self.anchor_weight = float(cfg.get("anchor_weight", 1.0e-4))
        self.graph_weight = float(cfg.get("graph_weight", 5.0e-5))
        self.relation_min_confidence = float(cfg.get("relation_min_confidence", 0.5))
        self.adjacent_distance = float(cfg.get("adjacent_distance", 0.05))
        self.support_vertical_gap = float(cfg.get("support_vertical_gap", 0.08))
        self.support_min_overlap = float(cfg.get("support_min_overlap", 0.05))
        self.huber_delta = float(cfg.get("huber_delta", 0.05))

        self.objects: List[ObjectNode] = []
        self.edges: List[RelationEdge] = []
        self.assignment_probs: Optional[torch.Tensor] = None
        self.local_means: Optional[torch.Tensor] = None
        self.initialized = False
        self.debug_stats: Dict[str, float | int] = {}

    def initialize(self, num_gaussians: int) -> None:
        if not self.enabled:
            return
        self.assignment_probs = torch.zeros(num_gaussians, self.max_objects, device=self.device)
        self.assignment_probs[:, 0] = 1.0
        self.local_means = torch.zeros(num_gaussians, 3, device=self.device)
        self.objects = []
        self.edges = []
        self.initialized = True

    def expand_gaussians(self, new_num_gaussians: int) -> None:
        if not self.enabled:
            return
        if self.assignment_probs is None:
            self.initialize(new_num_gaussians)
            return
        old_n = int(self.assignment_probs.shape[0])
        if new_num_gaussians <= old_n:
            return
        add_n = new_num_gaussians - old_n
        extra = torch.zeros(add_n, self.max_objects, device=self.device)
        extra[:, 0] = 1.0
        self.assignment_probs = torch.cat([self.assignment_probs, extra], dim=0)
        if self.local_means is None:
            self.local_means = torch.zeros(new_num_gaussians, 3, device=self.device)
        else:
            self.local_means = torch.cat([self.local_means, torch.zeros(add_n, 3, device=self.device)], dim=0)

    @torch.no_grad()
    def extract_candidates_from_semantic(self, sem_out: torch.Tensor) -> List[ObjectCandidate]:
        if not self.enabled:
            return []
        if sem_out.dim() == 4:
            prob = torch.softmax(sem_out.detach(), dim=1)
            pred = torch.argmax(prob, dim=1).squeeze(0)
            conf = torch.max(prob, dim=1).values.squeeze(0)
        elif sem_out.dim() == 3:
            prob = torch.softmax(sem_out.detach().unsqueeze(0), dim=1).squeeze(0)
            pred = torch.argmax(prob, dim=0)
            conf = torch.max(prob, dim=0).values
        elif sem_out.dim() == 2:
            pred = sem_out.detach().long()
            conf = torch.ones_like(pred, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported semantic tensor shape: {tuple(sem_out.shape)}")

        pred_np = pred.cpu().numpy().astype(np.int32)
        conf_np = conf.cpu().numpy().astype(np.float32)
        candidates: List[ObjectCandidate] = []
        for class_id in np.unique(pred_np):
            class_id = int(class_id)
            if class_id in self.ignore_class_ids:
                continue
            binary = (pred_np == class_id).astype(np.uint8)
            if int(binary.sum()) < self.min_component_area:
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < self.min_component_area:
                    continue
                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                w = int(stats[label_id, cv2.CC_STAT_WIDTH])
                h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                comp = labels == label_id
                mask = torch.from_numpy(comp).to(self.device).bool()
                candidates.append(
                    ObjectCandidate(
                        class_id=class_id,
                        mask=mask,
                        score=float(conf_np[comp].mean()) if area > 0 else 0.0,
                        area=area,
                        bbox_xyxy=(x, y, x + w, y + h),
                    )
                )
        return candidates

    def _create_object(self, candidate: ObjectCandidate) -> int:
        obj_id = len(self.objects) + 1
        if obj_id >= self.max_objects:
            return 0
        node = ObjectNode(object_id=obj_id, class_id=candidate.class_id)
        node.class_prob[candidate.class_id] = candidate.score
        node.confidence = candidate.score
        node.num_observations = 1
        self.objects.append(node)
        return obj_id

    def _match_or_create_object(self, candidate: ObjectCandidate) -> int:
        # Conservative first version: one stable object node per semantic class.
        best_score = -1.0
        best_id = 0
        for obj in self.objects:
            score = self.match_class_bonus if obj.class_id == candidate.class_id else 0.0
            score += 0.01 * min(obj.num_observations, 10)
            if score > best_score:
                best_score = score
                best_id = obj.object_id
        if best_score < self.match_class_bonus:
            return self._create_object(candidate)
        obj = self.objects[best_id - 1]
        obj.num_observations += 1
        obj.class_prob[candidate.class_id] = max(obj.class_prob.get(candidate.class_id, 0.0), candidate.score)
        obj.confidence = min(1.0, obj.confidence + 0.05 * candidate.score)
        return best_id

    @staticmethod
    def current_frame_w2c(params: Dict[str, torch.Tensor], time_idx: int) -> torch.Tensor:
        """Build the current estimated world-to-camera pose from SLAM parameters."""
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        cam_tran = params["cam_trans"][..., time_idx].detach()
        w2c = torch.eye(4, device=cam_tran.device, dtype=cam_tran.dtype)
        w2c[:3, :3] = build_rotation(cam_rot)
        w2c[:3, 3] = cam_tran
        return w2c

    @staticmethod
    def project_gaussians(
        means3d: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ones = torch.ones((means3d.shape[0], 1), device=means3d.device, dtype=means3d.dtype)
        pts_h = torch.cat([means3d, ones], dim=1)
        pts_cam = (w2c.to(means3d.device) @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2]
        z_safe = z.clamp_min(1e-6)
        intr = intrinsics.to(means3d.device)
        u = intr[0, 0] * pts_cam[:, 0] / z_safe + intr[0, 2]
        v = intr[1, 1] * pts_cam[:, 1] / z_safe + intr[1, 2]
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        return u.long(), v.long(), z, valid

    @torch.no_grad()
    def update_from_semantic(
        self,
        params: Dict[str, torch.Tensor],
        curr_data: Dict[str, torch.Tensor],
        sem_out: torch.Tensor,
        time_idx: int,
    ) -> Dict[str, int | float]:
        if not self.enabled:
            return {"num_candidates": 0, "num_objects": 0}
        n = int(params["means3D"].shape[0])
        if not self.initialized or self.assignment_probs is None:
            self.initialize(n)
        else:
            self.expand_gaussians(n)

        candidates = self.extract_candidates_from_semantic(sem_out)
        depth = curr_data["depth"]
        depth_img = depth[0] if depth.dim() == 3 else depth
        height, width = int(depth_img.shape[-2]), int(depth_img.shape[-1])

        if len(candidates) == 0:
            self.debug_stats = {"num_candidates": 0, "num_valid_projected": 0, "num_inside_mask": 0, "num_assigned": 0}
            return {"num_candidates": 0, "num_objects": len(self.objects)}

        w2c = self.current_frame_w2c(params, int(time_idx))
        u, v, z, valid = self.project_gaussians(
            params["means3D"].detach(), w2c, curr_data["intrinsics"].detach(), height, width
        )
        valid_idx = torch.where(valid)[0]

        if self.forget > 0:
            self.assignment_probs.mul_(1.0 - self.forget)
            self.assignment_probs[:, 0].add_(self.forget)

        energy = torch.full_like(self.assignment_probs, fill_value=self.default_energy)
        energy[:, 0] = self.unknown_energy
        inside_total = 0

        if valid_idx.numel() > 0:
            uu = u[valid_idx].clamp(0, width - 1)
            vv = v[valid_idx].clamp(0, height - 1)
            for cand in candidates:
                obj_id = self._match_or_create_object(cand)
                if obj_id <= 0 or obj_id >= self.max_objects:
                    continue
                mask_values = cand.mask[vv, uu]
                idx = valid_idx[mask_values]
                if idx.numel() == 0:
                    continue
                inside_total += int(idx.numel())
                obs_depth = depth_img[v[idx].clamp(0, height - 1), u[idx].clamp(0, width - 1)].to(self.device)
                valid_depth = obs_depth > 0
                idx = idx[valid_depth]
                if idx.numel() == 0:
                    continue
                obs_depth = obs_depth[valid_depth]
                depth_err = torch.abs(obs_depth - z[idx]) / max(self.depth_sigma, 1e-6)
                e_val = self.lambda_depth * torch.clamp(depth_err, max=10.0)
                energy[idx, obj_id] = torch.minimum(energy[idx, obj_id], e_val)

        posterior = self.assignment_probs * torch.exp(-energy)
        posterior = posterior / (posterior.sum(dim=1, keepdim=True) + 1e-8)
        self.assignment_probs = posterior.detach()
        self.update_object_geometry(params)
        self.update_relations()

        if len(self.objects) > 0:
            assigned_mask = torch.max(self.assignment_probs[:, 1 : len(self.objects) + 1], dim=1).values > self.assign_threshold
            num_assigned = int(assigned_mask.sum().item())
            max_prob = float(torch.max(self.assignment_probs[:, 1 : len(self.objects) + 1]).item())
        else:
            num_assigned = 0
            max_prob = 0.0
        self.debug_stats = {
            "num_candidates": len(candidates),
            "num_valid_projected": int(valid_idx.numel()),
            "num_inside_mask": int(inside_total),
            "num_assigned": num_assigned,
            "max_object_prob": max_prob,
        }
        return {"num_candidates": len(candidates), "num_objects": len(self.objects), **self.debug_stats}

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
                obj.centroid = None
                obj.aabb_min = None
                obj.aabb_max = None
                continue
            pts = xyz[mask]
            weights = probs[mask].unsqueeze(1)
            centroid = (weights * pts).sum(dim=0) / (weights.sum() + 1e-8)
            obj.gaussian_indices = torch.where(mask)[0]
            obj.centroid = centroid.detach()
            obj.aabb_min = pts.min(dim=0).values.detach()
            obj.aabb_max = pts.max(dim=0).values.detach()
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
        self.edges = []
        valid_objects = [o for o in self.objects if o.aabb_min is not None and o.aabb_max is not None and o.centroid is not None]
        for i, obj_a in enumerate(valid_objects):
            for obj_b in valid_objects[i + 1 :]:
                d = self._aabb_distance(obj_a.aabb_min, obj_a.aabb_max, obj_b.aabb_min, obj_b.aabb_max)
                if float(d.item()) < self.adjacent_distance:
                    self.edges.append(RelationEdge(obj_a.object_id, obj_b.object_id, "adjacent", 0.5))
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
            target = obj.centroid.to(xyz.device) + self.local_means[idx].to(xyz.device)
            weights = self.assignment_probs[idx, obj.object_id].to(xyz.device).detach().unsqueeze(1)
            loss = loss + (weights * (xyz[idx] - target).pow(2)).mean()
            count += 1
        return loss * 0.0 if count == 0 else self.anchor_weight * loss / count

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
            if obj_a is None or obj_b is None or obj_a.aabb_min is None or obj_b.aabb_min is None:
                continue
            a_min, a_max = obj_a.aabb_min.to(params["means3D"].device), obj_a.aabb_max.to(params["means3D"].device)
            b_min, b_max = obj_b.aabb_min.to(params["means3D"].device), obj_b.aabb_max.to(params["means3D"].device)
            if edge.relation_type == "adjacent":
                loss = loss + edge.confidence * self._huber(self._aabb_distance(a_min, a_max, b_min, b_max))
                count += 1
            elif edge.relation_type == "support":
                gap_z = b_min[2] - a_max[2]
                ov_x = self._overlap_1d(a_min[0], a_max[0], b_min[0], b_max[0])
                ov_y = self._overlap_1d(a_min[1], a_max[1], b_min[1], b_max[1])
                loss = loss + edge.confidence * (self._huber(gap_z) + self._huber(1.0 - ov_x * ov_y))
                count += 1
        return loss * 0.0 if count == 0 else self.graph_weight * loss / count

    def compute_losses(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "obj_anchor": self.compute_object_anchor_loss(params),
            "graph": self.compute_graph_relation_loss(params),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "num_objects": len(self.objects),
            "num_edges": len(self.edges),
            "debug_stats": self.debug_stats,
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
