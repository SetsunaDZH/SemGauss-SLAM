from configs.replica.replica import config

# Use a separate output folder name so the object-graph experiment does not
# overwrite the original SemGauss-SLAM results.
config = dict(config)
config["run_name"] = f"{config['run_name']}_og"

# Lightweight object-centric Gaussian scene graph layer.
#
# The first version does not introduce a heavy instance segmentation network.
# It converts the existing SemGauss semantic output into connected-component
# object candidates and fuses them with the 3D Gaussian map through projection,
# depth consistency, and multi-frame assignment probabilities.
config["object_graph"] = dict(
    enabled=True,
    max_objects=256,

    # 2D semantic connected-component proposal settings.
    min_component_area=120,
    ignore_class_ids=[0, 255],

    # Gaussian-object association settings.
    assign_threshold=0.55,
    min_gaussians_per_object=30,
    lambda_mask=1.0,
    lambda_depth=0.1,
    lambda_semantic=0.5,
    depth_sigma=0.05,
    forget=0.01,

    # Object matching.  This first implementation is intentionally conservative
    # and mainly relies on semantic-class consistency before geometry becomes stable.
    match_iou_threshold=0.10,
    match_class_bonus=0.25,

    # Weak object and graph regularization terms.  Keep them small initially.
    anchor_weight=1.0e-4,
    graph_weight=5.0e-5,
    relation_min_confidence=0.5,

    # Geometry relation thresholds for adjacent/support edges.
    adjacent_distance=0.05,
    support_vertical_gap=0.08,
    support_min_overlap=0.05,
    huber_delta=0.05,
)
