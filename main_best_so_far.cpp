/*
 * qem_voxel_map.cpp
 *
 * Phase 2 of the staged QEM-based mesh-reconstruction pipeline:
 *   voxel-anchored, incremental QEM map for registered LiDAR point clouds.
 *
 * Production-style sibling to qem_mesh.cpp. Shares its data layout:
 *   - PCD point clouds (binary or ASCII), one file per scan, sensor frame
 *   - Pose file in TUM, G2O/SLAM, or CSV format
 *   - PCD filenames contain a timestamp matched to a pose key '%010d_%09d'
 *
 * Pipeline per scan (mirrors process_scan in qem_mesh.cpp):
 *   1. Read PCD -> points (+ normals if present), sensor frame
 *   1. Read PCD -> points (+ normals if present), sensor frame
 *   2. Transform to global frame using the matched pose (R, t)
 *   3. Sensor origin in global frame = t
 *   4. Subsample (process_every_n), fill in missing normals via spatial PCA
 *   5. Orient normals toward the sensor (sign consistency)
 *   6. Compute per-point LiDAR weights = cos^2(incidence)
 *   7. Drop grazing returns (incident_cos < min_incident_cos)
 *   8. OpenMP-parallel ingest: each thread accumulates hits and ray-carved
 *      misses into a thread-local voxel map. Serial merge folds the
 *      thread-local maps into the master.
 *
 * The map is sparse: cells appear lazily as observations land in them.
 * Each cell stores
 *   - additive QEM   (A, b, c)              for vertex extraction via LDLT
 *   - normal stats   (sum, weight)          for multi-modal detection later
 *   - occupancy      (hit/miss counts and weights)
 *   - last_hit_scan / last_evidence_scan bookkeeping
 *
 * Output: two files
 *   - <output>.ply       ASCII PLY of v* per surface cell, with vertex normals
 *   - <output>_meta.csv  per-vertex metadata: voxel ijk, classification
 *                        (flat/edge/corner), QEM residual, normal consistency,
 *                        hit/miss counts. These are the signals the next
 *                        phases (multi-mode splitting; meshing) will consume.
 *
 * Cell labelling is three-state (surface / free / unknown). Unknown is
 * first-class: cells with no evidence stay unknown, no triangles are emitted
 * just because labels had to be filled in somewhere.

 * BEHAVIOR NOTE (this revision):
 *   NVT/BEO is enabled, but by default it only denoises PCA-fallback normals.
 *   Input normals and scan-line normals are treated as trusted unless
 *   --nvt_all_normals is passed. Scan-line metadata is built on the full scan
 *   before process_every_n subsampling, so organized row/column neighborhoods
 *   remain valid.
 *
 * Build:
 *   g++ -O3 -fopenmp -DHAS_OPENMP -I/usr/include/eigen3 \
 *       -std=c++17 qem_voxel_map.cpp -o qem_voxel_map
 *
 * Usage (drop-in shape compatible with qem_mesh's CLI for the common flags):
 *   ./qem_mesh  --pcd_folder /indoormapping/oxford_spires/keble-college/undist-scans-and-poses-seq-3/vilens-slam/undist-clouds/undist-clouds/ --pose_file  /indoormapping/oxford_spires/keble-college/undist-scans-and-poses-seq-3/vilens-slam/slam-poses.csv \
 --mesh_output /home/affanm/planar_mesh/planar_mesh/cpp_implementation/meshes/qem_mesh_oxford_spires_kc03_030626_2.ply --scanline_jump_abs 0.08  \
 --scanline_jump_rel 0.01   --process_every_n 1 --mesh_every 10 --min_hit_count_vertex 1 --enable_sparse_scaffold  --seam_close_iters 2 --enable_component_growth --enable_persistent_mesh --seed_promote_min_hits 1 --seed_l1_min_unique_scans 1 --seed_l1_min_children 1 --seed_l1_min_consistency 0.55 --seed_l1_max_sqrt_residual 0.12 --seed_l1_max_ray_normal_fraction 0.80 --seed_l1_child_normal_dot 0.60 --seed_l1_child_plane_dist 0.20

 */

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::string make_run_id() {
    using namespace std::chrono;
    const auto ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    unsigned rnd = 0;
    try { rnd = std::random_device{}(); } catch (...) { rnd = (unsigned)(ns & 0xffffffffu); }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%lld_%08x", (long long)ns, rnd);
    return std::string(buf);
}

// ========================================================================= //
// 1. Settings                                                                //
// ========================================================================= //

struct Settings {
    // Spatial discretisation.
    double voxel_size            = 0.1;     // metres
    // LiDAR weighting / filtering.
    double range_precision       = 0.015;   // sigma_range, metres

    // Probabilistic voxel-plane layer. This keeps QEM as the vertex solver,
    // but adds the uncertainty model used by adaptive voxel mapping: LiDAR
    // range/bearing noise + pose uncertainty -> point covariance -> voxel
    // plane uncertainty. The resulting sigma is used to weight QEM evidence
    // and to make seed/parent/mesh gates scale-aware instead of purely fixed.
    bool   enable_probabilistic_planes = true;
    double bearing_sigma_rad      = 0.0015; // LiDAR bearing std-dev, radians
    double pose_trans_sigma       = 0.02;   // fallback pose translation std-dev, m
    double pose_rot_sigma_rad     = 0.002;  // fallback pose rotation std-dev, radians
    double prob_min_sigma         = 0.005;  // lower bound for plane/point sigma, m
    double prob_max_sigma         = 0.25;   // upper bound for adaptive gate sigma, m
    double prob_qem_ref_sigma     = 0.015;  // ref sigma for QEM weight scaling, m
    double prob_qem_min_scale     = 0.10;   // never reduce a valid hit below this scale
    double prob_gate_sigma        = 3.0;    // probabilistic gate width, e.g. 3 sigma
    double prob_gate_min_factor   = 0.50;   // adaptive gate cannot tighten below base*this
    double prob_gate_max_factor   = 3.00;   // adaptive gate cannot relax above base*this
    double prob_planar_eigen_thresh = 0.0;  // <=0 uses 0.25*voxel_size^2
    double min_incident_cos      = 0.30;    // reject grazing (|n.dir| < this)
    int    process_every_n       = 2;       // subsample factor per scan
    int    estimate_normals_k    = 20;      // k for PCA fallback
    bool   orient_to_sensor      = true;    // flip normals to face sensor
    // Normal Voting Tensor / Binary Eigenvalue Optimization denoising.
    bool   enable_nvt            = true;    // denoise normals before QEM accumulation
    bool   nvt_only_for_pca      = true;    // do not alter trusted input normals unless disabled
    int    nvt_k                 = 16;      // neighbors for NVT voting
    int    nvt_iters             = 1;       // 1 is usually enough for LiDAR; >1 may oversmooth
    double nvt_rho_cos           = 0.85;    // hard normal agreement threshold
    double nvt_tau               = 0.15;    // BEO eigenvalue threshold on normalized tensor
    double nvt_damping           = 3.0;     // d in n' = d n + T_BEO n
    double nvt_min_conf          = 0.10;    // lower bound before soft confidence mapping
    double nvt_conf_soft_floor   = 0.50;    // final QEM confidence multiplier is floor+(1-floor)*conf
    double nvt_max_radius        = 0.0;     // metres; <=0 uses 5*voxel_size safety cap
    // LiDAR scan-line / range-image awareness. These gates use organized PCD
    // layout, ring/channel fields, or optional elevation/azimuth quantization
    // to avoid estimating/smoothing normals across depth discontinuities.
    bool   enable_scanline       = true;    // use scan-line geometry when available
    bool   scanline_normals      = true;    // prefer range-image normals before PCA fallback
    int    scanline_rows         = 0;       // 0 = auto; else quantize elevation into rows
    int    scanline_cols         = 0;       // 0 = auto/2048; used for ring/elevation scans
    int    scanline_gate_rows    = 1;       // local row window for depth-jump gating
    int    scanline_gate_cols    = 2;       // local column window for depth-jump gating
    double scanline_jump_abs     = 0.30;    // metres; absolute depth-jump threshold
    double scanline_jump_rel     = 0.03;    // relative threshold: + rel * range
    double scanline_boundary_conf = 0.65;   // QEM normal-confidence multiplier at depth jumps
    // PandarQT64 support for unorganized per-scan clouds that have no
    // ring/channel field. The input PCD points are still treated as sensor-
    // frame points and are mapped with the external SLAM pose. By default we
    // use the PandarQT design channel table from Appendix I of the user
    // manual. A unit-specific angle correction file is still preferred when
    // available.
    bool   pandar_qt64_mode      = false;   // --sensor pandar_qt64 or --pandar_qt64
    int    pandar_qt64_channels  = 64;
    double pandar_qt64_vmin_deg  = -52.121;
    double pandar_qt64_vmax_deg  =  52.133;
    bool   pandar_qt64_infer_rings = false; // optional --pandar_qt64_infer
    bool   pandar_qt64_uniform     = false; // optional --pandar_qt64_uniform
    std::string pandar_qt64_calib = "";     // optional CSV/TXT: channel,elevation[,azimuth]
    // Ray-carving / free-space evidence.
    bool   carve_rays            = true;
    int    max_ray_voxels        = 2000;    // safety cap per ray
    // Cell labelling (three-state surface / free / unknown).
    double thresh_surface        = 0.10;    // occupancy_score >= this -> surface
    double thresh_free           = -0.50;   // occupancy_score <= this -> free
    int    min_evidence          = 2;       // hits + misses needed to label
    int    min_hit_count_vertex  = 2;       // min hits to extract a vertex

    // Weak/seed voxel layer. These are real LiDAR endpoint observations that
    // are too weak/sparse/grazing to become ordinary surface voxels yet. They
    // are accumulated in a separate hypothesis map, exported for diagnosis,
    // and promoted into the main QEM map only after enough self- or neighbor-
    // consistent evidence appears. They are NOT used by meshing directly.
    bool   enable_seed_voxels    = true;
    double seed_min_incident_cos = 0.08;    // below this, reject as too grazing/noisy
    double seed_hit_weight_scale = 0.25;    // weak endpoint hit weight multiplier
    double seed_min_hit_weight   = 0.005;   // floor to keep a tiny amount of QEM evidence
    bool   seed_carve_rays       = false;   // default: weak hits do not carve free-space
    double seed_miss_weight_scale = 0.05;   // used only if seed_carve_rays=true
    bool   seed_ray_normal_fallback = true; // points whose normal fell back to -ray become seeds
    int    seed_export_min_hits  = 1;       // pending seed PLY export threshold
    int    seed_promote_min_hits = 3;       // promote from seed map by self evidence
    int    seed_promote_neighbor_min_hits = 1; // allow neighbor-supported promotion
    int    seed_promote_min_neighbors = 2;  // confirmed neighbor cells required
    double seed_promote_min_occupancy = -0.35; // block promotion into strongly free main cells
    double seed_promote_min_consistency = 0.70;
    double seed_promote_max_sqrt_residual = 0.10; // metres
    double seed_promote_neighbor_normal_dot = 0.85;
    double seed_promote_neighbor_plane_dist_factor = 1.50;
    bool   export_seed_vertices = true;
    // Ray-normal fallback safety for ordinary L0 seed maturation. Keep this
    // separate from the stricter L1 parent gate so adding L1 does not silently
    // tighten the older self/neighbor seed path.
    double seed_max_ray_normal_fraction = 0.50;
    // Same-resolution seed-to-seed cluster promotion. This complements the
    // L1 parent validator and catches weak coherent groups that straddle a
    // 0.20 m parent boundary.
    int    seed_promote_cluster_min_size = 3;
    int    seed_promote_cluster_radius_voxels = 1;
    double seed_promote_cluster_normal_dot = 0.85;
    double seed_promote_cluster_plane_dist_factor = 1.0;
    double seed_promote_cluster_merged_max_sqrt_residual = 0.08;

    // Single-level adaptive seed parent layer. L0 seed voxels stay at
    // voxel_size (usually 0.10 m). L1 parents are 2x coarser (0.20 m at
    // voxel_size=0.10) and are used only as validators: a parent can certify
    // that weak L0 children lie on a coherent surface, but the parent itself
    // is not exported or meshed as geometry.
    bool   enable_seed_l1_parents = true;
    int    seed_l1_factor = 2;                 // fixed practical default: 2x2x2 child block
    int    seed_l1_min_unique_scans = 2;       // require observations from >=2 scans
    int    seed_l1_min_children = 2;           // direct L0 seed children inside the parent
    double seed_l1_min_consistency = 0.70;
    double seed_l1_max_sqrt_residual = 0.08;  // metres, merged-parent QEM residual
    double seed_l1_max_ray_normal_fraction = 0.35;
    double seed_l1_child_plane_dist = 0.10;   // metres; <=0 means voxel_size
    double seed_l1_child_normal_dot = 0.80;
    double seed_l1_child_min_occupancy = -0.35;
    bool   seed_l1_promote_children = true;

    // Evidential occupancy grid mapping (EOGM / Dempster-Shafer style).
    // We retain surface, free, unknown, and conflict mass per voxel instead
    // of collapsing all evidence into a single hit/miss score. This is used
    // as a maturity/confidence gate for seed promotion and for optional
    // generative hole filling. QEM still decides geometry.
    bool   enable_eogm = true;
    // Evidence policy for seed/L1 promotion.
    //   old  = use the original hit/miss occupancy-style gates.
    //   eogm = use only EOGM belief/plausibility/conflict gates for maturity.
    // Geometry gates (QEM residual, normal consistency, ray-normal fraction,
    // parent/child plane checks) remain mandatory in both modes.
    // Important: this is NOT old && EOGM; it is old OR EOGM by mode.
    std::string seed_evidence_mode = "old";
    // Optional: apply EOGM as an additional face-veto during normal meshing.
    // Default false so EOGM can be tested as a seed-promotion model without
    // silently reducing mesh recall after promotion. Generative fill still uses
    // its own EOGM vetoes regardless of this flag.
    bool   eogm_mesh_gate = false;
    double eogm_hit_scale = 0.45;            // main endpoint evidence strength from QEM weight
    double eogm_seed_hit_scale = 0.22;       // weak/seed endpoint evidence strength
    double eogm_miss_scale = 0.075;          // ray-carved free evidence strength
    double eogm_seed_miss_scale = 0.025;     // weak-ray free evidence when seed_carve_rays is enabled
    double eogm_max_hit_reliability = 0.75;
    double eogm_max_seed_hit_reliability = 0.35;
    double eogm_max_miss_reliability = 0.22;
    double eogm_boundary_discount = 0.50;    // scan-boundary endpoints are less trustworthy
    double eogm_ray_normal_discount = 0.25;  // ray-normal fallback gives endpoint evidence, weak geometry
    double eogm_seed_min_plaus_surface = 0.45; // Pl(S)=mS+mU, conflict not counted as support
    double eogm_seed_max_bel_free = 0.70;
    double eogm_seed_max_conflict = 0.65;
    double eogm_cluster_min_plaus_surface = 0.50;
    double eogm_cluster_max_bel_free = 0.65;
    double eogm_cluster_max_conflict = 0.60;
    double eogm_mesh_max_bel_free = 0.72;
    double eogm_mesh_max_conflict = 0.75;

    // Minimal EOGM-gated generative extrapolation. This is deliberately
    // conservative: it never writes generated vertices back to the QEM map.
    // It only adds optional mesh-only virtual vertices when three real QEM
    // vertices are mutually planar/consistent but too sparse for direct DC
    // triangles. FREE or high-conflict EOGM samples veto the fill.
    bool   enable_eogm_generative_fill = false;
    int    gen_max_faces = 200000;
    int    gen_min_support_confirmed = 1;
    double gen_max_edge_factor = 8.0;
    double gen_max_merged_sqrt_residual = 0.060;
    double gen_max_point_plane_dist_factor = 1.25;
    double gen_max_bel_free = 0.35;
    double gen_max_conflict = 0.45;
    double gen_min_support_plaus_surface = 0.45;

    // Confidence tier policy. Confirmed vertices are ordinary surface cells.
    // Weak/promoted tiers are exported for vertex recall and may be used only
    // by corner_dc_plus adaptive fill, not by baseline corner_dc connectivity.
    bool   export_weak_vertices = true;
    bool   cdp_use_weak_vertices = true;
    bool   cdp_weak_triangles_require_confirmed = true;
    // Vertex solve.
    double lambda_p              = 0.01;    // anchor regularisation
    bool   clamp_to_voxel        = true;
    // QEM eigen-classification thresholds (l2/l1 and l3/l1 ratios).
    double rank2_ratio_thresh    = 0.15;
    double rank3_ratio_thresh    = 0.15;
    // Misc.
    int    num_scans             = -1;
    int    snapshot_interval     = 0;       // 0 = off

    // Visualization dump (file-based; viewer served by viz_server.py).
    //   Enable by setting dump_dir to a non-empty path. ROI is optional --
    //   when dump_roi_radius <= 0 the dump includes every surface vertex,
    //   capped by dump_max_verts. dump_every throttles the per-scan writes
    //   so long runs don't fill the disk.
    std::string dump_dir         = "";
    Vec3        dump_roi_center  = Vec3::Zero();
    double      dump_roi_radius  = 0.0;
    int         dump_every       = 1;
    int         dump_max_verts   = 500000;

    // Phase 3A meshing. This is intentionally local/incremental: it connects
    // the already-extracted voxel-QEM vertices over smooth patches using
    // voxel-neighbor adjacency and conservative triangle rejection. It does
    // not yet do crease seam insertion; wall/floor/edge stitching is a later
    // stage.
    std::string mesh_output      = "";      // final mesh PLY; empty disables final mesh export
    std::string mesh_dump_dir    = "";      // optional per-snapshot/submap mesh PLY directory
    int         mesh_every       = 0;       // 0 = no incremental mesh snapshots
    int         mesh_submap_scans = 0;      // 0 = mesh all accumulated cells; >0 = recent last-hit window
    int         mesh_min_hit_count = 3;
    double      mesh_max_sqrt_residual = 0.0; // metres; <=0 disables residual filter
    double      mesh_min_normal_consistency = 0.0;
    double      mesh_normal_dot  = 0.90;    // smooth-patch normal compatibility; abs(dot) is used
    double      mesh_neighbor_radius_factor = 1.85;
    double      mesh_max_edge_factor = 2.50;
    double      mesh_triangle_normal_dot = 0.35;
    double      mesh_min_area_factor = 0.005; // area >= factor * voxel_size^2
    double      mesh_max_boundary_ratio = 1.01; // <=1.0; 1.01 means no boundary filter
    double      mesh_max_fan_angle = 2.50;  // radians; reject fan triangles across large angular gaps
    bool        mesh_reject_free_centroid = true;

    // Phase 3B open-scene dual contouring / surface-net meshing. This uses
    // the same per-voxel QEM vertices, but connectivity is generated from the
    // voxel grid and ray-carved free-space evidence rather than local tangent
    // fan triangulation. It deliberately treats UNKNOWN as open space to avoid
    // frontier caps.
    std::string mesh_mode        = "smooth"; // smooth | dual | corner_dc | corner_dc_plus | hybrid
    // PlanarMesh-style persistent incremental meshing. When enabled, a single
    // mesh is kept alive across scans; each export re-meshes only the voxels
    // touched since the previous export (the "dirty" region) and splices the
    // result into the persistent mesh, leaving clean regions untouched. The
    // QEM vertex/plane backend is unchanged. This replaces the old behavior of
    // rebuilding the whole mesh from scratch on every snapshot/export.
    bool        persistent_incremental_mesh = true;
    // Spatial halo (in voxels) grown around the dirty region before the local
    // re-mesh, so triangles straddling the dirty/clean seam reconnect to their
    // clean neighbors. 0 = auto-derive from the active mesh edge factors.
    int         persistent_mesh_halo_voxels = 0;
    // Seam-aware hole closing. After the dirty region is spliced into the
    // persistent mesh, run a PlanarMesh-style angular hole-closing pass over the
    // boundary vertices that sit on the persistent<->local seam. Unlike the
    // smooth mesher, this pass synthesizes the bridging edge needed to close a
    // cavity, so holes that straddle the incremental seam are closed the way a
    // single live mesh would close them.
    bool        enable_seam_closing = true;
    int         seam_close_iters    = 2;     // cascade passes (a closed triangle can expose the next)
    // Retire persistent faces whose interior (centroid / edge midpoints /
    // interior samples) crosses observed FREE / high-conflict space, even when
    // all three vertex cells are still valid. Catches stale "bridge" triangles
    // spanning a doorway/window that vertex-aliveness alone cannot detect. Also
    // vetoes such triangles during seam closing.
    bool        persistent_retire_free_faces = true;
    // Sparse-region scaffold structuring. Where raw LiDAR points are too sparse
    // to fill voxels at the working resolution, inherit a coarse-level QEM plane
    // / normal / evidence into the empty fine voxels ("virtual inherited points")
    // so the mesher has vertices to connect. Turning this on enables the
    // hierarchical scaffold inheritance + fill machinery and (unless the user
    // picked a mesh_mode explicitly) selects corner_dc_plus so the scaffold fill
    // pass can emit faces.
    bool        sparse_region_scaffold = true;
    bool        dc_require_free  = true;     // only create quads on observed surface/free interfaces
    double      dc_normal_dot    = 0.50;     // looser than smooth mode: allows curves and creases
    double      dc_max_edge_factor = 2.50;   // max quad triangle edge / voxel_size
    double      dc_min_area_factor = 0.005;  // area >= factor * voxel_size^2
    double      dc_triangle_normal_dot = 0.15; // reject only badly flipped/degenerate faces

    // corner_dc_plus: boundary-guided adaptive QEM grow/fill.
    // First builds corner-derived DC, then grows triangles from mesh boundary
    // edges toward weak/unmeshed QEM vertices when the merged local QEM is
    // planar/consistent and free-space does not reject the candidate. This is
    // an RRS-like search: boundary vertices/edges own an adaptive radius; weak
    // QEM vertices are accepted only if they fall inside that radius.
    int         cdp_iters = 1;
    int         cdp_min_incident_faces = 2;      // vertices below this are fill targets
    int         cdp_max_candidates_per_edge = 24;
    double      cdp_flat_radius_factor = 6.0;    // radius for simple planar boundaries
    double      cdp_curve_radius_factor = 3.0;   // radius for non-flat if cdp_allow_curve=true
    double      cdp_max_radius_factor = 8.0;
    double      cdp_max_edge_factor = 6.0;
    double      cdp_planar_normal_dot = 0.90;
    double      cdp_curve_normal_dot = 0.75;
    double      cdp_min_consistency = 0.88;
    double      cdp_max_boundary_ratio = 0.85;
    double      cdp_max_sqrt_residual = 0.05;        // per-cell vertex residual gate, metres
    double      cdp_max_merged_sqrt_residual = 0.05; // merged QEM residual gate, metres
    double      cdp_max_point_plane_dist_factor = 1.0;
    bool        cdp_allow_curve = false;          // keep conservative by default
    bool        cdp_reject_free_samples = true;

    // Hierarchical scaffold inheritance: L0 remains the real voxel grid. L1/L2/...
    // are coarser QEM parent supports that push weak virtual QEM constraints
    // back into crossed L0 child voxels. Virtual QEM is separated from real
    // observations and decays as real child QEM evidence accumulates.
    bool        enable_hierarchical_scaffold = false;
    bool        enable_planar_scaffold_inherit = false; // legacy alias: enables hierarchical scaffold inheritance
    bool        enable_planar_scaffold = false;         // legacy alias: enables hierarchical mesh-only scaffold fill
    bool        enable_hierarchical_scaffold_fill = true; // mesh-only tiling pass; makes hierarchy a superset of old scaffold
    int         scaffold_base_factor = 2;          // base=2 => L1=2x, L2=4x, L3=8x
    int         scaffold_max_level = 2;            // levels above L0 to use
    int         scaffold_min_parent_children = 2;  // old single-L1 scaffold default
    int         scaffold_min_parent_children_per_level = 2;
    double      scaffold_min_child_fraction = 0.0;
    double      scaffold_max_merged_sqrt_residual = 0.040; // old single-L1 scaffold default, metres
    double      scaffold_residual_level_decay = 1.00;      // 1.0 keeps L2 a true superset unless user tightens it
    double      scaffold_max_rank_ratio = 0.10;
    double      scaffold_rank_level_decay = 1.00;
    double      scaffold_min_normal_consistency = 0.0;     // old scaffold had no separate consistency gate
    double      scaffold_max_edge_factor = 2.5;
    double      scaffold_max_bel_free = 0.35;
    double      scaffold_max_conflict = 0.45;
    double      scaffold_inherit_weight = 0.10;
    double      scaffold_inherit_level_decay = 0.50;
    double      scaffold_inherit_decay_weight_ref = 1.0;
    bool        scaffold_inherit_into_real_cells = true;
    // Mesh-only fill can mimic the old single scaffold exactly (all children of a valid parent)
    // or be tightened to only child voxels crossed by the parent plane.
    bool        scaffold_fill_require_plane_crossing = false;

    // Let inherited/scaffold-only cells contribute weak corner-sign evidence
    // for corner_dc. Without this, inherited cells export vertices but remain
    // UNKNOWN for topology, so isolated sparse scaffold patches cannot form
    // signs/edges and only help as adjacent gap-fill. This branch is gated by
    // EOGM, inherited-decay scale, QEM residual, and a small sign weight so the
    // prior remains weaker than real SURFACE/FREE evidence.
    bool        scaffold_inherited_corner_sign = true;
    double      scaffold_inherited_sign_weight = 0.25;
    double      scaffold_inherited_min_decay_scale = 0.02;
    double      scaffold_inherited_sign_max_sqrt_residual = 0.060;

    // Component growth: PlanarMesh-like boundary growth on top of the fixed
    // voxel-QEM map. The voxel grid remains the evidence layer. This pass
    // treats connected mesh components as local surface components, estimates
    // a component tangent plane/QEM support, then grows boundary edges toward
    // still-uncovered QEM/scaffold vertices using adaptive boundary radii.
    // It is a topology pass: it does not create or modify voxel evidence.
    bool        enable_component_growth = false;
    int         component_growth_iters = 1;
    int         component_growth_min_component_faces = 2;
    int         component_growth_target_min_incident_faces = 2;
    int         component_growth_max_candidates_per_edge = 16;
    double      component_growth_boundary_radius_factor = 2.5;
    double      component_growth_flat_radius_factor = 5.0;
    double      component_growth_max_radius_factor = 6.0;
    double      component_growth_max_edge_factor = 3.0;
    double      component_growth_normal_dot = 0.94;
    double      component_growth_component_normal_dot = 0.90;
    double      component_growth_plane_dist_factor = 0.75;
    double      component_growth_max_merged_sqrt_residual = 0.040;
    double      component_growth_max_point_plane_dist_factor = 0.75;
    double      component_growth_max_bel_free = 0.35;
    double      component_growth_max_conflict = 0.45;
    double      component_growth_min_plaus_surface = 0.45;
    bool        component_growth_reject_free_samples = true;
    bool        component_growth_require_confirmed_anchor = true;
    bool        component_growth_require_merged_qem_when_available = true;

    // Persistent QEM-surface component state (PlanarMesh-like stages 1-3).
    // Stage 1: after every mesh build, connected mesh components are assigned
    // stable component IDs by voxel-key overlap and their owned vertices,
    // edges, faces, boundary vertices, and boundary radii are cached.
    // Stage 2: scan integration marks dirty voxel neighborhoods; growth can
    // optionally process only components/targets touched by those dirty keys.
    // Stage 3: boundary growth uses an RRS-like predicate: a target vertex is
    // eligible only when it lies inside at least one boundary vertex radius.
    bool        component_growth_persistent_state = false;
    bool        component_growth_dirty_only = false;
    int         component_growth_dirty_radius_voxels = 1;
    bool        component_growth_use_rrs_boundary_search = true;
    bool        component_growth_clear_dirty_after_mesh = true;

    // Persistent face reuse / write-back. This is the part that turns
    // component persistence from read-only bookkeeping into an output prior:
    // previously accepted faces are re-emitted unless current QEM/EOGM/free
    // evidence retires them. Vertex positions are still re-solved from the
    // current voxel QEM table, so retained faces keep topology, not stale XYZ.
    bool        component_persistent_reuse_faces = true;
    bool        component_persistent_reuse_dirty_faces = true;
    bool        component_persistent_reuse_require_all_vertices = true;
    double      component_persistent_reuse_max_bel_free = 0.35;
    double      component_persistent_reuse_max_conflict = 0.45;
    double      component_persistent_reuse_min_plaus_surface = 0.35;

    // Stages 3-6: QEM-rank aware component growth, deletion, shrink, and
    // optional simplification. The rank-aware RRS treats flat/Rank-1 vertices
    // as planar disks, edge/Rank-2 vertices as line-supported disks, and
    // corner/Rank-3 vertices as small terminal junctions. This prevents an
    // edge or corner from growing arbitrarily across a plane while still
    // allowing plane-edge-corner stitching when QEM residuals agree.
    bool        component_growth_qem_rank_rrs = true;
    double      component_growth_edge_line_dist_factor = 0.60;
    double      component_growth_corner_radius_factor = 0.55;
    double      component_growth_rank_transition_normal_dot = 0.80;

    // Stage 4: FIS-like free-space deletion. We do not keep the full raw ray
    // history here, so this approximates PlanarMesh FIS by deleting faces whose
    // centroid/edge samples now fall in FREE or high-conflict EOGM cells.
    bool        enable_component_fis_delete = false;
    bool        component_fis_delete_dirty_only = true;
    double      component_fis_delete_max_bel_free = 0.35;
    double      component_fis_delete_max_conflict = 0.45;
    double      component_fis_delete_min_plaus_surface = 0.35;

    // Stage 5: radius shrink/delete. If a component edge becomes longer than
    // the radius allowed by the QEM-rank of its endpoint vertices, remove its
    // incident faces. Boundary and high-curvature regions therefore get finer
    // local triangles while broad rank-1 planes can keep larger ones.
    bool        enable_component_radius_shrink = false;
    bool        component_radius_shrink_dirty_only = true;
    double      component_shrink_edge_over_radius = 1.15;
    int         component_shrink_min_component_faces = 2;

    // Stage 6: optional adaptive simplification. This is conservative and off
    // by default: it only thins redundant interior faces on flat components and
    // then compacts unused vertices. It is not a full constrained Delaunay
    // re-triangulation, but it gives a safe first compactness knob.
    bool        enable_component_adaptive_simplification = false;
    int         component_simplify_min_component_faces = 40;
    double      component_simplify_centroid_radius_factor = 1.5;
    double      component_simplify_normal_dot = 0.97;
    bool        component_simplify_preserve_boundary_faces = true;
    bool        component_compact_after_topology_ops = true;

    // Optional mesh-vertex smoothing. This is intentionally disabled by
    // default and conservative when enabled. It is NOT a planarization pass:
    // confirmed rank-1/flat vertices are protected unless explicitly allowed.
    // The smoother is intended for weak/promoted/generated or otherwise noisy
    // mesh vertices after connectivity is built. It uses bilateral edge stops,
    // EOGM/free-space vetoes, and an optional QEM projection/acceptance test.
    bool   enable_vertex_smoothing = false;
    int    vertex_smooth_iters = 1;
    bool   vertex_smooth_apply_to_confirmed = false;
    bool   vertex_smooth_allow_confirmed_flat = false;
    bool   vertex_smooth_preserve_edges = true;
    bool   vertex_smooth_normal_only = true;
    bool   vertex_smooth_qem_project = true;
    int    vertex_smooth_min_degree = 2;
    int    vertex_smooth_min_hit_count = 3;
    double vertex_smooth_radius_factor = 2.0;
    double vertex_smooth_sigma_spatial_factor = 1.0;
    double vertex_smooth_sigma_normal = 0.25;
    double vertex_smooth_sigma_plane_factor = 0.75;
    double vertex_smooth_normal_dot = 0.85;
    double vertex_smooth_min_normal_consistency = 0.70;
    double vertex_smooth_min_sqrt_residual = 0.03;
    double vertex_smooth_max_move_factor = 0.25;
    double vertex_smooth_anchor_scale = 1.0;
    double vertex_smooth_candidate_scale = 1.0;
    double vertex_smooth_residual_tol_factor = 1.0;
    double vertex_smooth_max_bel_free = 0.60;
    double vertex_smooth_max_conflict = 0.60;
    double vertex_smooth_max_boundary_ratio = 0.85;
};

// ========================================================================= //
// 2. Voxel key + hash                                                        //
// ========================================================================= //

struct VoxKey {
    int32_t i, j, k;
    bool operator==(const VoxKey& o) const noexcept {
        return i == o.i && j == o.j && k == o.k;
    }
};
struct VoxHash {
    size_t operator()(const VoxKey& v) const noexcept {
        // Triple-of-ints hash. The big primes follow the standard pattern used
        // in spatial hashes (Teschner et al.); decorrelates the three axes well
        // enough for sparse voxel maps. Matches the style of qem_mesh.cpp.
        uint64_t h = (uint64_t)(uint32_t)v.i * 73856093ULL;
        h ^= (uint64_t)(uint32_t)v.j * 19349663ULL;
        h ^= (uint64_t)(uint32_t)v.k * 83492791ULL;
        return (size_t)h;
    }
};


// ========================================================================= //
// 2b. Probabilistic plane statistics                                         //
// ========================================================================= //
//
// This is the QEM-side equivalent of VoxelMap's Plane/plane_cov block. V1 of
// the patch used only sufficient statistics and an eigen-gap approximation.
// This V2 path keeps a bounded covariance sample buffer per voxel and computes
// the 6x6 plane covariance with the same eigenvector-Jacobian form used in the
// VoxelMap implementation / Eq. 7-8 of the paper. Once the buffer reaches the
// cap, the exact covariance is frozen and the buffer is released, preserving
// the QEM map's bounded memory profile. The all-hit streaming sums still keep
// the current center/normal/radius for mesh diagnostics and gates.

struct FittedProbPlane {
    bool valid = false;
    bool planar = false;
    int points_size = 0;
    Vec3 center = Vec3::Zero();
    Vec3 normal = Vec3::Zero();
    Vec3 x_normal = Vec3::Zero();
    Vec3 y_normal = Vec3::Zero();
    Mat3 covariance = Mat3::Zero();
    Mat6 plane_cov = Mat6::Zero(); // [normal(3), center/q(3)] covariance
    double min_eigen_value = 0.0;
    double mid_eigen_value = 0.0;
    double max_eigen_value = 0.0;
    double radius = 0.0;
    double d = 0.0;
    double sigma_plane = 0.0;      // conservative point-to-plane std-dev, m
    double normal_cov_trace = 0.0;
};

struct ProbPlaneSample {
    Vec3 p = Vec3::Zero();
    Mat3 cov = Mat3::Zero();
};

struct ProbPlaneStats {
    static constexpr int kCovSampleCap = 50;

    int points_size = 0;
    Vec3 sum_p = Vec3::Zero();
    Mat3 sum_ppT = Mat3::Zero();
    Mat3 sum_cov = Mat3::Zero();

    std::vector<ProbPlaneSample> cov_samples;
    bool exact_cov_valid = false;
    Mat6 exact_plane_cov = Mat6::Zero();
    double exact_normal_cov_trace = 0.0;
    int exact_cov_points = 0;

    static FittedProbPlane fit_geometry_from_sums(int n, const Vec3& sp, const Mat3& sppT) {
        FittedProbPlane out;
        out.points_size = n;
        if (n < 3) return out;

        const double inv_n = 1.0 / (double)n;
        out.center = sp * inv_n;
        out.covariance = sppT * inv_n - out.center * out.center.transpose();
        out.covariance = 0.5 * (out.covariance + out.covariance.transpose());

        Eigen::SelfAdjointEigenSolver<Mat3> eig(out.covariance);
        if (eig.info() != Eigen::Success) return out;
        Vec3 evals = eig.eigenvalues().cwiseMax(0.0); // ascending: min, mid, max
        Mat3 evecs = eig.eigenvectors();

        out.normal = evecs.col(0);
        out.y_normal = evecs.col(1);
        out.x_normal = evecs.col(2);
        if (out.normal.norm() < 1e-12 || out.y_normal.norm() < 1e-12 || out.x_normal.norm() < 1e-12)
            return out;
        out.normal.normalize();
        out.y_normal.normalize();
        out.x_normal.normalize();

        out.min_eigen_value = evals[0];
        out.mid_eigen_value = evals[1];
        out.max_eigen_value = evals[2];
        out.radius = std::sqrt(std::max(0.0, out.max_eigen_value));
        out.d = -out.normal.dot(out.center);
        out.valid = true;
        return out;
    }

    static bool exact_cov_from_samples(const std::vector<ProbPlaneSample>& samples,
                                       Mat6& out_cov,
                                       double& out_normal_trace,
                                       int& out_points) {
        out_cov.setZero();
        out_normal_trace = 0.0;
        out_points = (int)samples.size();
        const int N = (int)samples.size();
        if (N < 3) return false;

        Vec3 sp = Vec3::Zero();
        Mat3 sppT = Mat3::Zero();
        for (const auto& s : samples) {
            sp.noalias() += s.p;
            sppT.noalias() += s.p * s.p.transpose();
        }
        FittedProbPlane geom = fit_geometry_from_sums(N, sp, sppT);
        if (!geom.valid) return false;

        const double lmin = geom.min_eigen_value;
        const Vec3 n = geom.normal;
        Mat3 U;
        U.col(0) = geom.normal;
        U.col(1) = geom.y_normal;
        U.col(2) = geom.x_normal;
        const Vec3 evals(lmin, geom.mid_eigen_value, geom.max_eigen_value);
        const Mat3 J_Q = (1.0 / (double)N) * Mat3::Identity();

        for (const auto& smp : samples) {
            Mat3 F = Mat3::Zero();
            const Eigen::RowVector3d dT = (smp.p - geom.center).transpose();
            for (int m = 1; m < 3; ++m) {
                const double denom = (double)N * (lmin - evals[m]);
                if (std::abs(denom) < 1e-12) continue;
                const Vec3 um = U.col(m);
                Mat3 M = um * n.transpose() + n * um.transpose();
                F.row(m) = (dT / denom) * M;
            }

            Eigen::Matrix<double, 6, 3> J;
            J.setZero();
            // Matches VoxelMap: J_top = eigenvectors * F, J_bottom = I/N.
            J.block<3,3>(0,0) = U * F;
            J.block<3,3>(3,0) = J_Q;
            out_cov.noalias() += J * smp.cov * J.transpose();
        }

        out_cov = 0.5 * (out_cov + out_cov.transpose());
        out_normal_trace = out_cov.block<3,3>(0,0).trace();
        return true;
    }

    void freeze_exact_cov_if_ready() {
        if (exact_cov_valid || (int)cov_samples.size() < kCovSampleCap) return;
        Mat6 cov;
        double tr = 0.0;
        int pts = 0;
        if (exact_cov_from_samples(cov_samples, cov, tr, pts)) {
            exact_plane_cov = cov;
            exact_normal_cov_trace = tr;
            exact_cov_points = pts;
            exact_cov_valid = true;
            std::vector<ProbPlaneSample>().swap(cov_samples);
        }
    }

    void add_point(const Vec3& p, const Mat3& cov_world) {
        points_size++;
        sum_p.noalias() += p;
        sum_ppT.noalias() += p * p.transpose();
        sum_cov.noalias() += cov_world;
        if (!exact_cov_valid && (int)cov_samples.size() < kCovSampleCap) {
            cov_samples.push_back({p, cov_world});
            freeze_exact_cov_if_ready();
        }
    }

    void merge_from(const ProbPlaneStats& o) {
        points_size += o.points_size;
        sum_p.noalias() += o.sum_p;
        sum_ppT.noalias() += o.sum_ppT;
        sum_cov.noalias() += o.sum_cov;

        if (!exact_cov_valid) {
            if (o.exact_cov_valid) {
                exact_plane_cov = o.exact_plane_cov;
                exact_normal_cov_trace = o.exact_normal_cov_trace;
                exact_cov_points = o.exact_cov_points;
                exact_cov_valid = true;
                std::vector<ProbPlaneSample>().swap(cov_samples);
            } else {
                for (const auto& smp : o.cov_samples) {
                    if ((int)cov_samples.size() >= kCovSampleCap) break;
                    cov_samples.push_back(smp);
                }
                freeze_exact_cov_if_ready();
            }
        }
    }

    FittedProbPlane fit(const Settings& s) const {
        FittedProbPlane out = fit_geometry_from_sums(points_size, sum_p, sum_ppT);
        if (!out.valid) return out;

        const double inv_n = 1.0 / (double)std::max(points_size, 1);
        const Mat3 avg_point_cov = sum_cov * inv_n;
        const Mat3 center_cov_fallback = avg_point_cov * inv_n;

        bool have_exact = false;
        if (exact_cov_valid) {
            out.plane_cov = exact_plane_cov;
            out.normal_cov_trace = exact_normal_cov_trace;
            have_exact = true;
        } else if (cov_samples.size() >= 3) {
            Mat6 cov;
            double tr = 0.0;
            int pts = 0;
            if (exact_cov_from_samples(cov_samples, cov, tr, pts)) {
                out.plane_cov = cov;
                out.normal_cov_trace = tr;
                have_exact = true;
            }
        }

        if (!have_exact) {
            // Fallback used only for very young cells with fewer than three
            // covariance samples. Mature cells use exact Eq. 7-8 covariance.
            const double gap1 = std::max(out.mid_eigen_value - out.min_eigen_value, 1e-12);
            const double gap2 = std::max(out.max_eigen_value - out.min_eigen_value, 1e-12);
            const double meas_var = std::max(0.0, avg_point_cov.trace() / 3.0);
            const double normal_var_y = std::clamp(meas_var * inv_n / gap1, 0.0, 0.25);
            const double normal_var_x = std::clamp(meas_var * inv_n / gap2, 0.0, 0.25);
            Mat3 normal_cov = normal_var_y * (out.y_normal * out.y_normal.transpose())
                            + normal_var_x * (out.x_normal * out.x_normal.transpose());
            out.plane_cov.setZero();
            out.plane_cov.block<3,3>(0,0) = normal_cov;
            out.plane_cov.block<3,3>(3,3) = center_cov_fallback;
            out.normal_cov_trace = normal_cov.trace();
        }

        const double min_sigma2 = s.prob_min_sigma * s.prob_min_sigma;
        const double max_sigma2 = s.prob_max_sigma * s.prob_max_sigma;
        const Mat3 normal_cov = out.plane_cov.block<3,3>(0,0);
        const Mat3 center_cov = out.plane_cov.block<3,3>(3,3);

        // Scalar gate sigma: surface thickness + center uncertainty along the
        // normal + normal-direction uncertainty over the observed plane radius.
        double sigma2 = out.min_eigen_value
                      + out.normal.transpose() * center_cov * out.normal
                      + out.radius * out.radius * std::max(0.0, normal_cov.trace());
        sigma2 = std::clamp(std::max(sigma2, min_sigma2), min_sigma2, max_sigma2);
        out.sigma_plane = std::sqrt(sigma2);

        const double planar_thresh = (s.prob_planar_eigen_thresh > 0.0)
            ? s.prob_planar_eigen_thresh
            : 0.25 * s.voxel_size * s.voxel_size;
        out.planar = out.min_eigen_value <= planar_thresh;
        return out;
    }
};

static Mat3 skew_matrix(const Vec3& v) {
    Mat3 S;
    S <<     0.0, -v.z(),  v.y(),
          v.z(),     0.0, -v.x(),
         -v.y(),  v.x(),     0.0;
    return S;
}

static Mat3 lidar_point_cov_world(const Vec3& p_lidar, const Mat3& R, const Settings& s) {
    const double r = p_lidar.norm();
    const double min_var = s.prob_min_sigma * s.prob_min_sigma;
    if (r < 1e-9) return min_var * Mat3::Identity();

    const Vec3 u = p_lidar / r;
    const Mat3 radial = u * u.transpose();
    const Mat3 tangential = Mat3::Identity() - radial;
    const double range_var = s.range_precision * s.range_precision;
    const double bearing_var = (s.bearing_sigma_rad * r) * (s.bearing_sigma_rad * r);
    Mat3 cov_lidar = range_var * radial + bearing_var * tangential;

    const Mat3 p_hat = skew_matrix(p_lidar);
    const Mat3 rot_cov = (s.pose_rot_sigma_rad * s.pose_rot_sigma_rad) * Mat3::Identity();
    const Mat3 trans_cov = (s.pose_trans_sigma * s.pose_trans_sigma) * Mat3::Identity();
    Mat3 cov_world = R * cov_lidar * R.transpose()
                   + R * p_hat * rot_cov * p_hat.transpose() * R.transpose()
                   + trans_cov;
    return 0.5 * (cov_world + cov_world.transpose());
}

static double probabilistic_qem_weight_scale(const Vec3& n, const Mat3& point_cov_w, const Settings& s) {
    if (!s.enable_probabilistic_planes) return 1.0;
    double sigma2 = n.transpose() * point_cov_w * n;
    const double min_sigma2 = s.prob_min_sigma * s.prob_min_sigma;
    sigma2 = std::max(sigma2, min_sigma2);
    const double ref = (s.prob_qem_ref_sigma > 0.0) ? s.prob_qem_ref_sigma : s.range_precision;
    const double ref2 = std::max(ref * ref, min_sigma2);
    double scale = (sigma2 <= ref2) ? 1.0 : (ref2 / sigma2);
    return std::clamp(scale, s.prob_qem_min_scale, 1.0);
}

// ========================================================================= //
// 3. VoxelCell                                                               //
// ========================================================================= //
//
// Stores a 3D quadric in the (A, b, c) form rather than the 4x4 Q form. The
// math is identical -- f(x) = x^T A x - 2 b^T x + c -- but the (A, b, c)
// split lets the LDLT solve below be written directly, and saves the cost of
// packing/unpacking the 4x4 every time we read it.
//
// Accumulation is additive: per observation (s, n, w) with unit normal n,
//   A += w * n n^T
//   b += w * (n.s) * n
//   c += w * (n.s)^2
// QEM additivity is what makes incremental + streaming fusion cheap.

struct VoxelCell {
    // QEM accumulators.
    Mat3   A = Mat3::Zero();
    Vec3   b = Vec3::Zero();
    double c = 0.0;
    double weight_sum = 0.0;
    // Normal statistics.
    Vec3   normal_sum = Vec3::Zero();
    double normal_weight = 0.0;
    // Virtual QEM inherited from hierarchical scaffold parents. It is geometry-only
    // evidence: it does not affect occupancy hit/miss counts or EOGM masses.
    Mat3   A_inherited = Mat3::Zero();
    Vec3   b_inherited = Vec3::Zero();
    double c_inherited = 0.0;
    double weight_inherited = 0.0;
    Vec3   normal_sum_inherited = Vec3::Zero();
    double normal_weight_inherited = 0.0;
    double inherited_decay_ref = 0.0;
    int    inherited_count = 0;
    int    inherited_from_scan = -1;
    uint32_t inherited_level_mask = 0;
    // Streaming probabilistic plane statistics for uncertainty-aware gates.
    ProbPlaneStats prob_plane;
    // Occupancy evidence.
    int    hit_count  = 0;
    int    miss_count = 0;
    int    boundary_hit_count = 0;
    double hit_weight = 0.0;
    double miss_weight = 0.0;
    double boundary_weight = 0.0;
    // Diagnostics: how much of this main cell came from promoted weak/seed observations.
    int    promoted_seed_count = 0;
    double promoted_seed_weight = 0.0;
    // Parent-supported promotions are a subset of promoted_seed_count: these
    // weak L0 child voxels were validated by a coherent 0.20 m L1 parent.
    int    parent_supported_seed_count = 0;
    double parent_supported_seed_weight = 0.0;
    // Ray-normal-fallback normals are placeholders perpendicular to the beam;
    // track their weight so seed/L1 promotion can reject ray-dominated QEMs.
    int    ray_normal_hit_count = 0;
    double ray_normal_weight = 0.0;

    // Dempster-Shafer / EOGM mass assignment over {SURFACE, FREE}.
    // mU is ignorance/unknown and mC is retained conflict. Masses sum to 1.
    double eogm_surface = 0.0;
    double eogm_free = 0.0;
    double eogm_unknown = 1.0;
    double eogm_conflict = 0.0;
    // Bookkeeping. Two timestamps are kept because they answer different questions:
    //   last_hit_scan      = last scan that added a SURFACE HIT to this cell.
    //                        Use this for vertex freshness and submap mesh windows.
    //   last_evidence_scan = last scan that touched the cell at all (hit OR miss).
    //                        Useful for free-space carving diagnostics.
    int    last_hit_scan      = -1;
    int    last_evidence_scan = -1;

    static double clamp_mass(double x) { return std::clamp(x, 0.0, 1.0); }

    void normalize_eogm() {
        eogm_surface = std::max(0.0, eogm_surface);
        eogm_free = std::max(0.0, eogm_free);
        eogm_unknown = std::max(0.0, eogm_unknown);
        eogm_conflict = std::max(0.0, eogm_conflict);
        double sum = eogm_surface + eogm_free + eogm_unknown + eogm_conflict;
        if (sum < 1e-12) {
            eogm_surface = eogm_free = eogm_conflict = 0.0;
            eogm_unknown = 1.0;
            return;
        }
        eogm_surface /= sum;
        eogm_free /= sum;
        eogm_unknown /= sum;
        eogm_conflict /= sum;
    }

    // Conjunctive Dempster-Shafer update with explicit retained conflict.
    // We do not normalize conflict away; high conflict remains a risk signal.
    void combine_eogm_mass(double ms, double mf, double mu, double mc=0.0) {
        ms = clamp_mass(ms); mf = clamp_mass(mf); mu = clamp_mass(mu); mc = clamp_mass(mc);
        double qsum = ms + mf + mu + mc;
        if (qsum < 1e-12) return;
        ms /= qsum; mf /= qsum; mu /= qsum; mc /= qsum;

        double s0 = eogm_surface, f0 = eogm_free, u0 = eogm_unknown, c0 = eogm_conflict;
        double ns = s0 * ms + s0 * mu + u0 * ms;
        double nf = f0 * mf + f0 * mu + u0 * mf;
        double nu = u0 * mu;
        double nc = c0 + (1.0 - c0) * mc + s0 * mf + f0 * ms;
        eogm_surface = ns;
        eogm_free = nf;
        eogm_unknown = nu;
        eogm_conflict = nc;
        normalize_eogm();
    }

    void add_eogm_surface(double reliability) {
        reliability = clamp_mass(reliability);
        if (reliability <= 0.0) return;
        combine_eogm_mass(reliability, 0.0, 1.0 - reliability, 0.0);
    }

    void add_eogm_free(double reliability) {
        reliability = clamp_mass(reliability);
        if (reliability <= 0.0) return;
        combine_eogm_mass(0.0, reliability, 1.0 - reliability, 0.0);
    }

    void add_hit(const Vec3& s, const Vec3& n, double w, int scan_idx,
                 bool scan_boundary=false, bool ray_normal_fallback=false,
                 double eogm_surface_reliability=-1.0,
                 const Mat3* point_cov_world=nullptr) {
        // n is assumed unit. add_plane(s, n, w) in QEM terms.
        const double n_dot_s = n.dot(s);
        A.noalias() += w * (n * n.transpose());
        b.noalias() += (w * n_dot_s) * n;
        c           += w * n_dot_s * n_dot_s;
        weight_sum  += w;
        normal_sum.noalias() += w * n;
        normal_weight += w;
        hit_count++;
        hit_weight += w;
        if (scan_boundary) { boundary_hit_count++; boundary_weight += w; }
        if (ray_normal_fallback) { ray_normal_hit_count++; ray_normal_weight += w; }
        if (point_cov_world) prob_plane.add_point(s, *point_cov_world);
        if (eogm_surface_reliability < 0.0) {
            eogm_surface_reliability = std::min(0.65, 1.0 - std::exp(-0.45 * std::max(0.0, w)));
            if (scan_boundary) eogm_surface_reliability *= 0.5;
            if (ray_normal_fallback) eogm_surface_reliability *= 0.25;
        }
        add_eogm_surface(eogm_surface_reliability);
        last_hit_scan      = scan_idx;
        last_evidence_scan = scan_idx;
    }

    void add_miss(double w, int scan_idx, double eogm_free_reliability=-1.0) {
        miss_count++;
        miss_weight += w;
        if (eogm_free_reliability < 0.0) {
            eogm_free_reliability = std::min(0.22, 1.0 - std::exp(-0.075 * std::max(0.0, w)));
        }
        add_eogm_free(eogm_free_reliability);
        last_evidence_scan = scan_idx;
    }

    void merge_from(const VoxelCell& o) {
        A.noalias() += o.A;
        b.noalias() += o.b;
        c           += o.c;
        weight_sum  += o.weight_sum;
        normal_sum.noalias() += o.normal_sum;
        normal_weight += o.normal_weight;
        hit_count   += o.hit_count;
        miss_count  += o.miss_count;
        boundary_hit_count += o.boundary_hit_count;
        hit_weight  += o.hit_weight;
        miss_weight += o.miss_weight;
        boundary_weight += o.boundary_weight;
        promoted_seed_count += o.promoted_seed_count;
        promoted_seed_weight += o.promoted_seed_weight;
        parent_supported_seed_count += o.parent_supported_seed_count;
        parent_supported_seed_weight += o.parent_supported_seed_weight;
        ray_normal_hit_count += o.ray_normal_hit_count;
        ray_normal_weight += o.ray_normal_weight;
        prob_plane.merge_from(o.prob_plane);
        A_inherited.noalias() += o.A_inherited;
        b_inherited.noalias() += o.b_inherited;
        c_inherited           += o.c_inherited;
        weight_inherited      += o.weight_inherited;
        normal_sum_inherited.noalias() += o.normal_sum_inherited;
        normal_weight_inherited        += o.normal_weight_inherited;
        inherited_decay_ref    = std::max(inherited_decay_ref, o.inherited_decay_ref);
        inherited_count       += o.inherited_count;
        inherited_from_scan    = std::max(inherited_from_scan, o.inherited_from_scan);
        inherited_level_mask  |= o.inherited_level_mask;
        combine_eogm_mass(o.eogm_surface, o.eogm_free, o.eogm_unknown, o.eogm_conflict);
        last_hit_scan      = std::max(last_hit_scan,      o.last_hit_scan);
        last_evidence_scan = std::max(last_evidence_scan, o.last_evidence_scan);
    }

    bool has_inherited() const { return weight_inherited > 1e-12; }

    double inherited_decay_scale() const {
        if (!has_inherited()) return 0.0;
        if (weight_sum <= 1e-12 || hit_count <= 0) return 1.0;
        double ref = inherited_decay_ref > 1e-12 ? inherited_decay_ref : std::max(weight_inherited, 1e-12);
        return std::clamp(ref / (ref + std::max(0.0, weight_sum)), 0.0, 1.0);
    }

    Mat3   A_eff() const { return A + inherited_decay_scale() * A_inherited; }
    Vec3   b_eff() const { return b + inherited_decay_scale() * b_inherited; }
    double c_eff() const { return c + inherited_decay_scale() * c_inherited; }
    double weight_eff() const { return weight_sum + inherited_decay_scale() * weight_inherited; }
    Vec3   normal_sum_eff() const { return normal_sum + inherited_decay_scale() * normal_sum_inherited; }
    double normal_weight_eff() const { return normal_weight + inherited_decay_scale() * normal_weight_inherited; }

    void clear_inherited() {
        A_inherited.setZero();
        b_inherited.setZero();
        c_inherited = 0.0;
        weight_inherited = 0.0;
        normal_sum_inherited.setZero();
        normal_weight_inherited = 0.0;
        inherited_decay_ref = 0.0;
        inherited_count = 0;
        inherited_from_scan = -1;
        inherited_level_mask = 0;
    }

    void add_inherited_plane(const Vec3& p, const Vec3& n_in, double w,
                             int scan_idx, int level, double decay_ref) {
        Vec3 n = n_in;
        double nn = n.norm();
        if (nn < 1e-12 || w <= 0.0) return;
        n /= nn;
        const double n_dot_p = n.dot(p);
        A_inherited.noalias() += w * (n * n.transpose());
        b_inherited.noalias() += (w * n_dot_p) * n;
        c_inherited           += w * n_dot_p * n_dot_p;
        weight_inherited      += w;
        normal_sum_inherited.noalias() += w * n;
        normal_weight_inherited        += w;
        inherited_decay_ref = std::max(inherited_decay_ref, std::max(decay_ref, w));
        inherited_count++;
        inherited_from_scan = std::max(inherited_from_scan, scan_idx);
        if (level >= 0 && level < 31) inherited_level_mask |= (1u << level);
    }

    double occupancy_score() const {
        double tot = hit_weight + miss_weight;
        return tot > 1e-12 ? (hit_weight - miss_weight) / tot : 0.0;
    }

    double eogm_bel_surface() const { return eogm_surface; }
    double eogm_bel_free() const { return eogm_free; }
    double eogm_ignorance() const { return eogm_unknown; }
    double eogm_conflict_mass() const { return eogm_conflict; }
    // Conservative plausibility: retained conflict is not counted as positive support.
    double eogm_plaus_surface() const { return std::min(1.0, eogm_surface + eogm_unknown); }

    double ray_normal_fraction() const {
        return hit_weight > 1e-12 ? (ray_normal_weight / hit_weight) : 0.0;
    }

    int confirmed_hit_count() const {
        int weak = std::max(0, promoted_seed_count);
        return std::max(0, hit_count - weak);
    }

    int confidence_tier() const {
        if (confirmed_hit_count() > 0) return 0;
        if (parent_supported_seed_count > 0) return 2;
        if (promoted_seed_count > 0) return 1;
        if (has_inherited()) return 6;
        return 0;
    }

    static const char* tier_name(int tier) {
        switch (tier) {
            case 1: return "weak_promoted";
            case 2: return "parent_supported";
            case 3: return "seed_pending";
            case 4: return "generated_hypothesis";
            case 5: return "scaffold";
            case 6: return "inherited";
            default: return "confirmed";
        }
    }

    double scan_boundary_ratio() const {
        return hit_weight > 1e-12 ? boundary_weight / hit_weight : 0.0;
    }

    // surface / free / unknown. Defaults bias toward calling things "surface"
    // in mixed-evidence cells (occlusion / glancing) and require strong
    // negative evidence to call something "free".
    enum class Label { SURFACE, FREE, UNKNOWN };
    Label label(const Settings& s) const {
        if ((hit_count + miss_count) < s.min_evidence) return Label::UNKNOWN;
        double sc = occupancy_score();
        if (sc >= s.thresh_surface) return Label::SURFACE;
        if (sc <= s.thresh_free)    return Label::FREE;
        return Label::UNKNOWN;
    }
    static const char* label_name(Label l) {
        switch (l) {
            case Label::SURFACE: return "surface";
            case Label::FREE:    return "free";
            default:             return "unknown";
        }
    }

    Vec3 mean_normal() const {
        const double w = normal_weight_eff();
        if (w < 1e-12) return Vec3::Zero();
        Vec3 n = normal_sum_eff() / w;
        double nn = n.norm();
        return nn > 1e-12 ? Vec3(n / nn) : n;
    }

    // In [0, 1]. 1 = unimodal normals; low = multi-modal -> split candidate.
    double normal_consistency() const {
        const double w = normal_weight_eff();
        if (w < 1e-12) return 0.0;
        return (normal_sum_eff() / w).norm();
    }

    // Per-observation residual at x.  Genuine flat patch -> ~0.
    // Multiple unrelated surfaces in one cell -> orders of magnitude higher
    // (the "split me" signal from Phase 1).
    double residual_per_observation(const Vec3& x) const {
        const double w = weight_eff();
        if (w < 1e-12) return std::nan("");
        double f = x.transpose() * A_eff() * x;
        f += c_eff();
        f -= 2.0 * b_eff().dot(x);
        return f / w;
    }

    // Regularised solve: argmin_x  x^T A x - 2 b^T x + c + lambda_p ||x - p||^2
    // Stationarity: (A + lambda_p I) x = b + lambda_p p.  LDLT is exact on
    // the symmetric PSD system (A is PSD by construction, regularisation is
    // strictly positive so the sum is SPD).
    Vec3 solve_vertex(const Vec3& anchor, double lambda_p,
                      bool clamp, const Vec3& lo, const Vec3& hi) const {
        Mat3 A_reg = A_eff() + lambda_p * Mat3::Identity();
        Vec3 rhs   = b_eff() + lambda_p * anchor;
        Vec3 x     = A_reg.ldlt().solve(rhs);
        if (clamp) {
            x = x.cwiseMax(lo).cwiseMin(hi);
        }
        return x;
    }

    struct EigenInfo {
        Vec3 evals;        // descending
        Mat3 evecs;        // columns aligned with evals (descending)
        const char* kind;  // "flat" / "edge" / "corner"
    };
    // QEM eigenstructure -> local geometry classification.
    // Pancake (one dominant eigenvalue) = flat patch.
    // Cigar (two dominant)              = edge.
    // Ball  (three dominant)            = corner.
    EigenInfo eigen_analysis(const Settings& s) const {
        EigenInfo info;
        Eigen::SelfAdjointEigenSolver<Mat3> eig(A_eff());
        // ascending -> reverse to descending
        for (int i = 0; i < 3; i++) info.evals[i] = eig.eigenvalues()[2 - i];
        for (int c = 0; c < 3; c++) info.evecs.col(c) = eig.eigenvectors().col(2 - c);

        double l1 = std::max(info.evals[0], 1e-12);
        double r2 = info.evals[1] / l1;
        double r3 = info.evals[2] / l1;
        if (r2 < s.rank2_ratio_thresh)      info.kind = "flat";
        else if (r3 < s.rank3_ratio_thresh) info.kind = "edge";
        else                                info.kind = "corner";
        return info;
    }
};

// ========================================================================= //
// 4. KDTree3D (used only for PCA normal fallback; same shape as qem_mesh)    //
// ========================================================================= //

class KDTree3D {
public:
    struct DistIdx {
        double dist; int idx;
        bool operator<(const DistIdx& o) const { return dist < o.dist; }
    };

    void build(const std::vector<Vec3>& points) {
        pts_ = points;
        int n = (int)pts_.size();
        if (n == 0) { root_ = -1; return; }
        idx_.resize(n); std::iota(idx_.begin(), idx_.end(), 0);
        nodes_.clear(); nodes_.reserve(2 * n);
        root_ = build_rec(0, n, 0);
    }

    bool empty() const { return root_ < 0; }

    void knn(const Vec3& q, int k,
             std::vector<double>& dists,
             std::vector<int>& indices) const {
        dists.clear(); indices.clear();
        if (root_ < 0 || k <= 0) return;
        std::priority_queue<DistIdx> heap;
        knn_rec(root_, q, k, heap);
        int m = (int)heap.size();
        dists.resize(m); indices.resize(m);
        for (int i = m - 1; i >= 0; --i) {
            dists[i] = std::sqrt(heap.top().dist);
            indices[i] = heap.top().idx;
            heap.pop();
        }
    }

private:
    struct Node { int point_idx, left, right, axis; double split; };
    std::vector<Node> nodes_;
    std::vector<Vec3> pts_;
    std::vector<int>  idx_;
    int root_ = -1;

    int build_rec(int lo, int hi, int depth) {
        if (lo >= hi) return -1;
        int axis = depth % 3, mid = (lo + hi) / 2;
        std::nth_element(idx_.begin() + lo, idx_.begin() + mid, idx_.begin() + hi,
            [&](int a, int b) { return pts_[a][axis] < pts_[b][axis]; });
        int ni = (int)nodes_.size();
        nodes_.push_back({idx_[mid], -1, -1, axis, pts_[idx_[mid]][axis]});
        nodes_[ni].left  = build_rec(lo, mid, depth + 1);
        nodes_[ni].right = build_rec(mid + 1, hi, depth + 1);
        return ni;
    }

    void knn_rec(int ni, const Vec3& q, int k,
                 std::priority_queue<DistIdx>& heap) const {
        if (ni < 0) return;
        const Node& nd = nodes_[ni];
        double d2 = (pts_[nd.point_idx] - q).squaredNorm();
        if ((int)heap.size() < k) heap.push({d2, nd.point_idx});
        else if (d2 < heap.top().dist) { heap.pop(); heap.push({d2, nd.point_idx}); }
        double diff = q[nd.axis] - nd.split;
        int first  = diff < 0 ? nd.left  : nd.right;
        int second = diff < 0 ? nd.right : nd.left;
        knn_rec(first, q, k, heap);
        double worst = (int)heap.size() < k ? 1e30 : heap.top().dist;
        if (diff * diff < worst) knn_rec(second, q, k, heap);
    }
};

// ========================================================================= //
// 5. PCD reader (binary + ASCII)                                             //
// ========================================================================= //

struct PointCloud {
    std::vector<Vec3> points;
    std::vector<Vec3> normals;  // empty if PCD has none
    // Raw PCD index (pre-NaN-skip) for each kept point. Critical for organized
    // PCDs: row/col indexing assumes the raw index, not the compacted one.
    std::vector<int> original_indices;
    // Optional scan-line metadata. Organized PCDs give width/height; many
    // spinning LiDAR PCDs expose a ring/channel/laser_id field.
    int width = 0;
    int height = 1;
    std::vector<int> rings;     // empty if no ring/channel-like field
    bool organized() const { return width > 0 && height > 1; }
    bool has_ring() const { return rings.size() == points.size(); }
};

static Eigen::Quaterniond quat_xyzw_to_eigen(double qx, double qy, double qz, double qw) {
    Eigen::Quaterniond q(qw, qx, qy, qz);
    if (q.norm() < 1e-12) return Eigen::Quaterniond::Identity();
    q.normalize();
    return q;
}

PointCloud read_pcd(const std::string& path) {
    PointCloud pc;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open " << path << "\n";
        return pc;
    }
    std::string line;
    std::vector<std::string> fields;
    std::vector<int> sizes;
    std::vector<char> types;
    std::vector<int> counts;
    int num_points = 0;
    bool is_binary = false;
    int point_size = 0;

    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string key; iss >> key;
        if (key == "FIELDS") {
            std::string s; while (iss >> s) fields.push_back(s);
        } else if (key == "SIZE") {
            int s; while (iss >> s) sizes.push_back(s);
        } else if (key == "TYPE") {
            char c; while (iss >> c) types.push_back(c);
        } else if (key == "COUNT") {
            int c; while (iss >> c) counts.push_back(c);
        } else if (key == "POINTS") {
            iss >> num_points;
        } else if (key == "WIDTH") {
            iss >> pc.width;
            if (num_points == 0) num_points = pc.width;
        } else if (key == "HEIGHT") {
            iss >> pc.height;
        } else if (key == "DATA") {
            std::string fmt; iss >> fmt;
            is_binary = (fmt == "binary");
            if (fmt == "binary_compressed") {
                std::cerr << path << ": binary_compressed not supported\n";
                return pc;
            }
            break;
        }
    }
    if (counts.empty()) counts.assign(fields.size(), 1);
    if (types.empty()) types.assign(fields.size(), 'F');
    if (sizes.empty()) sizes.assign(fields.size(), 4);
    if (pc.height <= 0) pc.height = 1;

    // Build field offsets accounting for COUNT.
    int xi=-1, yi=-1, zi=-1, nxi=-1, nyi=-1, nzi=-1, ringi=-1;
    std::vector<int> offsets(fields.size());
    for (int i = 0; i < (int)fields.size(); i++) {
        offsets[i] = point_size;
        std::string name = fields[i];
        for (char& c : name) c = (char)std::tolower((unsigned char)c);
        if      (name == "x")        xi  = i;
        else if (name == "y")        yi  = i;
        else if (name == "z")        zi  = i;
        else if (name == "normal_x") nxi = i;
        else if (name == "normal_y") nyi = i;
        else if (name == "normal_z") nzi = i;
        else if (name == "ring" || name == "channel" || name == "laser_id" ||
                 name == "laserid" || name == "scan_id" || name == "row") ringi = i;
        point_size += sizes[i] * counts[i];
    }
    bool has_normals = (nxi >= 0 && nyi >= 0 && nzi >= 0);
    bool has_ring = (ringi >= 0);
    pc.points.reserve(num_points);
    pc.original_indices.reserve(num_points);
    if (has_normals) pc.normals.reserve(num_points);
    if (has_ring) pc.rings.reserve(num_points);

    auto read_binary_scalar = [&](const std::vector<char>& buf, int fi) -> double {
        const char* ptr = buf.data() + offsets[fi];
        char ty = std::toupper((unsigned char)types[fi]);
        int sz = sizes[fi];
        if (ty == 'F') {
            if (sz == 4) { float v; std::memcpy(&v, ptr, 4); return v; }
            if (sz == 8) { double v; std::memcpy(&v, ptr, 8); return v; }
        } else if (ty == 'I') {
            if (sz == 1) { int8_t v; std::memcpy(&v, ptr, 1); return v; }
            if (sz == 2) { int16_t v; std::memcpy(&v, ptr, 2); return v; }
            if (sz == 4) { int32_t v; std::memcpy(&v, ptr, 4); return v; }
            if (sz == 8) { int64_t v; std::memcpy(&v, ptr, 8); return (double)v; }
        } else if (ty == 'U') {
            if (sz == 1) { uint8_t v; std::memcpy(&v, ptr, 1); return v; }
            if (sz == 2) { uint16_t v; std::memcpy(&v, ptr, 2); return v; }
            if (sz == 4) { uint32_t v; std::memcpy(&v, ptr, 4); return v; }
            if (sz == 8) { uint64_t v; std::memcpy(&v, ptr, 8); return (double)v; }
        }
        return 0.0;
    };

    if (is_binary) {
        std::vector<char> buf(point_size);
        for (int i = 0; i < num_points; i++) {
            f.read(buf.data(), point_size);
            if (!f) break;
            float x, y, z;
            std::memcpy(&x, buf.data() + offsets[xi], 4);
            std::memcpy(&y, buf.data() + offsets[yi], 4);
            std::memcpy(&z, buf.data() + offsets[zi], 4);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            pc.points.push_back(Vec3(x, y, z));
            pc.original_indices.push_back(i);
            if (has_normals) {
                float nx, ny, nz;
                std::memcpy(&nx, buf.data() + offsets[nxi], 4);
                std::memcpy(&ny, buf.data() + offsets[nyi], 4);
                std::memcpy(&nz, buf.data() + offsets[nzi], 4);
                pc.normals.push_back(Vec3(nx, ny, nz));
            }
            if (has_ring) {
                pc.rings.push_back((int)std::llround(read_binary_scalar(buf, ringi)));
            }
        }
    } else {
        for (int i = 0; i < num_points; i++) {
            if (!std::getline(f, line)) break;
            std::istringstream iss(line);
            std::vector<float> vals;
            float v;
            while (iss >> v) vals.push_back(v);
            int need = std::max({xi, yi, zi});
            if ((int)vals.size() <= need) continue;
            float x = vals[xi], y = vals[yi], z = vals[zi];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            pc.points.push_back(Vec3(x, y, z));
            pc.original_indices.push_back(i);
            if (has_normals && (int)vals.size() > std::max({nxi, nyi, nzi})) {
                pc.normals.push_back(Vec3(vals[nxi], vals[nyi], vals[nzi]));
            } else if (has_normals) {
                pc.normals.push_back(Vec3::Zero());
            }
            if (has_ring && (int)vals.size() > ringi) {
                pc.rings.push_back((int)std::llround(vals[ringi]));
            }
        }
    }
    // Treat an all-zero normals array as "no normals" so the downstream
    // pipeline falls through to scan-line normals / PCA estimation.
    if (has_normals) {
        bool any_nonzero = false;
        for (const auto& n : pc.normals)
            if (n.squaredNorm() > 1e-12) { any_nonzero = true; break; }
        if (!any_nonzero) pc.normals.clear();
    }
    if (has_ring && pc.rings.size() != pc.points.size()) pc.rings.clear();
    return pc;
}

// ========================================================================= //
// 6. Pose parsing (TUM / G2O / CSV)                                          //
// ========================================================================= //

static Mat4 quat_to_matrix(double x, double y, double z,
                            double qx, double qy, double qz, double qw) {
    Mat4 T = Mat4::Identity();
    T(0,3) = x; T(1,3) = y; T(2,3) = z;
    double n = qx*qx + qy*qy + qz*qz + qw*qw;
    if (n < 1e-12) return T;
    double s = 1.0 / std::sqrt(n);
    qx *= s; qy *= s; qz *= s; qw *= s;
    T(0,0) = 1 - 2*(qy*qy + qz*qz);
    T(0,1) =     2*(qx*qy - qz*qw);
    T(0,2) =     2*(qx*qz + qy*qw);
    T(1,0) =     2*(qx*qy + qz*qw);
    T(1,1) = 1 - 2*(qx*qx + qz*qz);
    T(1,2) =     2*(qy*qz - qx*qw);
    T(2,0) =     2*(qx*qz - qy*qw);
    T(2,1) =     2*(qy*qz + qx*qw);
    T(2,2) = 1 - 2*(qx*qx + qy*qy);
    return T;
}

struct PoseEntry { std::string key; Mat4 pose; };

static std::vector<PoseEntry> parse_poses_tum(const std::string& path) {
    std::vector<PoseEntry> poses;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string ts;
        double x, y, z, qx, qy, qz, qw;
        if (!(iss >> ts >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
        auto dot = ts.find('.');
        if (dot == std::string::npos) continue;
        long sec, nsec;
        try {
            sec  = std::stol(ts.substr(0, dot));
            // Pad nsec to 9 digits if shorter (TUM is loose about this).
            std::string nsec_s = ts.substr(dot + 1);
            if (nsec_s.size() < 9) nsec_s.append(9 - nsec_s.size(), '0');
            else if (nsec_s.size() > 9) nsec_s.resize(9);
            nsec = std::stol(nsec_s);
        } catch (...) { continue; }
        char key[32];
        std::snprintf(key, sizeof(key), "%010ld_%09ld", sec, nsec);
        poses.push_back({key, quat_to_matrix(x, y, z, qx, qy, qz, qw)});
    }
    return poses;
}

static std::vector<PoseEntry> parse_poses_g2o(const std::string& path) {
    std::vector<PoseEntry> poses;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string tag;
        int id;
        double x, y, z, qx, qy, qz, qw;
        long sec, nsec;
        if (!(iss >> tag) || tag != "VERTEX_SE3:QUAT_TIME") continue;
        if (!(iss >> id >> x >> y >> z >> qx >> qy >> qz >> qw >> sec >> nsec)) continue;
        char key[32];
        std::snprintf(key, sizeof(key), "%010ld_%09ld", sec, nsec);
        poses.push_back({key, quat_to_matrix(x, y, z, qx, qy, qz, qw)});
    }
    return poses;
}

static std::vector<PoseEntry> parse_poses_csv(const std::string& path) {
    std::vector<PoseEntry> poses;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream iss(line);
        int idx;
        long sec, nsec;
        double x, y, z, qx, qy, qz, qw;
        if (!(iss >> idx >> sec >> nsec >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
        char key[32];
        std::snprintf(key, sizeof(key), "%010ld_%09ld", sec, nsec);
        poses.push_back({key, quat_to_matrix(x, y, z, qx, qy, qz, qw)});
    }
    return poses;
}

// ========================================================================= //
// 7. Dataset assembly: pair PCDs to poses by filename timestamp              //
// ========================================================================= //

struct ScanEntry { std::string pcd_path; Mat4 pose; };

static std::vector<ScanEntry> load_dataset(const std::string& pcd_folder,
                                            const std::string& pose_file) {
    std::vector<std::string> pcds;
    for (auto& e : fs::directory_iterator(pcd_folder))
        if (e.path().extension() == ".pcd") pcds.push_back(e.path().string());
    std::sort(pcds.begin(), pcds.end());

    std::string ext = fs::path(pose_file).extension().string();
    std::vector<PoseEntry> raw;
    const char* fmt;
    if (ext == ".g2o" || ext == ".slam") { raw = parse_poses_g2o(pose_file); fmt = "G2O"; }
    else if (ext == ".csv")              { raw = parse_poses_csv(pose_file); fmt = "CSV"; }
    else                                  { raw = parse_poses_tum(pose_file); fmt = "TUM"; }
    std::printf("  Parsed %d poses (%s)\n", (int)raw.size(), fmt);

    std::unordered_map<std::string, Mat4> pose_map;
    for (auto& pe : raw) pose_map[pe.key] = pe.pose;

    std::vector<ScanEntry> dataset;
    auto is_digits = [](const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(),
            [](unsigned char c) { return std::isdigit(c) != 0; });
    };
    for (auto& pcd : pcds) {
        std::string stem = fs::path(pcd).stem().string();
        std::vector<std::string> tokens;
        {
            std::stringstream ss(stem);
            std::string tok;
            while (std::getline(ss, tok, '_')) if (!tok.empty()) tokens.push_back(tok);
        }
        // Try every adjacent (sec, nsec) pair from the right; first match wins.
        for (int i = (int)tokens.size() - 1; i >= 1; --i) {
            const std::string& sec_s  = tokens[i - 1];
            const std::string& nsec_s = tokens[i];
            if (!is_digits(sec_s) || !is_digits(nsec_s)) continue;
            try {
                long sec  = std::stol(sec_s);
                long nsec = std::stol(nsec_s);
                char key[32];
                std::snprintf(key, sizeof(key), "%010ld_%09ld", sec, nsec);
                auto it = pose_map.find(key);
                if (it != pose_map.end()) {
                    dataset.push_back({pcd, it->second});
                    break;
                }
            } catch (...) {}
        }
    }
    std::printf("  Matched %d / %d PCDs to poses\n",
                (int)dataset.size(), (int)pcds.size());
    return dataset;
}

// ========================================================================= //
// 8. PCA normal estimation                                                   //
// ========================================================================= //
//
// Used only for PCDs that arrive without normals. For each point, run
// spatial PCA on its k nearest neighbours, take the smallest-eigenvalue
// eigenvector as the surface normal. Sign is NOT globally consistent --
// VoxelQEMMap::process_scan reorients toward the sensor afterwards.
// Confidence in [0, 1] from the ratio l0/l1: planar neighbourhood
// (l0 << l1) -> high confidence; isotropic / noisy -> low.

static void estimate_normals_pca(const std::vector<Vec3>& points, int k,
                                  std::vector<Vec3>& out_normals,
                                  std::vector<double>& out_confidence) {
    int N = (int)points.size();
    out_normals.assign(N, Vec3::Zero());
    out_confidence.assign(N, 0.0);
    if (N < 3) return;
    k = std::min(k, N);

    KDTree3D tree;
    tree.build(points);

    #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
    #endif
    for (int i = 0; i < N; i++) {
        std::vector<double> dists;
        std::vector<int> idx;
        tree.knn(points[i], k, dists, idx);
        if ((int)idx.size() < 3) continue;
        Vec3 centroid = Vec3::Zero();
        for (int j : idx) centroid += points[j];
        centroid /= (double)idx.size();
        Mat3 cov = Mat3::Zero();
        for (int j : idx) {
            Vec3 d = points[j] - centroid;
            cov.noalias() += d * d.transpose();
        }
        cov /= (double)idx.size();
        Eigen::SelfAdjointEigenSolver<Mat3> eig(cov);
        // Smallest-eigenvalue eigenvector is the surface normal direction.
        out_normals[i] = eig.eigenvectors().col(0);
        double l0 = eig.eigenvalues()[0];
        double l1 = eig.eigenvalues()[1];
        double conf = 1.0 - l0 / (l1 + 1e-12);
        out_confidence[i] = std::clamp(conf, 0.0, 1.0);
    }
}


// ========================================================================= //
// 8b. LiDAR scan-line / range-image support                                //
// ========================================================================= //
//
// This optional front-end uses the geometry of spinning/organized LiDAR scans
// when available. It does not replace QEM. It improves the normals fed into QEM
// and prevents NVT smoothing across depth discontinuities. Sources of scan-line
// layout, in priority order:
//   1. organized PCD WIDTH/HEIGHT -> row/column from original point index
//   2. ring/channel/laser_id field -> row from ring, column from azimuth
//   3. --scanline_rows N -> row by elevation quantization, column by azimuth
// If none of these is available, all scan-line gates gracefully disable.

struct ScanlinePointInfo {
    int row = -1;
    int col = -1;
    int cols = 0;
    double azimuth = 0.0;       // radians in sensor frame
    double elevation = 0.0;     // radians in sensor frame
    double range = 0.0;
    bool valid = false;
    bool boundary = false;       // depth jump / scan gap in beam-neighborhood
    double boundary_score = 0.0; // jump-neighbors / checked-neighbors
    int prev_same_ring = -1;
    int next_same_ring = -1;
    int upper_ring_near = -1;
    int lower_ring_near = -1;
    double gap_prev = 0.0;
    double gap_next = 0.0;
};

static uint64_t scanline_grid_key(int row, int col) {
    return ((uint64_t)(uint32_t)row << 32) | (uint32_t)col;
}

static double wrap_angle_pi(double a) {
    constexpr double PI = 3.14159265358979323846;
    while (a <= -PI) a += 2.0 * PI;
    while (a >   PI) a -= 2.0 * PI;
    return a;
}

static double circular_angle_dist(double a, double b) {
    return std::abs(wrap_angle_pi(a - b));
}

static int circular_col_dist(int a, int b, int cols) {
    int d = std::abs(a - b);
    return cols > 0 ? std::min(d, cols - d) : d;
}

static int quantize_azimuth_col(const Vec3& p, int cols) {
    if (cols <= 0) cols = 2048;
    constexpr double PI = 3.14159265358979323846;
    double az = std::atan2(p[1], p[0]); // [-pi, pi]
    double u = (az + PI) / (2.0 * PI);
    int c = (int)std::floor(u * cols);
    if (c < 0) c = 0;
    if (c >= cols) c = cols - 1;
    return c;
}

static double scanline_jump_threshold(double r0, double r1, const Settings& s) {
    return s.scanline_jump_abs + s.scanline_jump_rel * std::min(r0, r1);
}

struct PandarChannelAngles {
    std::vector<double> elev;  // radians, channel order bottom->top / low->high elevation
    std::vector<double> az;    // radians, same order, intrinsic horizontal offset
    bool has_az = false;
};

static double deg2rad(double x) {
    return x * 3.14159265358979323846 / 180.0;
}

static PandarChannelAngles pandar_qt64_design_angles() {
    // PandarQT Appendix I design values, Channel 1 bottom -> Channel 64 top.
    // Accurate values should come from the unit angle-correction file when available.
    static const double AZ_DEG[64] = {
         8.736,  8.314,  7.964,  7.669,  7.417,  7.198,  7.007,  6.838,
         6.688,  6.554,  6.434,  6.326,  6.228,  6.140,  6.059,  5.987,
        -5.270, -5.216, -5.167, -5.123, -5.083, -5.047, -5.016, -4.988,
        -4.963, -4.942, -4.924, -4.910, -4.898, -4.889, -4.884, -4.881,
         5.493,  5.496,  5.502,  5.512,  5.525,  5.541,  5.561,  5.584,
         5.611,  5.642,  5.676,  5.716,  5.759,  5.808,  5.862,  5.921,
        -5.330, -5.396, -5.469, -5.550, -5.640, -5.740, -5.850, -5.974,
        -6.113, -6.269, -6.447, -6.651, -6.887, -7.163, -7.493, -7.892
    };
    static const double EL_DEG[64] = {
        -52.121, -49.785, -47.577, -45.477, -43.465, -41.528, -39.653, -37.831,
        -36.055, -34.320, -32.619, -30.950, -29.308, -27.690, -26.094, -24.517,
        -22.964, -21.420, -19.889, -18.372, -16.865, -15.368, -13.880, -12.399,
        -10.925,  -9.457,  -7.994,  -6.535,  -5.079,  -3.626,  -2.175,  -0.725,
          0.725,   2.175,   3.626,   5.079,   6.534,   7.993,   9.456,  10.923,
         12.397,  13.877,  15.365,  16.861,  18.368,  19.885,  21.415,  22.959,
         24.524,  26.101,  27.697,  29.315,  30.957,  32.627,  34.328,  36.064,
         37.840,  39.662,  41.537,  43.475,  45.487,  47.587,  49.795,  52.133
    };
    PandarChannelAngles a;
    a.elev.resize(64);
    a.az.resize(64);
    for (int i = 0; i < 64; i++) {
        a.elev[i] = deg2rad(EL_DEG[i]);
        a.az[i]   = deg2rad(AZ_DEG[i]);
    }
    a.has_az = true;
    return a;
}

static void sort_pandar_angles_by_elevation(PandarChannelAngles& a) {
    if (a.elev.empty()) return;
    if (a.az.size() < a.elev.size()) a.az.resize(a.elev.size(), 0.0);
    std::vector<int> perm(a.elev.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int x, int y) {
        return a.elev[x] < a.elev[y];
    });
    std::vector<double> elev_sorted;
    std::vector<double> az_sorted;
    elev_sorted.reserve(a.elev.size());
    az_sorted.reserve(a.elev.size());
    for (int idx : perm) {
        elev_sorted.push_back(a.elev[idx]);
        az_sorted.push_back(a.az[idx]);
    }
    a.elev.swap(elev_sorted);
    a.az.swap(az_sorted);
}

static PandarChannelAngles load_pandar_calib_angles(const Settings& s) {
    PandarChannelAngles out;
    if (s.pandar_qt64_calib.empty()) return out;
    std::ifstream f(s.pandar_qt64_calib);
    if (!f) {
        std::fprintf(stderr, "[PandarQT64] warning: cannot open calibration file '%s'; using design channel table\n",
                     s.pandar_qt64_calib.c_str());
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (vals.empty()) continue;

        // Accepted formats:
        //   elevation
        //   channel elevation
        //   channel elevation azimuth_offset
        // Channel is 1-based if provided. Values are degrees.
        int channel = -1;
        double elev_deg = 0.0;
        double az_deg = 0.0;
        bool has_az = false;
        if (vals.size() >= 3) {
            channel = (int)std::llround(vals[0]);
            elev_deg = vals[1];
            az_deg = vals[2];
            has_az = true;
        } else if (vals.size() == 2) {
            // If first value looks like a channel id, interpret as channel,elevation.
            if (vals[0] >= 1.0 && vals[0] <= 64.0 && std::abs(vals[0] - std::round(vals[0])) < 1e-6) {
                channel = (int)std::llround(vals[0]);
                elev_deg = vals[1];
            } else {
                elev_deg = vals[0];
                az_deg = vals[1];
                has_az = true;
            }
        } else {
            elev_deg = vals[0];
        }

        if (channel >= 1 && channel <= s.pandar_qt64_channels) {
            if ((int)out.elev.size() < s.pandar_qt64_channels) {
                out.elev.assign(s.pandar_qt64_channels, std::numeric_limits<double>::quiet_NaN());
                out.az.assign(s.pandar_qt64_channels, 0.0);
            }
            int idx = channel - 1;
            out.elev[idx] = deg2rad(elev_deg);
            if (has_az) { out.az[idx] = deg2rad(az_deg); out.has_az = true; }
        } else {
            out.elev.push_back(deg2rad(elev_deg));
            out.az.push_back(has_az ? deg2rad(az_deg) : 0.0);
            if (has_az) out.has_az = true;
        }
    }

    if ((int)out.elev.size() == s.pandar_qt64_channels) {
        bool ok = true;
        for (double e : out.elev) if (!std::isfinite(e)) { ok = false; break; }
        if (ok) {
            // Calibration files are often channel-indexed, not sorted by elevation.
            // Downstream row±1 means adjacent beams, so sort into physical vertical order.
            sort_pandar_angles_by_elevation(out);
            return out;
        }
    }
    std::fprintf(stderr, "[PandarQT64] warning: calibration file has %zu usable elevations, expected %d; using design channel table\n",
                 out.elev.size(), s.pandar_qt64_channels);
    return PandarChannelAngles{};
}

static std::vector<double> infer_elevation_centers_1d(const std::vector<double>& elevations,
                                                       int K, double min_deg, double max_deg,
                                                       bool data_driven) {
    std::vector<double> centers;
    if (K <= 0) return centers;
    centers.resize(K);
    double lo = deg2rad(min_deg);
    double hi = deg2rad(max_deg);
    if (!data_driven || elevations.empty()) {
        for (int k = 0; k < K; k++) {
            double u = K == 1 ? 0.5 : (double)k / (double)(K - 1);
            centers[k] = lo + u * (hi - lo);
        }
        return centers;
    }
    std::vector<double> e;
    e.reserve(elevations.size());
    for (double x : elevations) if (std::isfinite(x) && x >= lo - 0.05 && x <= hi + 0.05) e.push_back(x);
    if ((int)e.size() < K) {
        for (int k = 0; k < K; k++) {
            double u = K == 1 ? 0.5 : (double)k / (double)(K - 1);
            centers[k] = lo + u * (hi - lo);
        }
        return centers;
    }
    std::sort(e.begin(), e.end());
    for (int k = 0; k < K; k++) {
        double u = (k + 0.5) / (double)K;
        int idx = std::clamp((int)std::floor(u * (double)e.size()), 0, (int)e.size() - 1);
        centers[k] = e[idx];
    }
    for (int it = 0; it < 8; it++) {
        std::vector<double> sum(K, 0.0);
        std::vector<int> cnt(K, 0);
        for (double x : e) {
            int best = 0;
            double bd = std::abs(x - centers[0]);
            for (int k = 1; k < K; k++) {
                double d = std::abs(x - centers[k]);
                if (d < bd) { bd = d; best = k; }
            }
            sum[best] += x;
            cnt[best]++;
        }
        for (int k = 0; k < K; k++) if (cnt[k] > 0) centers[k] = sum[k] / cnt[k];
        std::sort(centers.begin(), centers.end());
    }
    return centers;
}

static int nearest_center_index(const std::vector<double>& centers, double value) {
    // Linear scan is intentionally used here: 64 elements is tiny, and it is
    // robust even if a hand-written calibration file is not sorted.
    if (centers.empty()) return -1;
    int best = 0;
    double bestd = std::abs(value - centers[0]);
    for (int i = 1; i < (int)centers.size(); i++) {
        double d = std::abs(value - centers[i]);
        if (d < bestd) { bestd = d; best = i; }
    }
    return best;
}

static int nearest_by_azimuth(const std::vector<int>& vec, int self_idx,
                              const std::vector<ScanlinePointInfo>& info,
                              double max_daz) {
    if (vec.empty()) return -1;
    double az = info[self_idx].azimuth;
    int best = -1;
    double bestd = 1e30;
    for (int idx : vec) {
        if (idx == self_idx) continue;
        if (idx < 0 || idx >= (int)info.size() || !info[idx].valid) continue;
        double d = circular_angle_dist(az, info[idx].azimuth);
        if (d < bestd) { bestd = d; best = idx; }
    }
    return bestd <= max_daz ? best : -1;
}

static void build_scanline_info(const PointCloud& pc,
                                const std::vector<int>& orig_indices,
                                const std::vector<Vec3>& pts_sensor,
                                const Settings& s,
                                std::vector<ScanlinePointInfo>& info) {
    int N = (int)pts_sensor.size();
    info.assign(N, ScanlinePointInfo{});
    if (!s.enable_scanline || N == 0) return;

    bool organized = pc.organized();
    bool ringed = pc.has_ring();
    bool pandar = s.pandar_qt64_mode && !organized && !ringed;
    bool quantized = (!organized && !ringed && !pandar && s.scanline_rows > 1);
    if (!organized && !ringed && !quantized && !pandar) return;

    int cols = 0;
    if (organized) cols = pc.width;
    else cols = s.scanline_cols > 0 ? s.scanline_cols : (s.pandar_qt64_mode ? 600 : 2048);

    std::vector<double> elevations;
    elevations.reserve(N);
    for (const auto& p : pts_sensor) {
        double r = p.norm();
        if (r < 1e-9) continue;
        elevations.push_back(std::asin(std::clamp(p[2] / r, -1.0, 1.0)));
    }
    PandarChannelAngles pandar_angles;
    std::vector<double> pandar_centers;
    std::vector<double> pandar_az_offsets;
    if (pandar) {
        pandar_angles = load_pandar_calib_angles(s);
        if (pandar_angles.elev.empty()) {
            if (s.pandar_qt64_uniform || s.pandar_qt64_infer_rings) {
                pandar_angles.elev = infer_elevation_centers_1d(elevations,
                    s.pandar_qt64_channels,
                    s.pandar_qt64_vmin_deg,
                    s.pandar_qt64_vmax_deg,
                    s.pandar_qt64_infer_rings);
                pandar_angles.az.assign(pandar_angles.elev.size(), 0.0);
                pandar_angles.has_az = false;
            } else {
                pandar_angles = pandar_qt64_design_angles();
            }
        }
        pandar_centers = pandar_angles.elev;
        pandar_az_offsets = pandar_angles.az;
    }

    double min_el = 0.0, max_el = 0.0;
    if (quantized) {
        min_el =  1e30;
        max_el = -1e30;
        for (double el : elevations) { min_el = std::min(min_el, el); max_el = std::max(max_el, el); }
        if (!(max_el > min_el)) return;
    }

    for (int i = 0; i < N; i++) {
        int orig = i < (int)orig_indices.size() ? orig_indices[i] : i;
        int row = -1, col = -1;
        double r = pts_sensor[i].norm();
        if (r <= 1e-9) continue;
        double az = std::atan2(pts_sensor[i][1], pts_sensor[i][0]);
        double el = std::asin(std::clamp(pts_sensor[i][2] / r, -1.0, 1.0));
        if (organized && orig >= 0 && pc.width > 0) {
            row = orig / pc.width;
            col = orig % pc.width;
            if (row < 0 || row >= pc.height) row = -1;
        } else if (ringed && i < (int)pc.rings.size()) {
            // Ring fields are stored alongside kept points in PointCloud, so
            // use the compacted point index, not the raw original PCD index.
            row = pc.rings[i];
            col = quantize_azimuth_col(pts_sensor[i], cols);
        } else if (pandar) {
            int ci = nearest_center_index(pandar_centers, el);
            if (ci >= 0) {
                row = ci; // channel index in ascending elevation / design table order
                double az_for_col = az;
                // The PandarQT manual defines a per-channel intrinsic horizontal
                // offset. Subtract it so adjacent rings are matched by approximate
                // rotor/reference azimuth, while same-ring ordering is unchanged.
                if (ci < (int)pandar_az_offsets.size()) az_for_col = wrap_angle_pi(az - pandar_az_offsets[ci]);
                Vec3 azp(std::cos(az_for_col), std::sin(az_for_col), 0.0);
                col = quantize_azimuth_col(azp, cols);
                az = az_for_col;
            }
        } else if (quantized) {
            double u = (el - min_el) / (max_el - min_el + 1e-12);
            row = std::clamp((int)std::floor(u * s.scanline_rows), 0, s.scanline_rows - 1);
            col = quantize_azimuth_col(pts_sensor[i], cols);
        }
        if (row < 0 || col < 0) continue;
        info[i].row = row;
        info[i].col = col;
        info[i].cols = cols;
        info[i].range = r;
        info[i].azimuth = az;
        info[i].elevation = el;
        info[i].valid = true;
    }

    // Ring/azimuth beam graph: robust for unorganized spinning-LiDAR scans.
    std::unordered_map<int, std::vector<int>> by_row;
    by_row.reserve(128);
    for (int i = 0; i < N; i++) if (info[i].valid) by_row[info[i].row].push_back(i);
    for (auto& kv : by_row) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end(), [&](int a, int b){ return info[a].azimuth < info[b].azimuth; });
        int M = (int)v.size();
        if (M < 2) continue;
        std::vector<double> gaps;
        gaps.reserve(M);
        for (int k = 0; k < M; k++) {
            int i0 = v[k];
            int i1 = v[(k + 1) % M];
            double gap = circular_angle_dist(info[i0].azimuth, info[i1].azimuth);
            gaps.push_back(gap);
        }
        std::vector<double> gs = gaps;
        std::nth_element(gs.begin(), gs.begin() + gs.size()/2, gs.end());
        double med_gap = std::max(gs[gs.size()/2], 1e-6);
        // Prevent sparse rings from linking points many metres apart.
        // 0.10 rad ≈ 5.7°, while dense rings still use the median-gap rule.
        double max_neighbor_gap = std::min(2.5 * med_gap, 0.10);
        for (int k = 0; k < M; k++) {
            int cur = v[k];
            int prv = v[(k - 1 + M) % M];
            int nxt = v[(k + 1) % M];
            double gprev = circular_angle_dist(info[cur].azimuth, info[prv].azimuth);
            double gnext = circular_angle_dist(info[cur].azimuth, info[nxt].azimuth);
            info[cur].gap_prev = gprev;
            info[cur].gap_next = gnext;
            if (gprev <= max_neighbor_gap) info[cur].prev_same_ring = prv;
            if (gnext <= max_neighbor_gap) info[cur].next_same_ring = nxt;
        }
    }

    // Adjacent pseudo-ring neighbors: nearest azimuth in row-1/row+1 when present.
    double default_gap = (2.0 * 3.14159265358979323846) / std::max(cols, 1);
    double max_adj_az = std::max(3.0 * default_gap, 0.03); // ~1.7 deg lower bound
    for (int i = 0; i < N; i++) {
        if (!info[i].valid) continue;
        auto it_lo = by_row.find(info[i].row - 1);
        auto it_hi = by_row.find(info[i].row + 1);
        if (it_lo != by_row.end()) info[i].lower_ring_near = nearest_by_azimuth(it_lo->second, i, info, max_adj_az);
        if (it_hi != by_row.end()) info[i].upper_ring_near = nearest_by_azimuth(it_hi->second, i, info, max_adj_az);
    }

    for (int i = 0; i < N; i++) {
        if (!info[i].valid) continue;
        int checked = 0, jumps = 0;
        for (int j : {info[i].prev_same_ring, info[i].next_same_ring,
                      info[i].lower_ring_near, info[i].upper_ring_near}) {
            if (j < 0) continue;
            checked++;
            double dr = std::abs(info[i].range - info[j].range);
            if (dr > scanline_jump_threshold(info[i].range, info[j].range, s)) jumps++;
        }
        // Large gaps in same ring are also a boundary/uncertainty signal.
        if (info[i].prev_same_ring < 0 || info[i].next_same_ring < 0) {
            checked++;
            jumps++;
        }
        if (checked > 0) {
            info[i].boundary_score = (double)jumps / (double)checked;
            info[i].boundary = jumps > 0;
        }
    }
}

static bool scanline_compatible(int i, int j,
                                const std::vector<ScanlinePointInfo>& info,
                                const Settings& s) {
    if (i < 0 || j < 0 || i >= (int)info.size() || j >= (int)info.size()) return true;
    const auto& a = info[i];
    const auto& b = info[j];
    if (!a.valid || !b.valid) return true;
    int dr_row = std::abs(a.row - b.row);
    int dr_col = circular_col_dist(a.col, b.col, std::max(a.cols, b.cols));
    double dr = std::abs(a.range - b.range);
    double jump = scanline_jump_threshold(a.range, b.range, s);
    if (dr_row <= s.scanline_gate_rows && dr_col <= s.scanline_gate_cols)
        return dr <= jump;
    if ((a.boundary || b.boundary) && dr > 1.5 * jump) return false;
    return true;
}

static void estimate_normals_scanline(const std::vector<Vec3>& pts_sensor,
                                      const std::vector<ScanlinePointInfo>& info,
                                      const Settings& s,
                                      std::vector<Vec3>& out_normals,
                                      std::vector<double>& out_confidence) {
    int N = (int)pts_sensor.size();
    out_normals.assign(N, Vec3::Zero());
    out_confidence.assign(N, 0.0);
    if (!s.enable_scanline || !s.scanline_normals || N < 3 || (int)info.size() != N) return;

    #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static, 1024)
    #endif
    for (int i = 0; i < N; i++) {
        if (!info[i].valid) continue;
        std::vector<int> nb;
        for (int j : {info[i].prev_same_ring, info[i].next_same_ring,
                      info[i].lower_ring_near, info[i].upper_ring_near}) {
            if (j >= 0 && scanline_compatible(i, j, info, s)) nb.push_back(j);
        }
        if ((int)nb.size() < 2) continue;
        std::vector<Vec3> cands;
        cands.reserve(6);
        for (int a = 0; a < (int)nb.size(); a++) {
            for (int b = a + 1; b < (int)nb.size(); b++) {
                Vec3 va = pts_sensor[nb[a]] - pts_sensor[i];
                Vec3 vb = pts_sensor[nb[b]] - pts_sensor[i];
                Vec3 n = va.cross(vb);
                double nn = n.norm();
                if (nn < 1e-10) continue;
                n /= nn;
                cands.push_back(n);
            }
        }
        if (cands.empty()) continue;
        Vec3 ref = cands[0];
        Vec3 sum = Vec3::Zero();
        for (Vec3 n : cands) {
            if (n.dot(ref) < 0.0) n = -n;
            sum += n;
        }
        double sum_norm = sum.norm();
        if (sum_norm < 1e-9) continue;
        Vec3 n = sum / sum_norm;
        out_normals[i] = n;
        double agreement = sum_norm / (double)cands.size();
        double neighbor_factor = std::min(1.0, (double)nb.size() / 4.0);
        out_confidence[i] = std::clamp(0.20 + 0.80 * agreement * neighbor_factor, 0.0, 1.0);
    }
}

// ========================================================================= //
// 8c. Normal Voting Tensor / BEO normal denoising                            //
// ========================================================================= //
//
// This is a per-scan front-end denoiser. It runs after initial normals are
// oriented toward the current sensor origin, and before QEM planes are
// accumulated into voxels. That ordering matters: once a bad normal has been
// fused into A,b,c, the QEM has no memory of the original observation to fix it.
//
// For each point i, build a vertex Normal Voting Tensor
//      T_i = (1/sum w_ij) sum_j w_ij n_j n_j^T
// using a hard normal-similarity gate. Eigenvalues are then binarized in the
// BEO style: flat=[1,0,0], edge=[1,1,0], corner=[1,1,1]. The filtered normal is
//      n_i' = normalize(d n_i + T_BEO n_i)
// which preserves sharp features while suppressing weak noisy eigendirections.
//
// The implementation uses kNN plus a radius safety cap. Pure kNN can span too
// large a physical distance on sparse far-range LiDAR, so nvt_max_radius limits
// smoothing support. Scan-line compatibility, when available, prevents voting
// across depth jumps / foreground-background boundaries.

static void denoise_normals_nvt(const std::vector<Vec3>& points,
                                const Vec3& sensor_origin,
                                const Settings& s,
                                std::vector<Vec3>& normals,
                                std::vector<double>& normal_confidence,
                                const std::vector<ScanlinePointInfo>* scanline_info = nullptr,
                                const std::vector<char>* eligible_mask = nullptr) {
    (void)sensor_origin;  // sign convention is preserved by alignment to the pre-NVT normal
    int N = (int)points.size();
    if (scanline_info && (int)scanline_info->size() != N) scanline_info = nullptr;
    if (eligible_mask && (int)eligible_mask->size() != N) eligible_mask = nullptr;
    if (!s.enable_nvt || s.nvt_iters <= 0 || N < 3) return;
    int k = std::min(std::max(3, s.nvt_k), N);
    double max_radius = s.nvt_max_radius > 0.0
        ? s.nvt_max_radius
        : std::max(5.0 * s.voxel_size, 1e-6);

    if ((int)normal_confidence.size() != N)
        normal_confidence.assign(N, 1.0);

    KDTree3D tree;
    tree.build(points);

    for (int iter = 0; iter < s.nvt_iters; iter++) {
        std::vector<Vec3> next_normals(N, Vec3::Zero());
        std::vector<double> next_conf(N, 0.0);

        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(dynamic, 256)
        #endif
        for (int i = 0; i < N; i++) {
            Vec3 ni = normals[i];
            double ni_len = ni.norm();
            if (ni_len < 1e-9) { next_normals[i] = ni; continue; }
            ni /= ni_len;
            if (eligible_mask && i < (int)eligible_mask->size() && !(*eligible_mask)[i]) {
                next_normals[i] = ni;
                next_conf[i] = normal_confidence[i];
                continue;
            }

            std::vector<double> dists;
            std::vector<int> idx;
            tree.knn(points[i], k, dists, idx);
            if ((int)idx.size() < 3) {
                next_normals[i] = ni;
                next_conf[i] = normal_confidence[i];
                continue;
            }

            Mat3 T = Mat3::Zero();
            double wsum = 0.0;
            for (int jj = 0; jj < (int)idx.size(); jj++) {
                int j = idx[jj];
                if (j == i) continue;
                if (jj < (int)dists.size() && dists[jj] > max_radius) continue;
                if (scanline_info && !scanline_compatible(i, j, *scanline_info, s))
                    continue;
                Vec3 nj = normals[j];
                double nj_len = nj.norm();
                if (nj_len < 1e-9) continue;
                nj /= nj_len;

                // Sign-align neighbor to central normal. The outer product is
                // sign-invariant, but the hard gate and final update are not.
                double dot = ni.dot(nj);
                if (dot < 0.0) { nj = -nj; dot = -dot; }
                if (dot < s.nvt_rho_cos) continue;

                double wc = std::clamp(normal_confidence[j],
                                       s.nvt_min_conf, 1.0);
                // Mild spatial falloff using the farthest returned neighbor as
                // local support scale. This is intentionally weak; the normal
                // gate is the main feature-preserving mechanism.
                double spatial = 1.0;
                if (jj < (int)dists.size() && max_radius > 1e-12) {
                    double u = std::min(dists[jj] / max_radius, 1.0);
                    spatial = std::exp(-2.0 * u * u);
                }
                double w = wc * spatial;
                T.noalias() += w * (nj * nj.transpose());
                wsum += w;
            }

            if (wsum < 1e-12) {
                next_normals[i] = ni;
                next_conf[i] = normal_confidence[i];
                continue;
            }
            T /= wsum;

            Eigen::SelfAdjointEigenSolver<Mat3> eig(T);
            // Eigen gives ascending eigenvalues. Convert to descending.
            double l0 = eig.eigenvalues()[2];
            double l1 = eig.eigenvalues()[1];
            double l2 = eig.eigenvalues()[0];
            Mat3 U;
            U.col(0) = eig.eigenvectors().col(2);
            U.col(1) = eig.eigenvectors().col(1);
            U.col(2) = eig.eigenvectors().col(0);

            Vec3 lambda_bin(1.0, 0.0, 0.0);  // flat by default
            double conf = 0.0;
            if (l2 >= s.nvt_tau) {
                // Corner / isotropic tensor: do not force smoothing to one
                // direction. Confidence is deliberately moderate.
                lambda_bin = Vec3(1.0, 1.0, 1.0);
                conf = 0.50;
            } else if (l1 >= s.nvt_tau) {
                // Edge: preserve the 2D normal subspace.
                lambda_bin = Vec3(1.0, 1.0, 0.0);
                conf = 0.50 + 0.50 * std::clamp((l1 - l2) / (l0 + 1e-12), 0.0, 1.0);
            } else {
                // Flat patch: one dominant normal direction.
                lambda_bin = Vec3(1.0, 0.0, 0.0);
                conf = std::clamp((l0 - l1) / (l0 + 1e-12), 0.0, 1.0);
            }

            Mat3 T_beo = U * lambda_bin.asDiagonal() * U.transpose();
            Vec3 nf = s.nvt_damping * ni + T_beo * ni;
            double nf_len = nf.norm();
            if (nf_len < 1e-9) nf = ni;
            else nf /= nf_len;

            // Keep the sign convention already established before NVT.
            // The QEM plane itself is sign-invariant, but mean normals and
            // visualization are not, so align to the pre-NVT normal ni.
            if (nf.dot(ni) < 0.0) nf = -nf;

            next_normals[i] = nf;
            next_conf[i] = std::clamp(conf, s.nvt_min_conf, 1.0);
        }

        normals.swap(next_normals);
        normal_confidence.swap(next_conf);
    }
}

// ========================================================================= //
// 9. VoxelQEMMap                                                             //
// ========================================================================= //

class VoxelQEMMap {
public:
    Settings s;
    std::unordered_map<VoxKey, VoxelCell, VoxHash> cells;
    // Pending weak/hypothesis voxels. These are not used for ordinary vertex
    // extraction or meshing until promoted into `cells` by consistency tests.
    std::unordered_map<VoxKey, VoxelCell, VoxHash> seed_cells;
    // Dirty L0 seeds touched this scan, plus neighbors of newly promoted seeds.
    // Promotion uses this active set instead of rescanning all pending seeds.
    std::unordered_set<VoxKey, VoxHash> dirty_seed_keys;

    // L1 seed-parent validator map. Parent cells are 2x coarser than the
    // ordinary voxel grid. They accumulate the same weak endpoint QEM as
    // seed_cells, but are used only to validate and promote observed L0 child
    // seed voxels. They are not emitted as geometry.
    std::unordered_map<VoxKey, VoxelCell, VoxHash> seed_l1_cells;
    std::unordered_map<VoxKey, std::unordered_set<int>, VoxHash> seed_l1_scan_ids;
    std::unordered_set<VoxKey, VoxHash> dirty_seed_l1_keys;
    std::unordered_set<VoxKey, VoxHash> scaffold_inherited_keys;

    // Dirty voxel set for persistent component updates/growth. Mutable because
    // mesh builders are const snapshot/export functions but component ownership
    // is cache state, not map evidence.
    mutable std::unordered_set<VoxKey, VoxHash> dirty_component_voxel_keys;

    std::string mesh_run_id;
    int scan_count = 0;

    // Running totals reset across scans (kept for global summary at the end).
    long n_grazing_total = 0;
    long n_misses_total  = 0;
    long n_points_total  = 0;
    long n_hits_total    = 0;
    long n_seed_hits_total = 0;
    long n_seed_promoted_total = 0;
    long n_seed_l1_parent_supported_total = 0;

    // Viz state ----------------------------------------------------------- //
    int dump_pass_count = 0;
    struct DumpEntry {
        int scan_idx;
        std::string ply;
        int n_vertices;
        int n_new_this_scan;
        double dump_time_sec;
    };
    std::vector<DumpEntry> dump_manifest;

    bool dump_active() const { return !s.dump_dir.empty(); }

    // ROI test: when no radius is set we accept everything (the dump_max_verts
    // cap is the only bound). With a radius set, only vertices inside the
    // sphere centred at dump_roi_center are dumped -- useful for zooming
    // into one part of a scene without serializing the whole map per scan.
    bool in_roi(const Vec3& p) const {
        if (s.dump_dir.empty()) return false;
        if (s.dump_roi_radius <= 0.0) return true;
        return (p - s.dump_roi_center).squaredNorm()
                <= s.dump_roi_radius * s.dump_roi_radius;
    }

    explicit VoxelQEMMap(const Settings& settings) : s(settings), mesh_run_id(make_run_id()) {}

    // ---- index helpers ----

    VoxKey voxel_index(const Vec3& p) const {
        return VoxKey{
            (int32_t)std::floor(p[0] / s.voxel_size),
            (int32_t)std::floor(p[1] / s.voxel_size),
            (int32_t)std::floor(p[2] / s.voxel_size),
        };
    }
    Vec3 voxel_center(const VoxKey& k) const {
        return Vec3((k.i + 0.5) * s.voxel_size,
                    (k.j + 0.5) * s.voxel_size,
                    (k.k + 0.5) * s.voxel_size);
    }
    void voxel_bounds(const VoxKey& k, Vec3& lo, Vec3& hi) const {
        lo = Vec3(k.i * s.voxel_size, k.j * s.voxel_size, k.k * s.voxel_size);
        hi = lo + Vec3::Constant(s.voxel_size);
    }

    static int32_t floor_div_i32(int32_t a, int32_t b) {
        int32_t q = a / b;
        int32_t r = a % b;
        if (r != 0 && ((r > 0) != (b > 0))) --q;
        return q;
    }

    VoxKey seed_l1_key_from_child(const VoxKey& child) const {
        int f = std::max(1, s.seed_l1_factor);
        return VoxKey{floor_div_i32(child.i, f), floor_div_i32(child.j, f), floor_div_i32(child.k, f)};
    }

    Vec3 voxel_center_level(const VoxKey& k, int factor) const {
        double vs = s.voxel_size * std::max(1, factor);
        return Vec3((k.i + 0.5) * vs, (k.j + 0.5) * vs, (k.k + 0.5) * vs);
    }

    void voxel_bounds_level(const VoxKey& k, int factor, Vec3& lo, Vec3& hi) const {
        double vs = s.voxel_size * std::max(1, factor);
        lo = Vec3(k.i * vs, k.j * vs, k.k * vs);
        hi = lo + Vec3::Constant(vs);
    }

    void mark_component_dirty(const VoxKey& k) const {
        // Runtime guard: dirty tracking is only useful when one of the
        // component-maintenance stages is enabled. Most importantly, this
        // prevents ordinary ray-carved free-space cells from exploding into
        // huge dirty neighborhoods when component persistence is not active.
        const bool needs_dirty =
            s.persistent_incremental_mesh ||
            s.enable_component_growth ||
            s.component_growth_persistent_state ||
            s.component_growth_dirty_only ||
            s.enable_component_fis_delete ||
            s.enable_component_radius_shrink ||
            s.enable_component_adaptive_simplification;
        if (!needs_dirty) return;

        int rad = std::max(0, s.component_growth_dirty_radius_voxels);
        for (int dx = -rad; dx <= rad; ++dx)
        for (int dy = -rad; dy <= rad; ++dy)
        for (int dz = -rad; dz <= rad; ++dz) {
            dirty_component_voxel_keys.insert(VoxKey{
                (int32_t)(k.i + dx),
                (int32_t)(k.j + dy),
                (int32_t)(k.k + dz)});
        }
    }

    bool component_key_is_dirty(const VoxKey& k) const {
        return dirty_component_voxel_keys.find(k) != dirty_component_voxel_keys.end();
    }

    // ---- Amanatides & Woo 3D DDA ----
    //
    // Yields the sequence of voxel indices the segment origin->end passes
    // through (inclusive of both endpoint voxels). Bounded by max_ray_voxels
    // so a runaway ray never blows the stack.
    void traverse_ray(const Vec3& origin, const Vec3& end,
                      std::vector<VoxKey>& out) const {
        out.clear();
        Vec3 d = end - origin;
        double len = d.norm();
        if (len < 1e-9) {
            out.push_back(voxel_index(origin));
            return;
        }
        VoxKey ijk = voxel_index(origin);
        VoxKey end_ijk = voxel_index(end);
        int step[3];
        double t_max[3]   = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
        double t_delta[3] = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
        const double vs = s.voxel_size;
        int32_t ijk_arr[3] = {ijk.i, ijk.j, ijk.k};
        for (int axis = 0; axis < 3; axis++) {
            if      (d[axis] > 0)  step[axis] =  1;
            else if (d[axis] < 0)  step[axis] = -1;
            else                   { step[axis] = 0; continue; }
            double boundary = (ijk_arr[axis] + (step[axis] > 0 ? 1 : 0)) * vs;
            t_max[axis]   = (boundary - origin[axis]) / d[axis];
            t_delta[axis] = vs / std::abs(d[axis]);
        }
        for (int it = 0; it < s.max_ray_voxels; it++) {
            out.push_back(VoxKey{ijk_arr[0], ijk_arr[1], ijk_arr[2]});
            if (ijk_arr[0] == end_ijk.i && ijk_arr[1] == end_ijk.j && ijk_arr[2] == end_ijk.k)
                return;
            int axis;
            if (t_max[0] <= t_max[1] && t_max[0] <= t_max[2]) axis = 0;
            else if (t_max[1] <= t_max[2])                    axis = 1;
            else                                              axis = 2;
            ijk_arr[axis] += step[axis];
            t_max[axis]   += t_delta[axis];
            // t > 1 means we passed the end point. Push the just-stepped voxel
            // and stop -- this handles the case where the end-voxel test above
            // missed by floating point (shouldn't, but defensive).
            if (t_max[axis] > 1.0 + 1e-9) {
                out.push_back(VoxKey{ijk_arr[0], ijk_arr[1], ijk_arr[2]});
                return;
            }
        }
    }

    // EOGM reliability mapping. Weight is the same incidence/normal-confidence
    // value used by QEM, but converted to a bounded mass assignment.
    double eogm_hit_reliability(double w, bool seed, bool scan_boundary, bool ray_normal) const {
        if (!s.enable_eogm) return 0.0;
        double scale = seed ? s.eogm_seed_hit_scale : s.eogm_hit_scale;
        double cap = seed ? s.eogm_max_seed_hit_reliability : s.eogm_max_hit_reliability;
        double r = 1.0 - std::exp(-scale * std::max(0.0, w));
        r = std::min(r, cap);
        if (scan_boundary) r *= s.eogm_boundary_discount;
        if (ray_normal) r *= s.eogm_ray_normal_discount;
        return std::clamp(r, 0.0, 1.0);
    }

    double eogm_miss_reliability(double w, bool seed_ray=false) const {
        if (!s.enable_eogm) return 0.0;
        double scale = seed_ray ? s.eogm_seed_miss_scale : s.eogm_miss_scale;
        double r = 1.0 - std::exp(-scale * std::max(0.0, w));
        return std::clamp(std::min(r, s.eogm_max_miss_reliability), 0.0, 1.0);
    }


    double probabilistic_plane_sigma(const VoxelCell& c) const {
        if (!s.enable_probabilistic_planes) return s.voxel_size;
        FittedProbPlane fp = c.prob_plane.fit(s);
        if (!fp.valid) return s.voxel_size;
        return std::clamp(fp.sigma_plane, s.prob_min_sigma, s.prob_max_sigma);
    }

    double probabilistic_single_cell_distance_limit(const VoxelCell& c, double base_limit) const {
        if (!s.enable_probabilistic_planes || base_limit <= 0.0) return base_limit;
        const double sigma = probabilistic_plane_sigma(c);
        const double prob_limit = s.prob_gate_sigma * sigma;
        return std::clamp(prob_limit,
                          base_limit * s.prob_gate_min_factor,
                          base_limit * s.prob_gate_max_factor);
    }

    double probabilistic_plane_distance_limit(const VoxelCell& a, const VoxelCell& b, double base_limit) const {
        if (!s.enable_probabilistic_planes || base_limit <= 0.0) return base_limit;
        const double sigma = std::sqrt(probabilistic_plane_sigma(a) * probabilistic_plane_sigma(a)
                                     + probabilistic_plane_sigma(b) * probabilistic_plane_sigma(b));
        const double prob_limit = s.prob_gate_sigma * sigma;
        return std::clamp(prob_limit,
                          base_limit * s.prob_gate_min_factor,
                          base_limit * s.prob_gate_max_factor);
    }

    double probabilistic_cluster_residual_limit(const std::vector<VoxKey>& keys, double base_limit) const {
        if (!s.enable_probabilistic_planes || base_limit <= 0.0 || keys.empty()) return base_limit;
        double sigma2 = 0.0;
        int n = 0;
        for (const VoxKey& k : keys) {
            auto it = seed_cells.find(k);
            if (it != seed_cells.end()) { const double sig = probabilistic_plane_sigma(it->second); sigma2 += sig * sig; n++; }
            else {
                auto mit = cells.find(k);
                if (mit != cells.end()) { const double sig = probabilistic_plane_sigma(mit->second); sigma2 += sig * sig; n++; }
            }
        }
        if (n == 0) return base_limit;
        const double sigma = std::sqrt(sigma2 / (double)n);
        const double prob_limit = s.prob_gate_sigma * sigma;
        return std::clamp(prob_limit,
                          base_limit * s.prob_gate_min_factor,
                          base_limit * s.prob_gate_max_factor);
    }

    bool seed_evidence_uses_eogm() const {
        std::string m = s.seed_evidence_mode;
        for (char& c : m) c = (char)std::tolower((unsigned char)c);
        return s.enable_eogm && (m == "eogm" || m == "ds" || m == "dsm" || m == "dsmt");
    }

    bool old_seed_occupancy_gate(const VoxKey& k, double min_occupancy) const {
        auto main_it = cells.find(k);
        if (main_it != cells.end()) {
            if (main_it->second.occupancy_score() < min_occupancy)
                return false;
        }
        return true;
    }

    static VoxelCell combined_eogm_only(const VoxelCell* a, const VoxelCell& b) {
        VoxelCell out;
        if (a) {
            out.eogm_surface = a->eogm_surface;
            out.eogm_free = a->eogm_free;
            out.eogm_unknown = a->eogm_unknown;
            out.eogm_conflict = a->eogm_conflict;
        }
        out.combine_eogm_mass(b.eogm_surface, b.eogm_free, b.eogm_unknown, b.eogm_conflict);
        return out;
    }

    bool eogm_seed_gate(const VoxKey& k, const VoxelCell& scell, bool cluster_gate=false) const {
        if (!s.enable_eogm) return true;
        auto mit = cells.find(k);
        const VoxelCell* main = (mit != cells.end()) ? &mit->second : nullptr;
        VoxelCell ev = combined_eogm_only(main, scell);
        double min_pl = cluster_gate ? s.eogm_cluster_min_plaus_surface : s.eogm_seed_min_plaus_surface;
        double max_free = cluster_gate ? s.eogm_cluster_max_bel_free : s.eogm_seed_max_bel_free;
        double max_conf = cluster_gate ? s.eogm_cluster_max_conflict : s.eogm_seed_max_conflict;
        if (ev.eogm_plaus_surface() < min_pl) return false;
        if (ev.eogm_bel_free() > max_free) return false;
        if (ev.eogm_conflict_mass() > max_conf) return false;
        return true;
    }

    bool seed_maturity_evidence_gate(const VoxKey& k, const VoxelCell& scell,
                                     bool cluster_gate=false,
                                     double old_min_occupancy=std::numeric_limits<double>::quiet_NaN()) const {
        // Mode switch: use exactly one evidence/maturity model.
        // Old mode: original hit/miss occupancy-style gating.
        // EOGM mode: Dempster-Shafer plausibility/free/conflict gating.
        // Do not stack them, otherwise EOGM becomes a strict subset filter.
        if (seed_evidence_uses_eogm())
            return eogm_seed_gate(k, scell, cluster_gate);
        double occ = std::isfinite(old_min_occupancy) ? old_min_occupancy : s.seed_promote_min_occupancy;
        return old_seed_occupancy_gate(k, occ);
    }

    bool eogm_cluster_gate_keys(const std::vector<VoxKey>& cluster_keys, bool cluster_gate=true) const {
        if (!s.enable_eogm) return true;
        VoxelCell ev;
        for (const VoxKey& k : cluster_keys) {
            auto mit = cells.find(k);
            if (mit != cells.end())
                ev.combine_eogm_mass(mit->second.eogm_surface, mit->second.eogm_free,
                                     mit->second.eogm_unknown, mit->second.eogm_conflict);
            auto sit = seed_cells.find(k);
            if (sit != seed_cells.end())
                ev.combine_eogm_mass(sit->second.eogm_surface, sit->second.eogm_free,
                                     sit->second.eogm_unknown, sit->second.eogm_conflict);
        }
        double min_pl = cluster_gate ? s.eogm_cluster_min_plaus_surface : s.eogm_seed_min_plaus_surface;
        double max_free = cluster_gate ? s.eogm_cluster_max_bel_free : s.eogm_seed_max_bel_free;
        double max_conf = cluster_gate ? s.eogm_cluster_max_conflict : s.eogm_seed_max_conflict;
        if (ev.eogm_plaus_surface() < min_pl) return false;
        if (ev.eogm_bel_free() > max_free) return false;
        if (ev.eogm_conflict_mass() > max_conf) return false;
        return true;
    }

    // ---- Seed/hypothesis voxel promotion -----------------------------------

    bool seed_cell_basic_ok(const VoxKey& k, const VoxelCell& scell,
                            Vec3* out_x = nullptr, double* out_sqrt_res = nullptr,
                            Vec3* out_n = nullptr) const {
        if (!s.enable_seed_voxels) return false;
        if (scell.hit_count < 1 || scell.weight_sum < 1e-12) return false;
        if (scell.normal_consistency() < s.seed_promote_min_consistency) return false;
        if (scell.ray_normal_fraction() > s.seed_max_ray_normal_fraction) return false;
        if (!seed_maturity_evidence_gate(k, scell, false, s.seed_promote_min_occupancy))
            return false;
        Vec3 anchor = voxel_center(k);
        Vec3 lo, hi; voxel_bounds(k, lo, hi);
        Vec3 x = scell.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
        double res = scell.residual_per_observation(x);
        if (!std::isfinite(res)) return false;
        double sr = std::sqrt(std::max(0.0, res));
        if (sr > s.seed_promote_max_sqrt_residual) return false;
        Vec3 n = scell.mean_normal();
        if (!normalized_or_zero(n)) return false;
        if (out_x) *out_x = x;
        if (out_sqrt_res) *out_sqrt_res = sr;
        if (out_n) *out_n = n;
        return true;
    }

    bool seed_neighbor_supported(const VoxKey& k, const VoxelCell& scell) const {
        Vec3 sx, sn;
        double sr = 0.0;
        if (!seed_cell_basic_ok(k, scell, &sx, &sr, &sn)) return false;
        if (scell.hit_count < s.seed_promote_neighbor_min_hits) return false;

        int good = 0;
        double max_plane_dist = s.seed_promote_neighbor_plane_dist_factor * s.voxel_size;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    VoxKey nk{(int32_t)(k.i + dx), (int32_t)(k.j + dy), (int32_t)(k.k + dz)};
                    auto it = cells.find(nk);
                    if (it == cells.end()) continue;
                    const VoxelCell& nc = it->second;
                    if (nc.label(s) != VoxelCell::Label::SURFACE) continue;
                    if (nc.hit_count < s.min_hit_count_vertex) continue;
                    Vec3 nn = nc.mean_normal();
                    if (!normalized_or_zero(nn)) continue;
                    if (std::abs(sn.dot(nn)) < s.seed_promote_neighbor_normal_dot) continue;
                    Vec3 anchor = voxel_center(nk);
                    Vec3 lo, hi; voxel_bounds(nk, lo, hi);
                    Vec3 nx = nc.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
                    double pd1 = std::abs((sx - nx).dot(nn));
                    double pd2 = std::abs((nx - sx).dot(sn));
                    double prob_plane_dist = probabilistic_plane_distance_limit(scell, nc, max_plane_dist);
                    if (pd1 > prob_plane_dist || pd2 > prob_plane_dist) continue;
                    good++;
                    if (good >= s.seed_promote_min_neighbors) return true;
                }
            }
        }
        return false;
    }

    // Relaxed gate for seed-to-seed cluster members. Unlike the ordinary
    // self/neighbor path, this does not require the seed to stand alone; the
    // cluster's merged QEM validates the group. Precision-critical gates remain.
    bool seed_geom_ok_relaxed(const VoxKey& k, const VoxelCell& scell,
                              Vec3* out_x = nullptr,
                              Vec3* out_n = nullptr) const {
        if (!s.enable_seed_voxels) return false;
        if (scell.hit_count < 1 || scell.weight_sum < 1e-12) return false;
        if (scell.normal_consistency() < s.seed_promote_min_consistency) return false;
        if (scell.ray_normal_fraction() > s.seed_max_ray_normal_fraction) return false;
        if (!seed_maturity_evidence_gate(k, scell, true, s.seed_promote_min_occupancy))
            return false;
        Vec3 anchor = voxel_center(k);
        Vec3 lo, hi; voxel_bounds(k, lo, hi);
        Vec3 x = scell.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
        double res = scell.residual_per_observation(x);
        if (!std::isfinite(res)) return false;
        double sr = std::sqrt(std::max(0.0, res));
        if (sr > s.seed_promote_max_sqrt_residual) return false;
        Vec3 n = scell.mean_normal();
        if (!normalized_or_zero(n)) return false;
        if (out_x) *out_x = x;
        if (out_n) *out_n = n;
        return true;
    }

    static double seed_anchor_score(const VoxelCell& c) {
        return c.hit_weight * (double)std::max(1, c.hit_count) * c.normal_consistency();
    }

    bool seed_cluster_merged_ok(const std::vector<VoxKey>& cluster_keys,
                                VoxKey& out_anchor_key) const {
        if ((int)cluster_keys.size() < s.seed_promote_cluster_min_size) return false;
        Mat3 A_sum = Mat3::Zero();
        Vec3 b_sum = Vec3::Zero();
        double c_sum = 0.0;
        double wsum = 0.0;
        double hit_w_total = 0.0;
        double miss_w_total = 0.0;
        int evidence_total = 0;
        Vec3 anchor_pos = Vec3::Zero();
        double best_score = -1.0;
        VoxKey best_key = cluster_keys[0];

        for (const VoxKey& k : cluster_keys) {
            auto it = seed_cells.find(k);
            if (it == seed_cells.end()) return false;
            const VoxelCell& sc = it->second;
            A_sum.noalias() += sc.A;
            b_sum.noalias() += sc.b;
            c_sum += sc.c;
            wsum += sc.weight_sum;
            hit_w_total += sc.hit_weight;
            evidence_total += sc.hit_count + sc.miss_count;
            anchor_pos += voxel_center(k);

            auto mit = cells.find(k);
            if (mit != cells.end()) {
                hit_w_total += mit->second.hit_weight;
                miss_w_total += mit->second.miss_weight;
                evidence_total += mit->second.hit_count + mit->second.miss_count;
            }

            double score = seed_anchor_score(sc);
            if (score > best_score) {
                best_score = score;
                best_key = k;
            }
        }
        if (wsum < 1e-12) return false;
        if (seed_evidence_uses_eogm()) {
            if (!eogm_cluster_gate_keys(cluster_keys, true)) return false;
        } else {
            if (evidence_total < s.min_evidence) return false;
            double total_w = hit_w_total + miss_w_total;
            if (total_w < 1e-12) return false;
            double future_score = (hit_w_total - miss_w_total) / total_w;
            if (future_score < s.thresh_surface) return false;
        }

        anchor_pos /= (double)cluster_keys.size();
        Mat3 Areg = A_sum + s.lambda_p * Mat3::Identity();
        Vec3 x = Areg.ldlt().solve(b_sum + s.lambda_p * anchor_pos);
        double f = x.transpose() * A_sum * x;
        f += c_sum;
        f -= 2.0 * b_sum.dot(x);
        double merged_sqrt_res = std::sqrt(std::max(0.0, f / wsum));
        if (merged_sqrt_res > probabilistic_cluster_residual_limit(cluster_keys, s.seed_promote_cluster_merged_max_sqrt_residual))
            return false;

        out_anchor_key = best_key;
        return true;
    }

    bool seed_cluster_supported(const VoxKey& k, const VoxelCell& scell,
                                VoxKey& out_anchor_key) const {
        Vec3 sx, sn;
        if (!seed_geom_ok_relaxed(k, scell, &sx, &sn)) return false;

        std::vector<VoxKey> cluster;
        cluster.push_back(k);
        int rad = std::max(1, s.seed_promote_cluster_radius_voxels);
        double max_plane_dist = s.seed_promote_cluster_plane_dist_factor * s.voxel_size;
        for (int dx = -rad; dx <= rad; dx++) {
            for (int dy = -rad; dy <= rad; dy++) {
                for (int dz = -rad; dz <= rad; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    VoxKey nk{(int32_t)(k.i + dx), (int32_t)(k.j + dy), (int32_t)(k.k + dz)};
                    auto it = seed_cells.find(nk);
                    if (it == seed_cells.end()) continue;
                    const VoxelCell& nc = it->second;
                    Vec3 nx, nn;
                    if (!seed_geom_ok_relaxed(nk, nc, &nx, &nn)) continue;
                    if (std::abs(sn.dot(nn)) < s.seed_promote_cluster_normal_dot) continue;
                    double pd1 = std::abs((sx - nx).dot(nn));
                    double pd2 = std::abs((nx - sx).dot(sn));
                    double prob_plane_dist = probabilistic_plane_distance_limit(scell, nc, max_plane_dist);
                    if (pd1 > prob_plane_dist || pd2 > prob_plane_dist) continue;
                    cluster.push_back(nk);
                }
            }
        }
        if ((int)cluster.size() < s.seed_promote_cluster_min_size) return false;
        return seed_cluster_merged_ok(cluster, out_anchor_key);
    }

    struct SeedPromoteDecision {
        bool ok = false;
        VoxKey anchor_key{};
        enum class Path { NONE, SELF, NEIGHBOR, CLUSTER } path = Path::NONE;
    };

    SeedPromoteDecision seed_cell_promotable(const VoxKey& k, const VoxelCell& scell) const {
        SeedPromoteDecision dec;
        dec.anchor_key = k;
        if (scell.hit_count >= s.seed_promote_min_hits && seed_cell_basic_ok(k, scell)) {
            dec.ok = true; dec.path = SeedPromoteDecision::Path::SELF; return dec;
        }
        if (seed_neighbor_supported(k, scell)) {
            dec.ok = true; dec.path = SeedPromoteDecision::Path::NEIGHBOR; return dec;
        }
        VoxKey anchor;
        if (seed_cluster_supported(k, scell, anchor)) {
            dec.ok = true; dec.anchor_key = anchor; dec.path = SeedPromoteDecision::Path::CLUSTER; return dec;
        }
        return dec;
    }

    void mark_seed_neighbors_dirty(const VoxKey& k) {
        int rad = std::max(1, s.seed_promote_cluster_radius_voxels);
        for (int dx = -rad; dx <= rad; dx++) {
            for (int dy = -rad; dy <= rad; dy++) {
                for (int dz = -rad; dz <= rad; dz++) {
                    VoxKey nk{(int32_t)(k.i + dx), (int32_t)(k.j + dy), (int32_t)(k.k + dz)};
                    if (seed_cells.find(nk) != seed_cells.end()) dirty_seed_keys.insert(nk);
                }
            }
        }
        if (s.enable_seed_l1_parents) dirty_seed_l1_keys.insert(seed_l1_key_from_child(k));
    }

    Vec3 solve_seed_l1_vertex(const VoxKey& pk, const VoxelCell& pc) const {
        Vec3 anchor = voxel_center_level(pk, s.seed_l1_factor);
        Vec3 lo, hi; voxel_bounds_level(pk, s.seed_l1_factor, lo, hi);
        return pc.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
    }

    bool seed_l1_parent_valid(const VoxKey& pk, Vec3* out_x=nullptr,
                              Vec3* out_n=nullptr, double* out_sqrt_res=nullptr) const {
        if (!s.enable_seed_l1_parents) return false;
        auto pit = seed_l1_cells.find(pk);
        if (pit == seed_l1_cells.end()) return false;
        const VoxelCell& parent = pit->second;
        if (parent.hit_count < 1 || parent.weight_sum < 1e-12) return false;
        if (parent.normal_consistency() < s.seed_l1_min_consistency) return false;
        if (parent.ray_normal_fraction() > s.seed_l1_max_ray_normal_fraction) return false;
        if (seed_evidence_uses_eogm()) {
            if (parent.eogm_plaus_surface() < s.eogm_cluster_min_plaus_surface) return false;
            if (parent.eogm_bel_free() > s.eogm_cluster_max_bel_free) return false;
            if (parent.eogm_conflict_mass() > s.eogm_cluster_max_conflict) return false;
        }
        auto sit = seed_l1_scan_ids.find(pk);
        int unique_scans = (sit == seed_l1_scan_ids.end()) ? 0 : (int)sit->second.size();
        if (unique_scans < s.seed_l1_min_unique_scans) return false;

        // Count direct L0 seed children in the 2x2x2 parent block. This keeps
        // parent validation tied to actual child voxels, not just one dense child.
        int child_count = 0;
        int f = std::max(1, s.seed_l1_factor);
        for (int dx = 0; dx < f; ++dx)
        for (int dy = 0; dy < f; ++dy)
        for (int dz = 0; dz < f; ++dz) {
            VoxKey ck{(int32_t)(pk.i * f + dx), (int32_t)(pk.j * f + dy), (int32_t)(pk.k * f + dz)};
            auto cit = seed_cells.find(ck);
            if (cit != seed_cells.end() && cit->second.hit_count > 0 && cit->second.weight_sum > 1e-12)
                child_count++;
        }
        if (child_count < s.seed_l1_min_children) return false;

        Vec3 x = solve_seed_l1_vertex(pk, parent);
        double res = parent.residual_per_observation(x);
        if (!std::isfinite(res)) return false;
        double sr = std::sqrt(std::max(0.0, res));
        if (sr > s.seed_l1_max_sqrt_residual) return false;
        Vec3 n = parent.mean_normal();
        if (!normalized_or_zero(n)) return false;
        if (out_x) *out_x = x;
        if (out_n) *out_n = n;
        if (out_sqrt_res) *out_sqrt_res = sr;
        return true;
    }

    bool seed_l1_child_ok(const VoxKey& ck, const VoxelCell& child,
                          const Vec3& parent_x, const Vec3& parent_n) const {
        if (child.hit_count < 1 || child.weight_sum < 1e-12) return false;
        if (child.ray_normal_fraction() > s.seed_l1_max_ray_normal_fraction) return false;
        if (!seed_maturity_evidence_gate(ck, child, true, s.seed_l1_child_min_occupancy)) return false;
        Vec3 cn = child.mean_normal();
        if (!normalized_or_zero(cn)) return false;
        if (std::abs(cn.dot(parent_n)) < s.seed_l1_child_normal_dot) return false;
        Vec3 anchor = voxel_center(ck);
        Vec3 lo, hi; voxel_bounds(ck, lo, hi);
        Vec3 cx = child.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
        double max_dist = s.seed_l1_child_plane_dist > 0.0 ? s.seed_l1_child_plane_dist : s.voxel_size;
        double pd = std::abs((cx - parent_x).dot(parent_n));
        max_dist = probabilistic_single_cell_distance_limit(child, max_dist);
        if (pd > max_dist) return false;
        return true;
    }

    long promote_seed_l1_children() {
        if (!s.enable_seed_voxels || !s.enable_seed_l1_parents || !s.seed_l1_promote_children)
            return 0;
        if (dirty_seed_l1_keys.empty()) return 0;

        std::vector<std::pair<VoxKey, VoxelCell>> to_promote;
        int f = std::max(1, s.seed_l1_factor);
        for (const VoxKey& pk : dirty_seed_l1_keys) {
            Vec3 px, pn;
            double psr = 0.0;
            if (!seed_l1_parent_valid(pk, &px, &pn, &psr)) continue;
            for (int dx = 0; dx < f; ++dx)
            for (int dy = 0; dy < f; ++dy)
            for (int dz = 0; dz < f; ++dz) {
                VoxKey ck{(int32_t)(pk.i * f + dx), (int32_t)(pk.j * f + dy), (int32_t)(pk.k * f + dz)};
                auto cit = seed_cells.find(ck);
                if (cit == seed_cells.end()) continue;
                if (!seed_l1_child_ok(ck, cit->second, px, pn)) continue;
                VoxelCell promoted = cit->second;
                promoted.promoted_seed_count += cit->second.hit_count;
                promoted.promoted_seed_weight += cit->second.hit_weight;
                promoted.parent_supported_seed_count += cit->second.hit_count;
                promoted.parent_supported_seed_weight += cit->second.hit_weight;
                to_promote.push_back({ck, promoted});
            }
        }
        dirty_seed_l1_keys.clear();

        long promoted_n = 0;
        for (auto& kv : to_promote) {
            const VoxKey& ck = kv.first;
            auto cit = seed_cells.find(ck);
            if (cit == seed_cells.end()) continue; // may have been promoted by another parent
            cells[ck].merge_from(kv.second);
            mark_component_dirty(ck);
            seed_cells.erase(cit);
            mark_seed_neighbors_dirty(ck);
            promoted_n++;
        }
        if (promoted_n > 0) {
            std::printf("    [seed_l1] parent_supported=%ld pending_l0=%zu parents=%zu\n",
                        promoted_n, seed_cells.size(), seed_l1_cells.size());
        }
        return promoted_n;
    }

    long promote_seed_voxels() {
        if (!s.enable_seed_voxels || dirty_seed_keys.empty()) {
            dirty_seed_keys.clear();
            return 0;
        }
        std::unordered_set<VoxKey, VoxHash> to_promote;
        long n_self = 0, n_neigh = 0, n_cluster = 0;
        for (const VoxKey& k : dirty_seed_keys) {
            auto it = seed_cells.find(k);
            if (it == seed_cells.end()) continue;
            SeedPromoteDecision dec = seed_cell_promotable(k, it->second);
            if (!dec.ok) continue;
            if (to_promote.insert(dec.anchor_key).second) {
                if (dec.path == SeedPromoteDecision::Path::SELF) n_self++;
                else if (dec.path == SeedPromoteDecision::Path::NEIGHBOR) n_neigh++;
                else if (dec.path == SeedPromoteDecision::Path::CLUSTER) n_cluster++;
            }
        }
        dirty_seed_keys.clear();

        long promoted = 0;
        for (const VoxKey& k : to_promote) {
            auto it = seed_cells.find(k);
            if (it == seed_cells.end()) continue;
            VoxelCell promoted_cell = it->second;
            promoted_cell.promoted_seed_count += it->second.hit_count;
            promoted_cell.promoted_seed_weight += it->second.hit_weight;
            cells[k].merge_from(promoted_cell);
            mark_component_dirty(k);
            seed_cells.erase(it);
            mark_seed_neighbors_dirty(k);
            promoted++;
        }
        if (promoted > 0) {
            std::printf("    [seed] promoted=%ld (self=%ld neigh=%ld cluster=%ld) pending=%zu\n",
                        promoted, n_self, n_neigh, n_cluster, seed_cells.size());
        }
        return promoted;
    }

    long finalize_seed_promotions(int max_rounds = 5) {
        long total = 0;
        for (int r = 0; r < max_rounds; ++r) {
            long a = promote_seed_l1_children();
            long b = promote_seed_voxels();
            total += a + b;
            if (a == 0 && b == 0) break;
        }
        rebuild_hierarchical_scaffold_inheritance(scan_count);
        return total;
    }

    // ---- Hierarchical scaffold inheritance ---------------------------------
    bool hierarchical_scaffold_enabled() const {
        return s.enable_hierarchical_scaffold || s.enable_planar_scaffold_inherit;
    }

    bool hierarchical_scaffold_fill_enabled() const {
        return (s.enable_hierarchical_scaffold || s.enable_planar_scaffold) &&
               s.enable_hierarchical_scaffold_fill;
    }

    int scaffold_factor_for_level(int level) const {
        level = std::max(0, level);
        int base = std::max(2, s.scaffold_base_factor);
        int f = 1;
        for (int i = 0; i < level; ++i) {
            if (f > 1024 / base) return 1024;
            f *= base;
        }
        return std::max(1, f);
    }

    VoxKey scaffold_parent_key(const VoxKey& child, int level) const {
        int f = scaffold_factor_for_level(level);
        return VoxKey{floor_div_i32(child.i, f), floor_div_i32(child.j, f), floor_div_i32(child.k, f)};
    }

    struct ScaffoldAggregate {
        Mat3 A = Mat3::Zero();
        Vec3 b = Vec3::Zero();
        double c = 0.0;
        double weight_sum = 0.0;
        Vec3 normal_sum = Vec3::Zero();
        double normal_weight = 0.0;
        int child_count = 0;
        double bel_free_max = 0.0;
        double conflict_max = 0.0;
    };

    int scaffold_min_children_for_level(int level) const {
        int f = scaffold_factor_for_level(level);
        int min_children = s.scaffold_min_parent_children
                         + std::max(0, level - 1) * s.scaffold_min_parent_children_per_level;
        if (s.scaffold_min_child_fraction > 0.0) {
            double frac = std::clamp(s.scaffold_min_child_fraction, 0.0, 1.0);
            int by_frac = (int)std::ceil(frac * (double)f * (double)f * (double)f);
            min_children = std::max(min_children, by_frac);
        }
        return std::max(1, min_children);
    }

    bool scaffold_source_cell_ok(const VoxelCell& c) const {
        if (c.label(s) != VoxelCell::Label::SURFACE) return false;
        if (c.confirmed_hit_count() < s.min_hit_count_vertex) return false;
        if (c.weight_sum < 1e-12) return false;
        if (c.normal_consistency() < s.scaffold_min_normal_consistency) return false;
        if (s.enable_eogm) {
            if (c.eogm_bel_free() > s.scaffold_max_bel_free) return false;
            if (c.eogm_conflict_mass() > s.scaffold_max_conflict) return false;
        }
        return true;
    }

    std::unordered_map<VoxKey, ScaffoldAggregate, VoxHash> build_scaffold_aggregates(int level) const {
        std::unordered_map<VoxKey, ScaffoldAggregate, VoxHash> agg;
        agg.reserve(cells.size() / 8 + 1);
        for (const auto& kv : cells) {
            const VoxKey& ck = kv.first;
            const VoxelCell& vc = kv.second;
            if (!scaffold_source_cell_ok(vc)) continue;
            ScaffoldAggregate& a = agg[scaffold_parent_key(ck, level)];
            a.A.noalias() += vc.A;
            a.b.noalias() += vc.b;
            a.c += vc.c;
            a.weight_sum += vc.weight_sum;
            a.normal_sum.noalias() += vc.normal_sum;
            a.normal_weight += vc.normal_weight;
            a.child_count++;
            a.bel_free_max = std::max(a.bel_free_max, vc.eogm_bel_free());
            a.conflict_max = std::max(a.conflict_max, vc.eogm_conflict_mass());
        }
        return agg;
    }

    bool scaffold_parent_plane(int level, const VoxKey& pk, const ScaffoldAggregate& a,
                               Vec3& out_n, double& out_d, Vec3& out_x) const {
        if (a.child_count < scaffold_min_children_for_level(level)) return false;
        if (a.weight_sum < 1e-12 || a.normal_weight < 1e-12) return false;
        Vec3 mean_n = a.normal_sum / a.normal_weight;
        double consistency = mean_n.norm();
        if (consistency < s.scaffold_min_normal_consistency) return false;
        if (s.enable_eogm) {
            if (a.bel_free_max > s.scaffold_max_bel_free) return false;
            if (a.conflict_max > s.scaffold_max_conflict) return false;
        }
        int factor = scaffold_factor_for_level(level);
        Vec3 anchor = voxel_center_level(pk, factor);
        Mat3 Areg = a.A + s.lambda_p * Mat3::Identity();
        Vec3 x = Areg.ldlt().solve(a.b + s.lambda_p * anchor);
        if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2])) return false;
        double f = x.transpose() * a.A * x;
        f += a.c;
        f -= 2.0 * a.b.dot(x);
        double sqrt_res = std::sqrt(std::max(0.0, f / a.weight_sum));
        double res_limit = s.scaffold_max_merged_sqrt_residual *
            std::pow(std::clamp(s.scaffold_residual_level_decay, 0.05, 1.0), std::max(0, level - 1));
        if (sqrt_res > res_limit) return false;
        Eigen::SelfAdjointEigenSolver<Mat3> eig(a.A);
        if (eig.info() != Eigen::Success) return false;
        double l_big = std::max(eig.eigenvalues()[2], 1e-12);
        double l_mid = std::max(eig.eigenvalues()[1], 0.0);
        double rank_limit = s.scaffold_max_rank_ratio *
            std::pow(std::clamp(s.scaffold_rank_level_decay, 0.05, 1.0), std::max(0, level - 1));
        if (l_mid / l_big > rank_limit) return false;
        Vec3 n = eig.eigenvectors().col(2);
        if (!normalized_or_zero(n)) return false;
        if (n.dot(mean_n) < 0.0) n = -n;
        out_n = n;
        out_d = -n.dot(x);
        out_x = x;
        return true;
    }

    bool scaffold_child_visibility_ok(const VoxKey& ck) const {
        auto it = cells.find(ck);
        if (it == cells.end()) return true;
        const VoxelCell& vc = it->second;
        if (vc.label(s) == VoxelCell::Label::FREE) return false;
        if (s.enable_eogm) {
            if (vc.eogm_bel_free() > s.scaffold_max_bel_free) return false;
            if (vc.eogm_conflict_mass() > s.scaffold_max_conflict) return false;
        }
        return true;
    }

    void clear_hierarchical_scaffold_inheritance(bool mark_dirty = true) {
        for (const VoxKey& k : scaffold_inherited_keys) {
            if (mark_dirty) mark_component_dirty(k);
            auto it = cells.find(k);
            if (it != cells.end()) it->second.clear_inherited();
        }
        scaffold_inherited_keys.clear();
    }

    long rebuild_hierarchical_scaffold_inheritance(int scan_idx) {
        if (!hierarchical_scaffold_enabled()) {
            // Scaffold turned off: virtual support vanishes everywhere, so dirty
            // those cells once to let the persistent mesher retire their faces.
            clear_hierarchical_scaffold_inheritance(/*mark_dirty=*/true);
            return 0;
        }
        // Refreshing virtual priors must NOT, by itself, dirty the whole scaffold
        // every scan -- that would defeat the active-parent gate below and churn
        // the incremental mesh. Clear silently, remember the previous keys, and
        // dirty selectively: active-parent children (in the build loop) and any
        // cell whose inherited support disappeared (handled after the rebuild).
        std::unordered_set<VoxKey, VoxHash> old_inherited_keys = scaffold_inherited_keys;
        clear_hierarchical_scaffold_inheritance(/*mark_dirty=*/false);
        long total_children = 0;
        int max_level = std::clamp(s.scaffold_max_level, 1, 6);
        double base_weight = std::max(1e-12, s.scaffold_inherit_weight);
        double level_decay = std::clamp(s.scaffold_inherit_level_decay, 0.01, 1.0);
        double decay_ref_scale = std::max(1e-12, s.scaffold_inherit_decay_weight_ref);
        for (int level = 1; level <= max_level; ++level) {
            int factor = scaffold_factor_for_level(level);
            auto agg = build_scaffold_aggregates(level);
            long level_children = 0;
            long level_parents = 0;
            double w = base_weight * std::pow(level_decay, level - 1);
            double decay_ref = decay_ref_scale * w;
            for (const auto& kv : agg) {
                const VoxKey& pk = kv.first;
                const ScaffoldAggregate& a = kv.second;
                Vec3 n, x;
                double d = 0.0;
                if (!scaffold_parent_plane(level, pk, a, n, d, x)) continue;
                level_parents++;

                // A parent is "active" this scan if any of its children received
                // a hit this scan. Only active parents propagate dirtiness to
                // their inherited children, so a stable scaffold region is not
                // re-meshed every export -- this keeps the persistent incremental
                // mesher bounded while still re-solving sparse cells whose
                // underlying surface actually moved.
                bool parent_active = false;
                for (int dx = 0; dx < factor && !parent_active; ++dx)
                for (int dy = 0; dy < factor && !parent_active; ++dy)
                for (int dz = 0; dz < factor && !parent_active; ++dz) {
                    VoxKey ck{(int32_t)(pk.i * factor + dx),
                              (int32_t)(pk.j * factor + dy),
                              (int32_t)(pk.k * factor + dz)};
                    auto cit = cells.find(ck);
                    if (cit != cells.end() && cit->second.hit_count > 0 &&
                        cit->second.last_hit_scan == scan_idx)
                        parent_active = true;
                }

                for (int dx = 0; dx < factor; ++dx)
                for (int dy = 0; dy < factor; ++dy)
                for (int dz = 0; dz < factor; ++dz) {
                    VoxKey ck{(int32_t)(pk.i * factor + dx),
                              (int32_t)(pk.j * factor + dy),
                              (int32_t)(pk.k * factor + dz)};
                    auto cit = cells.find(ck);
                    if (!s.scaffold_inherit_into_real_cells && cit != cells.end() && cit->second.weight_sum > 1e-12)
                        continue;
                    if (!scaffold_child_visibility_ok(ck)) continue;
                    Vec3 ctr = voxel_center(ck);
                    double reach_scale = 0.5 * s.voxel_size * (std::abs(n[0]) + std::abs(n[1]) + std::abs(n[2]));
                    if (std::abs(n.dot(ctr) + d) > reach_scale) continue;
                    Vec3 p = ctr - (n.dot(ctr) + d) * n;
                    VoxelCell& child = cells[ck];
                    child.add_inherited_plane(p, n, w, scan_idx, level, decay_ref);
                    scaffold_inherited_keys.insert(ck);
                    if (parent_active) mark_component_dirty(ck);
                    level_children++;
                }
            }
            total_children += level_children;
            if (level_children > 0) {
                std::printf("    [hier_scaffold] L%d factor=%d parents=%ld children_set=%ld weight=%.4f\n",
                            level, factor, level_parents, level_children, w);
            }
        }
        // Retire-side dirtying: any cell that had inherited support last scan but
        // does not anymore (parent dropped, plane no longer crosses it, etc.)
        // must be re-meshed so its now-unsupported faces can be dropped.
        for (const VoxKey& k : old_inherited_keys)
            if (scaffold_inherited_keys.find(k) == scaffold_inherited_keys.end())
                mark_component_dirty(k);
        return total_children;
    }

    // Aggregate the QEM/normal/evidence of all source-ok children inside one
    // coarse parent block. O(factor^3) instead of O(all cells); used by the
    // incremental inheritance rebuild so per-scan cost scales with the scan
    // footprint, not the whole map.
    ScaffoldAggregate build_scaffold_aggregate_for_parent(int level, const VoxKey& pk) const {
        ScaffoldAggregate a;
        const int factor = scaffold_factor_for_level(level);
        for (int dx = 0; dx < factor; ++dx)
        for (int dy = 0; dy < factor; ++dy)
        for (int dz = 0; dz < factor; ++dz) {
            VoxKey ck{(int32_t)(pk.i * factor + dx),
                      (int32_t)(pk.j * factor + dy),
                      (int32_t)(pk.k * factor + dz)};
            auto it = cells.find(ck);
            if (it == cells.end()) continue;
            const VoxelCell& vc = it->second;
            if (!scaffold_source_cell_ok(vc)) continue;
            a.A.noalias() += vc.A;
            a.b.noalias() += vc.b;
            a.c += vc.c;
            a.weight_sum += vc.weight_sum;
            a.normal_sum.noalias() += vc.normal_sum;
            a.normal_weight += vc.normal_weight;
            a.child_count++;
            a.bel_free_max = std::max(a.bel_free_max, vc.eogm_bel_free());
            a.conflict_max = std::max(a.conflict_max, vc.eogm_conflict_mass());
        }
        return a;
    }

    // Aggregates restricted to coarse parents that overlap the active region.
    // Each parent is still summed over its full child block (so a parent
    // straddling the region edge keeps a correct plane); only parents with no
    // child in the region are skipped. Keeps the scaffold fill pass O(region)
    // during a persistent incremental remesh instead of O(all cells).
    std::unordered_map<VoxKey, ScaffoldAggregate, VoxHash>
    build_scaffold_aggregates_for_region(int level) const {
        std::unordered_map<VoxKey, ScaffoldAggregate, VoxHash> agg;
        if (!active_region_) return agg;
        std::unordered_set<VoxKey, VoxHash> parents;
        parents.reserve(active_region_->size());
        for (const VoxKey& rk : *active_region_) parents.insert(scaffold_parent_key(rk, level));
        agg.reserve(parents.size() * 2 + 1);
        for (const VoxKey& pk : parents) {
            ScaffoldAggregate a = build_scaffold_aggregate_for_parent(level, pk);
            if (a.child_count > 0) agg.emplace(pk, std::move(a));
        }
        return agg;
    }

    // Incremental scaffold inheritance: only recompute coarse parents whose
    // children were hit this scan. Stable parents keep their previously inherited
    // children untouched (and un-dirtied), which is what makes per-scan cost
    // bounded by the scan footprint. Equivalent in steady state to the full
    // rebuild restricted to the changed region.
    long rebuild_hierarchical_scaffold_inheritance_incremental(int scan_idx,
                                                               const std::vector<VoxKey>& hit_keys) {
        if (!hierarchical_scaffold_enabled()) {
            clear_hierarchical_scaffold_inheritance(/*mark_dirty=*/true);
            return 0;
        }
        if (hit_keys.empty()) return 0;

        const int max_level = std::clamp(s.scaffold_max_level, 1, 6);
        const double base_weight = std::max(1e-12, s.scaffold_inherit_weight);
        const double level_decay = std::clamp(s.scaffold_inherit_level_decay, 0.01, 1.0);
        const double decay_ref_scale = std::max(1e-12, s.scaffold_inherit_decay_weight_ref);

        long total_children = 0;
        for (int level = 1; level <= max_level; ++level) {
            const int factor = scaffold_factor_for_level(level);
            const double w = base_weight * std::pow(level_decay, level - 1);
            const double decay_ref = decay_ref_scale * w;

            // Active parents = coarse blocks containing a cell hit this scan.
            std::unordered_set<VoxKey, VoxHash> active_parents;
            active_parents.reserve(hit_keys.size() * 2 + 1);
            for (const VoxKey& hk : hit_keys) active_parents.insert(scaffold_parent_key(hk, level));

            for (const VoxKey& pk : active_parents) {
                // 1. Drop this block's previous inheritance and dirty those cells
                //    (their virtual support is about to change or disappear).
                for (int dx = 0; dx < factor; ++dx)
                for (int dy = 0; dy < factor; ++dy)
                for (int dz = 0; dz < factor; ++dz) {
                    VoxKey ck{(int32_t)(pk.i * factor + dx),
                              (int32_t)(pk.j * factor + dy),
                              (int32_t)(pk.k * factor + dz)};
                    auto sit = scaffold_inherited_keys.find(ck);
                    if (sit == scaffold_inherited_keys.end()) continue;
                    auto cit = cells.find(ck);
                    if (cit != cells.end()) cit->second.clear_inherited();
                    scaffold_inherited_keys.erase(sit);
                    mark_component_dirty(ck);
                }

                // 2. Recompute the parent plane from the current block evidence.
                ScaffoldAggregate a = build_scaffold_aggregate_for_parent(level, pk);
                Vec3 n, x; double d = 0.0;
                if (!scaffold_parent_plane(level, pk, a, n, d, x)) continue; // support gone -> stays retired

                // 3. Inherit the plane into the empty/sparse children and dirty them.
                for (int dx = 0; dx < factor; ++dx)
                for (int dy = 0; dy < factor; ++dy)
                for (int dz = 0; dz < factor; ++dz) {
                    VoxKey ck{(int32_t)(pk.i * factor + dx),
                              (int32_t)(pk.j * factor + dy),
                              (int32_t)(pk.k * factor + dz)};
                    auto cit = cells.find(ck);
                    if (!s.scaffold_inherit_into_real_cells && cit != cells.end() && cit->second.weight_sum > 1e-12)
                        continue;
                    if (!scaffold_child_visibility_ok(ck)) continue;
                    Vec3 ctr = voxel_center(ck);
                    double reach_scale = 0.5 * s.voxel_size * (std::abs(n[0]) + std::abs(n[1]) + std::abs(n[2]));
                    if (std::abs(n.dot(ctr) + d) > reach_scale) continue;
                    Vec3 p = ctr - (n.dot(ctr) + d) * n;
                    VoxelCell& child = cells[ck];
                    child.add_inherited_plane(p, n, w, scan_idx, level, decay_ref);
                    scaffold_inherited_keys.insert(ck);
                    mark_component_dirty(ck);
                    total_children++;
                }
            }
        }
        if (total_children > 0)
            std::printf("    [hier_scaffold/incr] hit_cells=%zu children_set=%ld (region-local)\n",
                        hit_keys.size(), total_children);
        return total_children;
    }

    // ---- Per-scan processing -----------------------------------------------

    void process_scan(const PointCloud& pc, const Mat4& pose, int scan_idx) {
        double t0 = now_sec();

        // The mapping core follows the original/pasted code: PCD points and
        // normals are sensor-frame measurements, and the external SLAM pose
        // maps them into the world/map frame. Do not use PCD VIEWPOINT here.
        // LiDAR geometry is computed from the same sensor-frame points before
        // the SLAM pose is applied.
        Mat3 sensor_R = pose.block<3,3>(0,0);
        Vec3 sensor_t = pose.block<3,1>(0,3);

        // 1. Build a valid full-scan sensor-frame array first.
        // Critical: full_orig must contain the RAW PCD index (pre-NaN-skip),
        // not the compacted index, because organized PCD row/col indexing uses
        // the raw point order. For PandarQT64 unorganized scans, full_orig is
        // harmless; ring/pseudo-ring geometry comes from pts_sensor.
        int N_in = (int)pc.points.size();
        bool has_normals = !pc.normals.empty();
        bool have_orig_idx = (int)pc.original_indices.size() == N_in;
        std::vector<int>  full_orig; full_orig.reserve(N_in);
        std::vector<Vec3> full_pts_s; full_pts_s.reserve(N_in);
        std::vector<Vec3> full_pts_g; full_pts_g.reserve(N_in);
        std::vector<Vec3> full_nrm_s; if (has_normals) full_nrm_s.reserve(N_in);
        for (int i = 0; i < N_in; i++) {
            const Vec3& p = pc.points[i];
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            full_orig.push_back(have_orig_idx ? pc.original_indices[i] : i);
            Vec3 ps = p;
            Vec3 pg = sensor_R * ps + sensor_t;
            full_pts_s.push_back(ps);
            full_pts_g.push_back(pg);
            if (has_normals) full_nrm_s.push_back(pc.normals[i]);
        }
        int NF = (int)full_pts_s.size();

        // 1b. Build optional scan-line/ring/pseudo-ring structure in sensor
        // frame before subsampling. For PandarQT64 unorganized scans without
        // a ring field, --sensor pandar_qt64 clusters elevation into 64
        // pseudo-rings and uses azimuth-sorted neighbors.
        std::vector<ScanlinePointInfo> full_scan_info;
        build_scanline_info(pc, full_orig, full_pts_s, s, full_scan_info);
        bool have_full_scanline = (int)full_scan_info.size() == NF;
        if (s.pandar_qt64_mode && have_full_scanline && scan_idx < 3) {
            std::unordered_map<int,int> row_hist;
            int valid_rows_pts = 0;
            int max_row_pts = 0;
            for (const auto& si : full_scan_info) {
                if (!si.valid) continue;
                int c = ++row_hist[si.row];
                max_row_pts = std::max(max_row_pts, c);
                valid_rows_pts++;
            }
            if (valid_rows_pts > 0 && (double)max_row_pts / (double)valid_rows_pts > 0.90) {
                std::fprintf(stderr,
                    "  [WARN] PandarQT64 pseudo-ring assignment is highly skewed (%d/%d points in one row). "
                    "Check that PCD points are still in the LiDAR sensor frame before SLAM pose application.\n",
                    max_row_pts, valid_rows_pts);
            }
        }

        // 2. Estimate/collect normals for the full scan. If PCD normals are
        // missing, try scan-line/ring normals first and PCA only as fallback.
        std::vector<Vec3> full_nrm_g(NF, Vec3::Zero());
        std::vector<double> full_conf(NF, 1.0);
        std::vector<unsigned char> full_source(NF, 0); // 0=input, 1=scanline, 2=PCA, 3=ray
        long n_sl_normals = 0;
        if (!has_normals) {
            std::vector<Vec3> sl_n;
            std::vector<double> sl_conf;
            if (have_full_scanline) estimate_normals_scanline(full_pts_s, full_scan_info, s, sl_n, sl_conf);

            std::vector<Vec3> pca_n;
            std::vector<double> pca_conf;
            estimate_normals_pca(full_pts_s, s.estimate_normals_k, pca_n, pca_conf);

            for (int i = 0; i < NF; i++) {
                bool use_sl = have_full_scanline && i < (int)sl_n.size() &&
                              sl_n[i].squaredNorm() > 1e-10 && sl_conf[i] >= 0.20;
                if (use_sl) {
                    full_nrm_g[i] = sensor_R * sl_n[i];
                    full_conf[i] = sl_conf[i];
                    full_source[i] = 1;
                    n_sl_normals++;
                } else if (i < (int)pca_n.size() && pca_n[i].squaredNorm() > 1e-10) {
                    full_nrm_g[i] = sensor_R * pca_n[i];
                    full_conf[i] = pca_conf[i];
                    full_source[i] = 2;
                } else {
                    full_nrm_g[i] = Vec3::Zero();
                    full_conf[i] = s.nvt_min_conf;
                    full_source[i] = 3;
                }
            }
        } else {
            for (int i = 0; i < NF; i++) {
                full_nrm_g[i] = sensor_R * full_nrm_s[i];
                full_conf[i] = 1.0;
                full_source[i] = 0;
            }
        }

        // 3. Now apply process_every_n to the output of scan-line/normal
        // estimation. This keeps the QEM ingest cost controllable while the
        // scan-line estimator still had access to adjacent raw pixels.
        int step = std::max(1, s.process_every_n);
        std::vector<int> selected;
        selected.reserve(NF / step + 1);
        for (int i = 0; i < NF; i += step) selected.push_back(i);
        int N = (int)selected.size();

        std::vector<Vec3> pts_s(N), pts_g(N), nrm_g(N);
        std::vector<double> normal_conf(N, 1.0);
        std::vector<unsigned char> normal_source(N, 0);
        std::vector<ScanlinePointInfo> scan_info;
        if (have_full_scanline) scan_info.resize(N);
        for (int ii = 0; ii < N; ii++) {
            int fi = selected[ii];
            pts_s[ii] = full_pts_s[fi];
            pts_g[ii] = full_pts_g[fi];
            nrm_g[ii] = full_nrm_g[fi];
            normal_conf[ii] = full_conf[fi];
            normal_source[ii] = full_source[fi];
            if (have_full_scanline) scan_info[ii] = full_scan_info[fi];
        }
        bool have_scanline = have_full_scanline && (int)scan_info.size() == N;

        // 3b. Per-point world covariance for the probabilistic plane layer.
        std::vector<Mat3> point_cov_w(N, Mat3::Zero());
        if (s.enable_probabilistic_planes) {
            #ifdef HAS_OPENMP
            #pragma omp parallel for schedule(static, 1024)
            #endif
            for (int i = 0; i < N; i++) {
                point_cov_w[i] = lidar_point_cov_world(pts_s[i], sensor_R, s);
            }
        }

        // 4. Per-point: compute ray dir and orient initial normals.
        std::vector<Vec3>   dirs(N);
        std::vector<double> weights(N, 0.0);
        std::vector<char>   keep(N, 1);
        std::atomic<long>   n_grazing{0};

        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static, 1024)
        #endif
        for (int i = 0; i < N; i++) {
            Vec3 d = pts_g[i] - sensor_t;
            double dn = d.norm();
            if (dn < 1e-9) { keep[i] = 0; continue; }
            d /= dn;
            dirs[i] = d;

            double nl = nrm_g[i].norm();
            Vec3 n;
            if (nl < 1e-6) {
                n = -d;
                normal_conf[i] = s.nvt_min_conf;
                normal_source[i] = 3;
            } else {
                n = nrm_g[i] / nl;
            }

            // Orient toward sensor: n should point back along (-d).
            if (s.orient_to_sensor && n.dot(d) > 0) n = -n;
            nrm_g[i] = n;

            // Apply the scan-boundary confidence multiplier exactly once for
            // every normal source. Boundary points are kept for occupancy, but
            // their tangent-plane contribution is weakened.
            if (have_scanline && scan_info[i].valid && scan_info[i].boundary)
                normal_conf[i] *= s.scanline_boundary_conf;
        }

        // 4b. NVT/BEO normal denoising before QEM accumulation. By default it
        // only denoises PCA fallback normals; input and scan-line normals are
        // treated as trusted unless --nvt_all_normals is used.
        bool any_eligible_nvt = false;
        std::vector<char> nvt_mask(N, 0);
        for (int i = 0; i < N; i++) {
            bool eligible = !s.nvt_only_for_pca || normal_source[i] == 2;
            nvt_mask[i] = eligible ? 1 : 0;
            any_eligible_nvt = any_eligible_nvt || eligible;
        }
        bool run_nvt = s.enable_nvt && any_eligible_nvt;
        if (run_nvt) {
            denoise_normals_nvt(pts_g, sensor_t, s, nrm_g, normal_conf,
                                have_scanline ? &scan_info : nullptr,
                                &nvt_mask);
        }

        // 4c. Compute final LiDAR weights and reject grazing returns. The
        // confidence multiplier is soft: it modulates QEM weight but does not
        // zero out valid observations because of scan-local confidence noise.
        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static, 1024)
        #endif
        for (int i = 0; i < N; i++) {
            if (!keep[i]) continue;
            Vec3 n = nrm_g[i];
            double nl = n.norm();
            if (nl < 1e-9) { keep[i] = 0; continue; }
            n /= nl;
            nrm_g[i] = n;
            double inc_cos = std::abs(n.dot(dirs[i]));
            double nc = std::clamp(normal_conf[i], s.nvt_min_conf, 1.0);
            double floor = std::clamp(s.nvt_conf_soft_floor, 0.0, 1.0);
            double nc_soft = floor + (1.0 - floor) * nc;
            double prob_scale = s.enable_probabilistic_planes
                ? probabilistic_qem_weight_scale(n, point_cov_w[i], s)
                : 1.0;
            double base_weight = inc_cos * inc_cos * nc_soft * prob_scale;
            if (s.enable_seed_voxels && s.seed_ray_normal_fallback && normal_source[i] == 3) {
                // A ray-direction normal is only a fallback placeholder; keep
                // the endpoint as a hypothesis until region evidence validates it.
                keep[i] = 2;
                weights[i] = std::max(s.seed_min_hit_weight,
                                      base_weight * s.seed_hit_weight_scale);
                continue;
            }
            if (inc_cos < s.min_incident_cos) {
                // Old behavior was to drop these observations completely. For
                // vertex recall, keep moderate grazing endpoints as weak seed
                // hypotheses. They do not affect the main map or meshing until
                // they accumulate/promote; very grazing returns are still rejected.
                if (s.enable_seed_voxels && inc_cos >= s.seed_min_incident_cos) {
                    keep[i] = 2; // seed/hypothesis hit, not a confirmed hit
                    weights[i] = std::max(s.seed_min_hit_weight,
                                          base_weight * s.seed_hit_weight_scale);
                } else {
                    keep[i] = 0;
                    n_grazing.fetch_add(1, std::memory_order_relaxed);
                }
                continue;
            }
            weights[i] = base_weight;
        }
        n_grazing_total += n_grazing.load();

        // 5. OpenMP-parallel ingest with thread-local maps.
        int n_threads = 1;
        #ifdef HAS_OPENMP
        n_threads = omp_get_max_threads();
        #endif
        std::vector<std::unordered_map<VoxKey, VoxelCell, VoxHash>> tls(n_threads);
        std::vector<std::unordered_map<VoxKey, VoxelCell, VoxHash>> tls_seed(n_threads);
        std::vector<std::unordered_map<VoxKey, VoxelCell, VoxHash>> tls_seed_l1(n_threads);

        #ifdef HAS_OPENMP
        #pragma omp parallel
        #endif
        {
            int tid = 0;
            #ifdef HAS_OPENMP
            tid = omp_get_thread_num();
            #endif
            auto& local = tls[tid];
            auto& local_seed = tls_seed[tid];
            auto& local_seed_l1 = tls_seed_l1[tid];
            std::vector<VoxKey> ray_voxels;
            ray_voxels.reserve(64);

            #ifdef HAS_OPENMP
            #pragma omp for schedule(static, 512)
            #endif
            for (int i = 0; i < N; i++) {
                if (!keep[i]) continue;
                VoxKey hit_ijk = voxel_index(pts_g[i]);
                bool scan_boundary = have_scanline && scan_info[i].valid && scan_info[i].boundary;

                if (keep[i] == 2) {
                    // Weak endpoint evidence: store in seed map only. It can
                    // mature into the main map later, but does not immediately
                    // create surface/free labels or mesh vertices.
                    bool ray_normal = (normal_source[i] == 3);
                    local_seed[hit_ijk].add_hit(pts_g[i], nrm_g[i], weights[i],
                                                scan_idx, scan_boundary, ray_normal,
                                                eogm_hit_reliability(weights[i], true, scan_boundary, ray_normal),
                                                s.enable_probabilistic_planes ? &point_cov_w[i] : nullptr);
                    if (s.enable_seed_l1_parents) {
                        VoxKey pk = seed_l1_key_from_child(hit_ijk);
                        local_seed_l1[pk].add_hit(pts_g[i], nrm_g[i], weights[i],
                                                  scan_idx, scan_boundary, ray_normal,
                                                  eogm_hit_reliability(weights[i], true, scan_boundary, ray_normal),
                                                s.enable_probabilistic_planes ? &point_cov_w[i] : nullptr);
                    }
                    if (s.carve_rays && s.seed_carve_rays) {
                        traverse_ray(sensor_t, pts_g[i], ray_voxels);
                        double mw = weights[i] * s.seed_miss_weight_scale;
                        for (const VoxKey& vk : ray_voxels) {
                            if (vk == hit_ijk) continue;
                            local[vk].add_miss(mw, scan_idx, eogm_miss_reliability(mw, true));
                        }
                    }
                    continue;
                }

                local[hit_ijk].add_hit(pts_g[i], nrm_g[i], weights[i], scan_idx, scan_boundary, false,
                                       eogm_hit_reliability(weights[i], false, scan_boundary, false),
                                       s.enable_probabilistic_planes ? &point_cov_w[i] : nullptr);
                if (s.carve_rays) {
                    traverse_ray(sensor_t, pts_g[i], ray_voxels);
                    for (const VoxKey& vk : ray_voxels) {
                        if (vk == hit_ijk) continue;
                        local[vk].add_miss(weights[i], scan_idx, eogm_miss_reliability(weights[i], false));
                    }
                }
            }
        }

        // 6. Serial merge into the master map.
        long n_new_cells = 0;
        long n_misses = 0;
        long n_hits = 0;
        std::vector<VoxKey> this_scan_hit_keys; // drives incremental scaffold inheritance
        for (auto& tm : tls) {
            for (auto& [k, v] : tm) {
                auto it = cells.find(k);
                if (it == cells.end()) {
                    cells.emplace(k, v);
                    n_new_cells++;
                } else {
                    it->second.merge_from(v);
                }
                n_misses += v.miss_count;
                n_hits   += v.hit_count;
                if (v.hit_count > 0) { mark_component_dirty(k); this_scan_hit_keys.push_back(k); }
                // Free-space carving (miss-only) does not grow the dirty set, but
                // it may contradict existing persistent faces; flag a retirement
                // sweep for the next mesh export.
                if (v.miss_count > 0) mesh_free_carve_pending_ = true;
            }
        }

        long n_seed_hits = 0;
        long n_new_seed_cells = 0;
        long n_new_seed_l1_cells = 0;
        if (s.enable_seed_voxels) {
            for (auto& tm : tls_seed) {
                for (auto& [k, v] : tm) {
                    auto it = seed_cells.find(k);
                    if (it == seed_cells.end()) {
                        seed_cells.emplace(k, v);
                        n_new_seed_cells++;
                    } else {
                        it->second.merge_from(v);
                    }
                    n_seed_hits += v.hit_count;
                    dirty_seed_keys.insert(k);
                }
            }
            if (s.enable_seed_l1_parents) {
                for (auto& tm : tls_seed_l1) {
                    for (auto& [pk, v] : tm) {
                        auto it = seed_l1_cells.find(pk);
                        if (it == seed_l1_cells.end()) {
                            seed_l1_cells.emplace(pk, v);
                            n_new_seed_l1_cells++;
                        } else {
                            it->second.merge_from(v);
                        }
                        seed_l1_scan_ids[pk].insert(scan_idx);
                        dirty_seed_l1_keys.insert(pk);
                    }
                }
            }
        }
        // Parent-supported maturation gets first chance so L1-validated
        // children keep the parent_supported confidence tier. Remaining seeds
        // can still mature through the older self/neighbor path as weak_promoted.
        long n_l1_promoted = promote_seed_l1_children();
        long n_promoted = promote_seed_voxels();

        n_misses_total += n_misses;
        n_points_total += N;
        n_hits_total   += n_hits;
        n_seed_hits_total += n_seed_hits;
        n_seed_promoted_total += n_promoted;
        n_seed_l1_parent_supported_total += n_l1_promoted;
        scan_count = std::max(scan_count, scan_idx + 1);

        // Sparse-region scaffold structuring: inherit coarse-level QEM planes /
        // normals / evidence into empty fine voxels so sparse areas get virtual
        // vertices to mesh. Rebuilt INCREMENTALLY per scan -- only coarse parents
        // whose children were hit this scan are recomputed -- so the per-scan
        // cost scales with the scan footprint, not the whole map.
        if (hierarchical_scaffold_enabled())
            rebuild_hierarchical_scaffold_inheritance_incremental(scan_idx, this_scan_hit_keys);

        long n_boundary = 0;
        if (have_scanline) for (const auto& si : scan_info) if (si.valid && si.boundary) n_boundary++;
        double elapsed = now_sec() - t0;
        std::printf("  Scan %4d: in=%6d valid=%6d used=%6d kept=%6d seed=%5ld graz=%5ld threads=%d new_cells=%6ld "
                    "seed_new=%6ld l1_new=%6ld promoted=%5ld l1_prom=%5ld seed_pending=%zu l1_pending=%zu cells_total=%zu misses=%7ld scanline=%s sl_norm=%5ld boundary=%5ld  %.2fs\n",
                    scan_idx, N_in, NF, N, n_hits, n_seed_hits, n_grazing.load(), n_threads,
                    n_new_cells, n_new_seed_cells, n_new_seed_l1_cells, n_promoted, n_l1_promoted,
                    seed_cells.size(), seed_l1_cells.size(), cells.size(), n_misses,
                    have_scanline ? "yes" : "no", n_sl_normals, n_boundary, elapsed);
    }

    // ---- Vertex extraction + output ----

    struct VertexRecord {
        VoxKey key;
        Vec3 position;
        Vec3 normal;
        const char* kind;
        double residual;
        double normal_consistency;
        double scan_boundary_ratio;
        double eval2_over_eval1;
        double eval3_over_eval1;
        double weight_sum;
        int hit_count;
        int miss_count;
        int last_hit_scan;
        double eogm_bel_surface;
        double eogm_bel_free;
        double eogm_plaus_surface;
        double eogm_unknown;
        double eogm_conflict;
        double prob_plane_sigma;
        double prob_plane_radius;
        double prob_plane_min_eigen;
        double prob_plane_mid_eigen;
        double prob_plane_max_eigen;
        double prob_plane_normal_cov_trace;
        int prob_plane_points;
        int prob_plane_valid;
        int prob_plane_is_planar;
        int confidence_tier; // 0=confirmed, 1=weak_promoted, 2=parent_supported, 4=generated, 6=inherited
    };

    int cell_vertex_tier(const VoxelCell& v) const {
        if (v.label(s) == VoxelCell::Label::SURFACE &&
            v.confirmed_hit_count() >= s.min_hit_count_vertex) return 0;
        if (v.parent_supported_seed_count > 0) return 2;
        if (v.promoted_seed_count > 0) return 1;
        if (hierarchical_scaffold_enabled() && v.has_inherited()) return 6;
        return 0;
    }

    bool cell_vertex_export_ok(const VoxelCell& v) const {
        if (v.label(s) == VoxelCell::Label::SURFACE &&
            v.confirmed_hit_count() >= s.min_hit_count_vertex) return true;
        if (hierarchical_scaffold_enabled() && v.has_inherited()) {
            if (v.label(s) == VoxelCell::Label::FREE) return false;
            if (s.enable_eogm && (v.eogm_bel_free() > s.scaffold_max_bel_free ||
                                  v.eogm_conflict_mass() > s.scaffold_max_conflict))
                return false;
            return v.weight_eff() > 1e-12;
        }
        if (!s.export_weak_vertices) return false;
        if (v.promoted_seed_count <= 0) return false;
        if (v.hit_count < 1 || v.weight_sum < 1e-12) return false;
        if (v.occupancy_score() < s.seed_l1_child_min_occupancy) return false;
        double max_ray = (v.parent_supported_seed_count > 0)
            ? s.seed_l1_max_ray_normal_fraction
            : s.seed_max_ray_normal_fraction;
        if (v.ray_normal_fraction() > max_ray) return false;
        return true;
    }

    // Compute v* + classification for every surface/weak-supported cell, in parallel.
    std::vector<VertexRecord> build_vertex_table() const {
        std::vector<VoxKey> keys;
        if (active_region_) {
            // Persistent incremental remesh: iterate ONLY the dirty region keys
            // and look each up in the (huge) global cell map. This makes the
            // vertex-table build O(region) instead of O(total cells), so the
            // per-export cost no longer grows with overall map size.
            keys.reserve(active_region_->size());
            for (const VoxKey& k : *active_region_) {
                auto it = cells.find(k);
                if (it == cells.end()) continue;
                if (!cell_vertex_export_ok(it->second)) continue;
                keys.push_back(k);
            }
        } else {
            keys.reserve(cells.size());
            for (auto& [k, v] : cells) {
                if (!cell_vertex_export_ok(v)) continue;
                keys.push_back(k);
            }
        }
        int M = (int)keys.size();
        std::vector<VertexRecord> recs(M);
        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static, 256)
        #endif
        for (int i = 0; i < M; i++) {
            const VoxKey& k = keys[i];
            const VoxelCell& cell = cells.at(k);
            Vec3 anchor = voxel_center(k);
            Vec3 lo, hi; voxel_bounds(k, lo, hi);
            Vec3 v = cell.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
            auto info = cell.eigen_analysis(s);
            double l1 = std::max(info.evals[0], 1e-12);
            FittedProbPlane fp = cell.prob_plane.fit(s);
            recs[i] = {
                k, v, cell.mean_normal(), info.kind,
                cell.residual_per_observation(v),
                cell.normal_consistency(),
                cell.scan_boundary_ratio(),
                info.evals[1] / l1,
                info.evals[2] / l1,
                cell.weight_sum,
                cell.hit_count, cell.miss_count, cell.last_hit_scan,
                cell.eogm_bel_surface(), cell.eogm_bel_free(), cell.eogm_plaus_surface(),
                cell.eogm_ignorance(), cell.eogm_conflict_mass(),
                fp.valid ? fp.sigma_plane : 0.0,
                fp.valid ? fp.radius : 0.0,
                fp.valid ? fp.min_eigen_value : 0.0,
                fp.valid ? fp.mid_eigen_value : 0.0,
                fp.valid ? fp.max_eigen_value : 0.0,
                fp.valid ? fp.normal_cov_trace : 0.0,
                fp.points_size,
                fp.valid ? 1 : 0,
                fp.planar ? 1 : 0,
                cell_vertex_tier(cell)
            };
        }
        return recs;
    }

    std::vector<VertexRecord> build_seed_vertex_table() const {
        std::vector<VoxKey> keys;
        keys.reserve(seed_cells.size());
        for (const auto& [k, v] : seed_cells) {
            if (v.hit_count < s.seed_export_min_hits) continue;
            if (v.weight_sum < 1e-12) continue;
            auto mit = cells.find(k);
            if (mit != cells.end() && mit->second.occupancy_score() < s.seed_promote_min_occupancy)
                continue;
            Vec3 n = v.mean_normal();
            if (!normalized_or_zero(n)) continue;
            keys.push_back(k);
        }
        int M = (int)keys.size();
        std::vector<VertexRecord> recs(M);
        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static, 256)
        #endif
        for (int i = 0; i < M; i++) {
            const VoxKey& k = keys[i];
            const VoxelCell& cell = seed_cells.at(k);
            Vec3 anchor = voxel_center(k);
            Vec3 lo, hi; voxel_bounds(k, lo, hi);
            Vec3 v = cell.solve_vertex(anchor, s.lambda_p, s.clamp_to_voxel, lo, hi);
            auto info = cell.eigen_analysis(s);
            double l1 = std::max(info.evals[0], 1e-12);
            FittedProbPlane fp = cell.prob_plane.fit(s);
            recs[i] = {
                k, v, cell.mean_normal(), info.kind,
                cell.residual_per_observation(v),
                cell.normal_consistency(),
                cell.scan_boundary_ratio(),
                info.evals[1] / l1,
                info.evals[2] / l1,
                cell.weight_sum,
                cell.hit_count, cell.miss_count, cell.last_hit_scan,
                cell.eogm_bel_surface(), cell.eogm_bel_free(), cell.eogm_plaus_surface(),
                cell.eogm_ignorance(), cell.eogm_conflict_mass(),
                fp.valid ? fp.sigma_plane : 0.0,
                fp.valid ? fp.radius : 0.0,
                fp.valid ? fp.min_eigen_value : 0.0,
                fp.valid ? fp.mid_eigen_value : 0.0,
                fp.valid ? fp.max_eigen_value : 0.0,
                fp.valid ? fp.normal_cov_trace : 0.0,
                fp.points_size,
                fp.valid ? 1 : 0,
                fp.planar ? 1 : 0,
                3
            };
        }
        return recs;
    }

    void export_seed_ply_and_csv(const std::string& ply_path) const {
        if (!s.enable_seed_voxels || !s.export_seed_vertices) return;
        auto recs = build_seed_vertex_table();
        int M = (int)recs.size();
        std::string stem = ply_path;
        size_t dot = stem.rfind('.');
        if (dot != std::string::npos) stem.resize(dot);
        std::string seed_ply = stem + "_seeds.ply";
        std::string seed_csv = stem + "_seeds_meta.csv";
        {
            std::ofstream out(seed_ply);
            if (!out) {
                std::fprintf(stderr, "Cannot write %s\n", seed_ply.c_str());
                return;
            }
            out << "ply\nformat ascii 1.0\n"
                << "comment pending weak/hypothesis QEM vertices; not used for meshing until promoted\n"
                << "comment voxel_size " << s.voxel_size << "\n"
                << "comment scans " << scan_count << "\n"
                << "element vertex " << M << "\n"
                << "property float x\nproperty float y\nproperty float z\n"
                << "property float nx\nproperty float ny\nproperty float nz\n"
                << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                << "property float residual\nproperty float consistency\n"
                << "property int last_scan\n"
                << "property uchar confidence_tier\n"
                << "end_header\n";
            for (const auto& r : recs) {
                out << r.position[0] << ' ' << r.position[1] << ' ' << r.position[2]
                    << ' ' << r.normal[0] << ' ' << r.normal[1] << ' ' << r.normal[2]
                    << " 40 170 255 "
                    << r.residual << ' ' << r.normal_consistency << ' ' << r.last_hit_scan
                    << ' ' << r.confidence_tier << '\n';
            }
        }
        {
            std::ofstream out(seed_csv);
            if (!out) {
                std::fprintf(stderr, "Cannot write %s\n", seed_csv.c_str());
                return;
            }
            out << "seed_idx,voxel_i,voxel_j,voxel_k,kind,"
                   "residual,normal_consistency,scan_boundary_ratio,"
                   "lambda2_over_lambda1,lambda3_over_lambda1,weight_sum,"
                   "hit_count,miss_count,last_hit_scan,eogm_bel_surface,eogm_bel_free,eogm_plaus_surface,eogm_unknown,eogm_conflict,prob_plane_sigma,prob_plane_radius,prob_plane_min_eigen,prob_plane_mid_eigen,prob_plane_max_eigen,prob_plane_normal_cov_trace,prob_plane_points,prob_plane_valid,prob_plane_is_planar,confidence_tier,confidence_tier_name,promotion_state\n";
            for (int i = 0; i < M; i++) {
                const auto& r = recs[i];
                out << i << ',' << r.key.i << ',' << r.key.j << ',' << r.key.k
                    << ',' << r.kind << ',' << r.residual << ',' << r.normal_consistency
                    << ',' << r.scan_boundary_ratio
                    << ',' << r.eval2_over_eval1
                    << ',' << r.eval3_over_eval1
                    << ',' << r.weight_sum
                    << ',' << r.hit_count << ',' << r.miss_count
                    << ',' << r.last_hit_scan
                    << ',' << r.eogm_bel_surface
                    << ',' << r.eogm_bel_free
                    << ',' << r.eogm_plaus_surface
                    << ',' << r.eogm_unknown
                    << ',' << r.eogm_conflict
                    << ',' << r.prob_plane_sigma
                    << ',' << r.prob_plane_radius
                    << ',' << r.prob_plane_min_eigen
                    << ',' << r.prob_plane_mid_eigen
                    << ',' << r.prob_plane_max_eigen
                    << ',' << r.prob_plane_normal_cov_trace
                    << ',' << r.prob_plane_points
                    << ',' << r.prob_plane_valid
                    << ',' << r.prob_plane_is_planar
                    << ',' << r.confidence_tier
                    << ',' << VoxelCell::tier_name(r.confidence_tier)
                    << ",pending\n";
            }
        }
        std::printf("Exported %d pending seed vertices -> %s\n  metadata -> %s\n",
                    M, seed_ply.c_str(), seed_csv.c_str());
    }

    void export_ply_and_csv(const std::string& ply_path) const {
        auto recs = build_vertex_table();
        int M = (int)recs.size();

        // ASCII PLY: positions + normals only. (Metadata goes to the CSV.)
        {
            std::ofstream out(ply_path);
            if (!out) {
                std::fprintf(stderr, "Cannot write %s\n", ply_path.c_str());
                return;
            }
            out << "ply\nformat ascii 1.0\n"
                << "comment Phase 2 voxel-anchored QEM map\n"
                << "comment voxel_size " << s.voxel_size << "\n"
                << "comment scans "      << scan_count   << "\n"
                << "element vertex " << M << "\n"
                << "property float x\nproperty float y\nproperty float z\n"
                << "property float nx\nproperty float ny\nproperty float nz\n"
                << "property uchar confidence_tier\n"
                << "end_header\n";
            for (auto& r : recs) {
                out << r.position[0] << ' ' << r.position[1] << ' ' << r.position[2]
                    << ' ' << r.normal[0]   << ' ' << r.normal[1]   << ' ' << r.normal[2]
                    << ' ' << r.confidence_tier
                    << '\n';
            }
        }

        // CSV sidecar: per-vertex metadata. Same row order as the PLY.
        std::string csv_path = ply_path;
        size_t dot = csv_path.rfind('.');
        if (dot != std::string::npos) csv_path.resize(dot);
        csv_path += "_meta.csv";
        {
            std::ofstream out(csv_path);
            if (!out) {
                std::fprintf(stderr, "Cannot write %s\n", csv_path.c_str());
                return;
            }
            out << "vertex_idx,voxel_i,voxel_j,voxel_k,kind,"
                   "residual,normal_consistency,scan_boundary_ratio,"
                   "lambda2_over_lambda1,lambda3_over_lambda1,weight_sum,"
                   "hit_count,miss_count,last_hit_scan,eogm_bel_surface,eogm_bel_free,eogm_plaus_surface,eogm_unknown,eogm_conflict,prob_plane_sigma,prob_plane_radius,prob_plane_min_eigen,prob_plane_mid_eigen,prob_plane_max_eigen,prob_plane_normal_cov_trace,prob_plane_points,prob_plane_valid,prob_plane_is_planar,confidence_tier,confidence_tier_name\n";
            for (int i = 0; i < M; i++) {
                const auto& r = recs[i];
                out << i << ',' << r.key.i << ',' << r.key.j << ',' << r.key.k
                    << ',' << r.kind << ',' << r.residual << ',' << r.normal_consistency
                    << ',' << r.scan_boundary_ratio
                    << ',' << r.eval2_over_eval1
                    << ',' << r.eval3_over_eval1
                    << ',' << r.weight_sum
                    << ',' << r.hit_count << ',' << r.miss_count
                    << ',' << r.last_hit_scan
                    << ',' << r.eogm_bel_surface
                    << ',' << r.eogm_bel_free
                    << ',' << r.eogm_plaus_surface
                    << ',' << r.eogm_unknown
                    << ',' << r.eogm_conflict
                    << ',' << r.prob_plane_sigma
                    << ',' << r.prob_plane_radius
                    << ',' << r.prob_plane_min_eigen
                    << ',' << r.prob_plane_mid_eigen
                    << ',' << r.prob_plane_max_eigen
                    << ',' << r.prob_plane_normal_cov_trace
                    << ',' << r.prob_plane_points
                    << ',' << r.prob_plane_valid
                    << ',' << r.prob_plane_is_planar
                    << ',' << r.confidence_tier
                    << ',' << VoxelCell::tier_name(r.confidence_tier) << '\n';
            }
        }
        std::printf("Exported %d vertices -> %s\n  metadata -> %s\n",
                    M, ply_path.c_str(), csv_path.c_str());
        export_seed_ply_and_csv(ply_path);
    }


    // ---- Phase 3A local smooth-patch meshing ---- //
    //
    // This mesher is deliberately conservative and incremental-friendly. It
    // does not run global Delaunay/Poisson. Instead it uses the voxel keys as a
    // sparse lattice, connects nearby QEM vertices whose normals agree, then
    // creates local fan triangles around each vertex. This meshes smooth
    // patches first. Creases where two rank-1 sheets meet should be handled by
    // a later seam-stitching stage rather than by merging all neighboring QEMs.

    struct MeshFace { int a, b, c; };
    struct MeshData {
        std::vector<VertexRecord> verts;
        std::vector<MeshFace> faces;
    };

    struct PersistentComponentFace { VoxKey a, b, c; };
    struct PersistentComponentEdge { VoxKey a, b; };

    struct PersistentQEMSurfaceComponent {
        int id = -1;
        std::unordered_set<VoxKey, VoxHash> vertex_keys;
        std::vector<PersistentComponentEdge> owned_edges;
        std::vector<PersistentComponentFace> owned_faces;
        std::unordered_set<VoxKey, VoxHash> boundary_vertex_keys;
        std::unordered_map<VoxKey, double, VoxHash> boundary_radius;
        Vec3 point = Vec3::Zero();
        Vec3 normal = Vec3::Zero();
        double sqrt_residual = 0.0;
        int face_count = 0;
        bool dirty = false;
        int last_refresh_scan = -1;
    };

    mutable std::unordered_map<int, PersistentQEMSurfaceComponent> persistent_components;
    mutable std::unordered_map<VoxKey, int, VoxHash> persistent_component_by_key;
    mutable int next_persistent_component_id = 1;
    mutable long persistent_component_refresh_count = 0;

    // ---- PlanarMesh-style persistent incremental mesh state ---- //
    // The single mesh that survives across scans. build_mesh_for_mode() keeps
    // this up to date by re-meshing only the dirty region each export.
    mutable MeshData persistent_mesh_;
    mutable bool     persistent_mesh_initialized_ = false;
    mutable long     persistent_mesh_update_count = 0;
    // Set when ray-carving produced free-space evidence since the last mesh
    // export. Miss-only cells are intentionally NOT added to the growth-dirty
    // set (so misses cannot explode growth neighborhoods), but they can still
    // contradict existing faces, so this flag forces a face-retirement sweep on
    // the next export even when no hit-driven dirty cells exist.
    mutable bool     mesh_free_carve_pending_ = false;
    // When non-null, build_vertex_table() (and therefore every mesh builder)
    // only emits vertices for cells whose key is in this set. Used to confine
    // a rebuild to the dirty neighborhood.
    mutable const std::unordered_set<VoxKey, VoxHash>* active_region_ = nullptr;

    struct FaceKey {
        int a, b, c;
        bool operator==(const FaceKey& o) const noexcept {
            return a == o.a && b == o.b && c == o.c;
        }
    };
    struct FaceKeyHash {
        size_t operator()(const FaceKey& f) const noexcept {
            uint64_t h = (uint64_t)(uint32_t)f.a * 73856093ULL;
            h ^= (uint64_t)(uint32_t)f.b * 19349663ULL;
            h ^= (uint64_t)(uint32_t)f.c * 83492791ULL;
            return (size_t)h;
        }
    };

    static FaceKey sorted_face_key(int a, int b, int c) {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return FaceKey{a, b, c};
    }

    static uint64_t edge_key(int a, int b) {
        if (a > b) std::swap(a, b);
        return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    }

    static int edge_use_count(const std::unordered_map<uint64_t, int>& edge_count, int a, int b) {
        auto it = edge_count.find(edge_key(a, b));
        return it == edge_count.end() ? 0 : it->second;
    }

    static bool triangle_edges_can_accept(const std::unordered_map<uint64_t, int>& edge_count,
                                          int a, int b, int c) {
        if (a == b || a == c || b == c) return false;
        return edge_use_count(edge_count, a, b) < 2 &&
               edge_use_count(edge_count, b, c) < 2 &&
               edge_use_count(edge_count, c, a) < 2;
    }

    static void register_triangle_edges(std::unordered_map<uint64_t, int>& edge_count,
                                        int a, int b, int c) {
        edge_count[edge_key(a, b)]++;
        edge_count[edge_key(b, c)]++;
        edge_count[edge_key(c, a)]++;
    }

    int persistent_id_by_majority_overlap(const std::unordered_set<VoxKey, VoxHash>& keys) const {
        std::unordered_map<int, int> votes;
        int best_id = -1, best_votes = 0;
        for (const VoxKey& k : keys) {
            auto it = persistent_component_by_key.find(k);
            if (it == persistent_component_by_key.end()) continue;
            int v = ++votes[it->second];
            if (v > best_votes) { best_votes = v; best_id = it->second; }
        }
        return best_id;
    }

    bool current_component_is_dirty(const MeshData& mesh,
                                    const std::vector<int>& ids,
                                    const std::vector<std::pair<int,int>>& boundary_edges) const {
        if (!s.component_growth_dirty_only) return true;
        if (dirty_component_voxel_keys.empty()) return false;
        for (int id : ids) {
            if (id >= 0 && id < (int)mesh.verts.size() && component_key_is_dirty(mesh.verts[id].key)) return true;
        }
        for (const auto& e : boundary_edges) {
            if (e.first >= 0 && e.first < (int)mesh.verts.size() && component_key_is_dirty(mesh.verts[e.first].key)) return true;
            if (e.second >= 0 && e.second < (int)mesh.verts.size() && component_key_is_dirty(mesh.verts[e.second].key)) return true;
        }
        return false;
    }

    static bool normalized_or_zero(Vec3& n) {
        double nn = n.norm();
        if (nn < 1e-12) { n = Vec3::Zero(); return false; }
        n /= nn;
        return true;
    }

    bool mesh_vertex_ok(const VertexRecord& r, int min_last_hit_scan, int max_last_hit_scan) const {
        // Option A / seed-recall mode: promoted weak and parent-supported vertices
        // may participate in baseline connectivity after they have passed promotion gates.
        const bool inherited = (r.confidence_tier == 6);
        if (inherited && !hierarchical_scaffold_enabled()) return false;
        if (!inherited && r.hit_count < s.mesh_min_hit_count) return false;
        if (min_last_hit_scan >= 0 && r.last_hit_scan < min_last_hit_scan) return false;
        if (max_last_hit_scan >= 0 && r.last_hit_scan > max_last_hit_scan) return false;
        if (s.mesh_max_sqrt_residual > 0.0) {
            if (!std::isfinite(r.residual)) return false;
            double mesh_residual_limit = s.mesh_max_sqrt_residual;
            if (s.enable_probabilistic_planes && r.prob_plane_sigma > 0.0) {
                const double prob_limit = s.prob_gate_sigma * r.prob_plane_sigma;
                mesh_residual_limit = std::clamp(prob_limit,
                                                 s.mesh_max_sqrt_residual * s.prob_gate_min_factor,
                                                 s.mesh_max_sqrt_residual * s.prob_gate_max_factor);
            }
            if (std::sqrt(std::max(0.0, r.residual)) > mesh_residual_limit) return false;
        }
        if (r.normal_consistency < s.mesh_min_normal_consistency) return false;
        if (r.scan_boundary_ratio > s.mesh_max_boundary_ratio) return false;
        return true;
    }

    bool mesh_pair_ok(const VertexRecord& a, const VertexRecord& b) const {
        double d = (a.position - b.position).norm();
        if (d > s.mesh_neighbor_radius_factor * s.voxel_size) return false;
        Vec3 na = a.normal, nb = b.normal;
        if (!normalized_or_zero(na) || !normalized_or_zero(nb)) return false;
        if (std::abs(na.dot(nb)) < s.mesh_normal_dot) return false;
        return true;
    }

    bool mesh_triangle_ok(const VertexRecord& a, const VertexRecord& b, const VertexRecord& c) const {
        const Vec3& pa = a.position;
        const Vec3& pb = b.position;
        const Vec3& pc = c.position;
        double e01 = (pa - pb).norm();
        double e12 = (pb - pc).norm();
        double e20 = (pc - pa).norm();
        double max_edge = s.mesh_max_edge_factor * s.voxel_size;
        if (e01 > max_edge || e12 > max_edge || e20 > max_edge) return false;

        Vec3 tri_n = (pb - pa).cross(pc - pa);
        double twice_area = tri_n.norm();
        double min_area = s.mesh_min_area_factor * s.voxel_size * s.voxel_size;
        if (twice_area < 2.0 * min_area) return false;
        tri_n /= twice_area;

        Vec3 avg_n = a.normal;
        Vec3 nb = b.normal;
        Vec3 nc = c.normal;
        if (!normalized_or_zero(avg_n) || !normalized_or_zero(nb) || !normalized_or_zero(nc)) return false;
        if (nb.dot(avg_n) < 0.0) nb = -nb;
        if (nc.dot(avg_n) < 0.0) nc = -nc;
        avg_n += nb + nc;
        if (!normalized_or_zero(avg_n)) return false;
        if (std::abs(tri_n.dot(avg_n)) < s.mesh_triangle_normal_dot) return false;

        if (s.mesh_reject_free_centroid) {
            Vec3 cent = (pa + pb + pc) / 3.0;
            auto it = cells.find(voxel_index(cent));
            if (it != cells.end() && it->second.label(s) == VoxelCell::Label::FREE) return false;
        }
        return true;
    }

    MeshData build_local_smooth_mesh(int min_last_hit_scan = -1,
                                     int max_last_hit_scan = -1) const {
        MeshData mesh;
        auto recs = build_vertex_table();
        mesh.verts.reserve(recs.size());

        std::unordered_map<VoxKey, int, VoxHash> idx_by_key;
        idx_by_key.reserve(recs.size() * 2 + 1);
        for (const auto& r : recs) {
            if (!mesh_vertex_ok(r, min_last_hit_scan, max_last_hit_scan)) continue;
            int ni = (int)mesh.verts.size();
            mesh.verts.push_back(r);
            idx_by_key[r.key] = ni;
        }
        const int N = (int)mesh.verts.size();
        if (N < 3) return mesh;

        std::vector<std::vector<int>> adj(N);
        std::unordered_set<uint64_t> edge_set;
        edge_set.reserve((size_t)N * 12);

        for (int i = 0; i < N; i++) {
            const VoxKey& k = mesh.verts[i].key;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        VoxKey nk{(int32_t)(k.i + dx), (int32_t)(k.j + dy), (int32_t)(k.k + dz)};
                        auto it = idx_by_key.find(nk);
                        if (it == idx_by_key.end()) continue;
                        int j = it->second;
                        if (j <= i) continue;
                        if (!mesh_pair_ok(mesh.verts[i], mesh.verts[j])) continue;
                        adj[i].push_back(j);
                        adj[j].push_back(i);
                        edge_set.insert(edge_key(i, j));
                    }
                }
            }
        }

        std::unordered_set<FaceKey, FaceKeyHash> face_set;
        face_set.reserve((size_t)N * 4);
        double max_fan_angle = std::clamp(s.mesh_max_fan_angle, 0.1, 6.283185307179586);

        for (int i = 0; i < N; i++) {
            if (adj[i].size() < 2) continue;
            Vec3 ni = mesh.verts[i].normal;
            if (!normalized_or_zero(ni)) continue;

            // Tangent basis for angular ordering around the vertex.
            Vec3 ref = (std::abs(ni.z()) < 0.9) ? Vec3(0,0,1) : Vec3(1,0,0);
            Vec3 u = ref.cross(ni);
            if (!normalized_or_zero(u)) continue;
            Vec3 v = ni.cross(u);
            if (!normalized_or_zero(v)) continue;

            struct AngularNeighbor { double ang; double rad; int idx; };
            std::vector<AngularNeighbor> nbrs;
            nbrs.reserve(adj[i].size());
            for (int j : adj[i]) {
                Vec3 d = mesh.verts[j].position - mesh.verts[i].position;
                double x = d.dot(u), y = d.dot(v);
                double rad = std::sqrt(x*x + y*y);
                if (rad < 1e-12) continue;
                nbrs.push_back({std::atan2(y, x), rad, j});
            }
            if (nbrs.size() < 2) continue;
            std::sort(nbrs.begin(), nbrs.end(), [](const auto& a, const auto& b) {
                return a.ang < b.ang;
            });

            int K = (int)nbrs.size();
            for (int a = 0; a < K; a++) {
                int bidx = (a + 1) % K;
                double gap = nbrs[bidx].ang - nbrs[a].ang;
                if (bidx == 0) gap += 6.283185307179586;
                if (gap > max_fan_angle) continue;

                int j = nbrs[a].idx;
                int k = nbrs[bidx].idx;
                if (j == k) continue;
                if (edge_set.find(edge_key(j, k)) == edge_set.end()) continue;
                if (!mesh_triangle_ok(mesh.verts[i], mesh.verts[j], mesh.verts[k])) continue;

                FaceKey fk = sorted_face_key(i, j, k);
                if (fk.a == fk.b || fk.b == fk.c || fk.a == fk.c) continue;
                if (face_set.insert(fk).second) {
                    int fa = i, fb = j, fc = k;
                    Vec3 tri_n = (mesh.verts[fb].position - mesh.verts[fa].position)
                               .cross(mesh.verts[fc].position - mesh.verts[fa].position);
                    Vec3 avg_n = mesh.verts[fa].normal;
                    Vec3 nb = mesh.verts[fb].normal;
                    Vec3 nc = mesh.verts[fc].normal;
                    if (normalized_or_zero(avg_n) && normalized_or_zero(nb) && normalized_or_zero(nc)) {
                        if (nb.dot(avg_n) < 0.0) nb = -nb;
                        if (nc.dot(avg_n) < 0.0) nc = -nc;
                        avg_n += nb + nc;
                        if (normalized_or_zero(avg_n) && tri_n.dot(avg_n) < 0.0) std::swap(fb, fc);
                    }
                    mesh.faces.push_back({fa, fb, fc});
                }
            }
        }
        return mesh;
    }



    // ---- Phase 3B open-scene dual contouring / surface-net meshing ---- //
    //
    // This is intentionally not classic watertight DC. Classic DC assumes a
    // reliable signed field everywhere. LiDAR scenes do not have that: they
    // have SURFACE hits, ray-carved FREE space, and a lot of UNKNOWN space.
    // Therefore this builder emits quads only where the surface has observed
    // free-space support. Unknown frontiers remain open; they are not capped.
    //
    // The mesh vertices are still the QEM vertices from surface voxels. The
    // connectivity is grid-based: for each exposed face of a surface voxel
    // against a FREE voxel, try to build 2x2 surface-voxel quads in the two
    // tangential directions. This behaves like an open-scene surface-net / DC
    // hybrid and is suitable for submap/incremental export.

    bool dc_vertex_ok(const VertexRecord& r, int min_last_hit_scan, int max_last_hit_scan) const {
        return mesh_vertex_ok(r, min_last_hit_scan, max_last_hit_scan);
    }

    bool dc_pair_ok(const VertexRecord& a, const VertexRecord& b) const {
        double d = (a.position - b.position).norm();
        if (d > s.dc_max_edge_factor * s.voxel_size) return false;
        Vec3 na = a.normal, nb = b.normal;
        if (!normalized_or_zero(na) || !normalized_or_zero(nb)) return false;
        // Sign-invariant because QEM plane normals may be flipped by source.
        if (std::abs(na.dot(nb)) < s.dc_normal_dot) return false;
        return true;
    }

    bool dc_triangle_ok(const VertexRecord& a, const VertexRecord& b, const VertexRecord& c,
                        const Vec3& expected_normal) const {
        const Vec3& pa = a.position;
        const Vec3& pb = b.position;
        const Vec3& pc = c.position;
        double e01 = (pa - pb).norm();
        double e12 = (pb - pc).norm();
        double e20 = (pc - pa).norm();
        double max_edge = s.dc_max_edge_factor * s.voxel_size;
        if (e01 > max_edge || e12 > max_edge || e20 > max_edge) return false;

        Vec3 tri_n = (pb - pa).cross(pc - pa);
        double twice_area = tri_n.norm();
        double min_area = s.dc_min_area_factor * s.voxel_size * s.voxel_size;
        if (twice_area < 2.0 * min_area) return false;
        tri_n /= twice_area;

        Vec3 en = expected_normal;
        if (!normalized_or_zero(en)) return false;
        if (std::abs(tri_n.dot(en)) < s.dc_triangle_normal_dot) return false;

        if (s.mesh_reject_free_centroid && !s.dc_require_free) {
            Vec3 cent = (pa + pb + pc) / 3.0;
            auto it = cells.find(voxel_index(cent));
            // In free-supported dual mode, faces intentionally lie on the
            // SURFACE/FREE interface, so a centroid may quantize into a FREE
            // voxel. Do this check only for surface-only dual mode.
            if (it != cells.end() && it->second.label(s) == VoxelCell::Label::FREE) return false;
        }
        return true;
    }

    bool dc_quad_ok(const MeshData& mesh, int a, int b, int c, int d,
                    const Vec3& expected_normal) const {
        if (a < 0 || b < 0 || c < 0 || d < 0) return false;
        if (a == b || a == c || a == d || b == c || b == d || c == d) return false;
        const auto& va = mesh.verts[a];
        const auto& vb = mesh.verts[b];
        const auto& vc = mesh.verts[c];
        const auto& vd = mesh.verts[d];
        if (!dc_pair_ok(va, vb) || !dc_pair_ok(vb, vc) || !dc_pair_ok(vc, vd) || !dc_pair_ok(vd, va)) return false;
        if (!dc_pair_ok(va, vc) && !dc_pair_ok(vb, vd)) return false;
        return true;
    }

    static VoxKey add_key_axis(const VoxKey& k, int axis, int step) {
        VoxKey out = k;
        if      (axis == 0) out.i += step;
        else if (axis == 1) out.j += step;
        else                out.k += step;
        return out;
    }

    static Vec3 axis_vec(int axis, int sign) {
        Vec3 v = Vec3::Zero();
        v[axis] = (double)sign;
        return v;
    }

    static VoxKey add_key2(const VoxKey& k, int axis_a, int step_a, int axis_b, int step_b) {
        return add_key_axis(add_key_axis(k, axis_a, step_a), axis_b, step_b);
    }

    void add_oriented_triangle(MeshData& mesh,
                               std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                               int a, int b, int c,
                               const Vec3& expected_normal) const {
        if (a == b || b == c || a == c) return;
        if (!dc_triangle_ok(mesh.verts[a], mesh.verts[b], mesh.verts[c], expected_normal)) return;

        int fa = a, fb = b, fc = c;
        Vec3 tri_n = (mesh.verts[fb].position - mesh.verts[fa].position)
                   .cross(mesh.verts[fc].position - mesh.verts[fa].position);
        Vec3 en = expected_normal;
        if (normalized_or_zero(en) && tri_n.dot(en) < 0.0) std::swap(fb, fc);

        FaceKey fk = sorted_face_key(fa, fb, fc);
        if (face_set.insert(fk).second) mesh.faces.push_back({fa, fb, fc});
    }

    bool surface_has_free_neighbor(const VoxKey& k, int axis, int sign) const {
        auto it = cells.find(add_key_axis(k, axis, sign));
        return it != cells.end() && it->second.label(s) == VoxelCell::Label::FREE;
    }

    bool surface_interface_supported(const std::unordered_map<VoxKey, int, VoxHash>& idx_by_key,
                                     const VoxKey& k, int axis, int sign) const {
        if (!s.dc_require_free) return true;
        // All four surface cells in a quad do not need free on the same side;
        // one supported face is enough to prevent creating only hidden/internal
        // surface-net sheets. The stricter all-four version is too brittle on
        // sparse scans and causes holes around grazing observations.
        return surface_has_free_neighbor(k, axis, sign);
    }

    MeshData build_open_dual_contour_mesh(int min_last_hit_scan = -1,
                                          int max_last_hit_scan = -1) const {
        MeshData mesh;
        auto recs = build_vertex_table();
        mesh.verts.reserve(recs.size());

        std::unordered_map<VoxKey, int, VoxHash> idx_by_key;
        idx_by_key.reserve(recs.size() * 2 + 1);
        for (const auto& r : recs) {
            if (!dc_vertex_ok(r, min_last_hit_scan, max_last_hit_scan)) continue;
            int ni = (int)mesh.verts.size();
            mesh.verts.push_back(r);
            idx_by_key[r.key] = ni;
        }
        if (mesh.verts.size() < 3) return mesh;

        std::unordered_set<FaceKey, FaceKeyHash> face_set;
        face_set.reserve(mesh.verts.size() * 6);

        auto idx = [&](const VoxKey& k) -> int {
            auto it = idx_by_key.find(k);
            return it == idx_by_key.end() ? -1 : it->second;
        };

        // For each exposed surface/free interface, build tangential 2x2 quads
        // using the QEM vertices of same-side surface voxels. This meshes open
        // surfaces while avoiding UNKNOWN caps.
        for (const auto& [k, base_idx] : idx_by_key) {
            (void)base_idx;
            for (int axis = 0; axis < 3; axis++) {
                int u_axis = (axis + 1) % 3;
                int v_axis = (axis + 2) % 3;
                for (int sign : {-1, 1}) {
                    if (!surface_interface_supported(idx_by_key, k, axis, sign)) continue;
                    Vec3 expected = axis_vec(axis, sign);

                    // Four possible 2x2 origins around the current exposed face.
                    // This fills quads even when k is not the low corner of the
                    // local surface-net patch.
                    for (int ou : {-1, 0}) {
                        for (int ov : {-1, 0}) {
                            VoxKey k00 = add_key2(k, u_axis, ou,     v_axis, ov);
                            VoxKey k10 = add_key2(k, u_axis, ou + 1, v_axis, ov);
                            VoxKey k01 = add_key2(k, u_axis, ou,     v_axis, ov + 1);
                            VoxKey k11 = add_key2(k, u_axis, ou + 1, v_axis, ov + 1);
                            int i00 = idx(k00), i10 = idx(k10), i01 = idx(k01), i11 = idx(k11);
                            if (!dc_quad_ok(mesh, i00, i10, i11, i01, expected)) continue;

                            // If requiring free evidence, at least one cell in
                            // the quad should be exposed toward this free side.
                            if (s.dc_require_free) {
                                bool any_free = surface_has_free_neighbor(k00, axis, sign)
                                             || surface_has_free_neighbor(k10, axis, sign)
                                             || surface_has_free_neighbor(k01, axis, sign)
                                             || surface_has_free_neighbor(k11, axis, sign);
                                if (!any_free) continue;
                            }

                            // Diagonal choice: use the shorter 3D diagonal to
                            // reduce folds on curved surfaces.
                            double d_a = (mesh.verts[i00].position - mesh.verts[i11].position).squaredNorm();
                            double d_b = (mesh.verts[i10].position - mesh.verts[i01].position).squaredNorm();
                            if (d_a <= d_b) {
                                add_oriented_triangle(mesh, face_set, i00, i10, i11, expected);
                                add_oriented_triangle(mesh, face_set, i00, i11, i01, expected);
                            } else {
                                add_oriented_triangle(mesh, face_set, i00, i10, i01, expected);
                                add_oriented_triangle(mesh, face_set, i10, i11, i01, expected);
                            }
                        }
                    }
                }
            }
        }
        return mesh;
    }


    // ---- Phase 3C grid-edge dual contouring (classic DC) ---- //
    //
    // Iterates voxel grid EDGES rather than faces. Each grid edge in the 3D
    // grid is shared by exactly 4 voxels arranged in a 2x2 perpendicular to
    // the edge. For each edge whose 4 surrounding voxels are all SURFACE,
    // emit one quad connecting their QEM vertices.
    //
    // Iteration trick: each grid edge has a unique "lex-min" cell among its
    // 4 surrounding cells (the cell at the smallest (i,j,k) in the
    // perpendicular plane). The lex-min cell processes the edge at its high
    // corner, in each axis direction. This visits every edge exactly once,
    // unlike the face-iteration surface-net builder which visits each patch
    // ~16x and depends on face_set deduplication.
    //
    // Compared to build_open_dual_contour_mesh (surface nets):
    //   + One quad per surface-crossing edge (no per-axis duplication).
    //   + No zig-zag overlap from multiple axes meshing the same surface.
    //   + Better performance on dense maps.
    //   - Same fundamental limitation as cell-labelled DC: surfaces that are
    //     both thin (1 voxel) AND tilted relative to grid axes can have
    //     holes, because no edge will have 4 SURFACE cells around it.
    //     Axis-aligned walls/floors mesh densely; oblique surfaces with
    //     enough observation thickness mesh well; razor-thin tilted sheets
    //     may have gaps. This is a property of cell-vs-corner data and
    //     cannot be fully fixed without corner-based sign data.
    //
    // Honors the existing dc_* settings (require_free, normal_dot,
    // max_edge_factor, min_area_factor, triangle_normal_dot).

    MeshData build_grid_edge_dc_mesh(int min_last_hit_scan = -1,
                                     int max_last_hit_scan = -1) const {
        MeshData mesh;
        auto recs = build_vertex_table();
        mesh.verts.reserve(recs.size());

        std::unordered_map<VoxKey, int, VoxHash> idx_by_key;
        idx_by_key.reserve(recs.size() * 2 + 1);
        for (const auto& r : recs) {
            if (!dc_vertex_ok(r, min_last_hit_scan, max_last_hit_scan)) continue;
            int ni = (int)mesh.verts.size();
            mesh.verts.push_back(r);
            idx_by_key[r.key] = ni;
        }
        if (mesh.verts.size() < 4) return mesh;

        std::unordered_set<FaceKey, FaceKeyHash> face_set;
        face_set.reserve(mesh.verts.size() * 3);

        auto idx_of = [&](const VoxKey& k) -> int {
            auto it = idx_by_key.find(k);
            return it == idx_by_key.end() ? -1 : it->second;
        };

        auto cell_has_free_along_axis = [&](const VoxKey& k, int axis) -> bool {
            for (int sign : {-1, +1}) {
                auto it = cells.find(add_key_axis(k, axis, sign));
                if (it != cells.end() && it->second.label(s) == VoxelCell::Label::FREE)
                    return true;
            }
            return false;
        };

        for (const auto& [k, base_idx] : idx_by_key) {
            (void)base_idx;
            for (int axis = 0; axis < 3; axis++) {
                int u_axis = (axis + 1) % 3;
                int v_axis = (axis + 2) % 3;

                // Anchor: grid edge in direction `axis` at high-(u,v) corner.
                // Surrounding 4 cells, with k as lex-min in the (u,v) plane:
                VoxKey c00 = k;
                VoxKey c10 = add_key_axis(k, u_axis, 1);
                VoxKey c01 = add_key_axis(k, v_axis, 1);
                VoxKey c11 = add_key2(k, u_axis, 1, v_axis, 1);

                int i00 = idx_of(c00);
                int i10 = idx_of(c10);
                int i01 = idx_of(c01);
                int i11 = idx_of(c11);
                if (i00 < 0 || i10 < 0 || i01 < 0 || i11 < 0) continue;

                // Free-space support: any of the 4 cells must have a FREE
                // neighbor along the edge direction. Indicates the edge
                // genuinely sits on the surface boundary, not interior to a
                // thick observation region.
                if (s.dc_require_free) {
                    bool has_free = cell_has_free_along_axis(c00, axis)
                                 || cell_has_free_along_axis(c10, axis)
                                 || cell_has_free_along_axis(c01, axis)
                                 || cell_has_free_along_axis(c11, axis);
                    if (!has_free) continue;
                }

                // Sign-align the 4 vertex normals to a reference, then
                // average. Reference is n0 (cell c00); if c00's normal is
                // locally flipped, the average still reflects local
                // orientation (which the orientation step in
                // add_oriented_triangle handles via expected_normal).
                Vec3 n0 = mesh.verts[i00].normal;
                Vec3 n1 = mesh.verts[i10].normal;
                Vec3 n2 = mesh.verts[i01].normal;
                Vec3 n3 = mesh.verts[i11].normal;
                if (!normalized_or_zero(n0) || !normalized_or_zero(n1)
                 || !normalized_or_zero(n2) || !normalized_or_zero(n3)) continue;
                if (n1.dot(n0) < 0.0) n1 = -n1;
                if (n2.dot(n0) < 0.0) n2 = -n2;
                if (n3.dot(n0) < 0.0) n3 = -n3;
                Vec3 expected = n0 + n1 + n2 + n3;
                if (!normalized_or_zero(expected)) continue;

                // Sign-invariant agreement gate on each of the 4 normals.
                auto n_agrees = [&](const Vec3& n) {
                    return std::abs(n.dot(expected)) >= s.dc_normal_dot;
                };
                if (!n_agrees(n0) || !n_agrees(n1)
                 || !n_agrees(n2) || !n_agrees(n3)) continue;

                // Split the quad by the shorter 3D diagonal to reduce folds
                // on curved patches.
                double d_a = (mesh.verts[i00].position - mesh.verts[i11].position).squaredNorm();
                double d_b = (mesh.verts[i10].position - mesh.verts[i01].position).squaredNorm();
                if (d_a <= d_b) {
                    add_oriented_triangle(mesh, face_set, i00, i10, i11, expected);
                    add_oriented_triangle(mesh, face_set, i00, i11, i01, expected);
                } else {
                    add_oriented_triangle(mesh, face_set, i00, i10, i01, expected);
                    add_oriented_triangle(mesh, face_set, i10, i11, i01, expected);
                }
            }
        }
        return mesh;
    }



    // ---- Phase 3D corner-derived dual contouring ---- //
    //
    // Corner-DC uses a derived signed value at grid corners instead of the
    // brittle cell-only 4-surface-cell test. Surface cells contribute signed
    // distance to their QEM plane; FREE cells contribute positive outside
    // evidence; UNKNOWN contributes nothing. A grid edge crosses the surface
    // when its two endpoint corner signs differ. If four surrounding QEM
    // vertices are available we emit a quad; if exactly three are available
    // we emit one triangle as a conservative partial-coverage recall boost.

    struct CornerSign {
        double signed_sum = 0.0;
        double evidence_weight = 0.0;
        bool defined() const { return evidence_weight > 1e-9; }
        int sign(double eps = 0.0) const {
            if (!defined()) return 0;
            if (signed_sum >  eps) return +1;
            if (signed_sum < -eps) return -1;
            return 0;
        }
    };

    CornerSign compute_corner_sign(
        const VoxKey& corner,
        const MeshData& mesh,
        const std::unordered_map<VoxKey, int, VoxHash>& vinfo_idx_by_cell) const
    {
        CornerSign info;
        Vec3 p(corner.i * s.voxel_size,
               corner.j * s.voxel_size,
               corner.k * s.voxel_size);

        for (int dx = 0; dx <= 1; dx++) {
            for (int dy = 0; dy <= 1; dy++) {
                for (int dz = 0; dz <= 1; dz++) {
                    VoxKey ck{(int32_t)(corner.i - dx),
                              (int32_t)(corner.j - dy),
                              (int32_t)(corner.k - dz)};
                    auto cit = cells.find(ck);
                    if (cit == cells.end()) continue;
                    const VoxelCell& cell = cit->second;
                    VoxelCell::Label lbl = cell.label(s);

                    if (lbl == VoxelCell::Label::SURFACE) {
                        auto vit = vinfo_idx_by_cell.find(ck);
                        if (vit == vinfo_idx_by_cell.end()) continue;
                        const VertexRecord& vr = mesh.verts[vit->second];
                        Vec3 n = vr.normal;
                        if (!normalized_or_zero(n)) continue;

                        // Avoid letting mixed/noisy normals dominate the sign.
                        double sr = std::sqrt(std::max(0.0, vr.residual));
                        double conf = std::clamp(vr.normal_consistency, 0.0, 1.0);
                        if (!std::isfinite(sr) || sr > std::max(0.08, 2.0 * s.cdp_max_sqrt_residual))
                            conf *= 0.25;
                        if (vr.scan_boundary_ratio > 0.95) conf *= 0.5;
                        if (conf < 0.10) continue;

                        double sd = (p - vr.position).dot(n);
                        double w = cell.hit_weight * conf;
                        info.signed_sum += sd * w;
                        info.evidence_weight += w;
                    } else if (lbl == VoxelCell::Label::FREE) {
                        info.signed_sum += s.voxel_size * cell.miss_weight;
                        info.evidence_weight += cell.miss_weight;
                    } else if (hierarchical_scaffold_enabled() &&
                               s.scaffold_inherited_corner_sign &&
                               cell.has_inherited()) {
                        // Re-attached inherited corner-sign branch: inherited
                        // QEM priors are not occupancy truth, but they can
                        // provide weak surface sign evidence inside validated
                        // scaffold support. This lets isolated sparse scaffold
                        // regions mesh, while the low weight + EOGM/residual/
                        // decay gates keep them weaker than real SURFACE/FREE.
                        if (s.enable_eogm &&
                            (cell.eogm_bel_free() > s.scaffold_max_bel_free ||
                             cell.eogm_conflict_mass() > s.scaffold_max_conflict))
                            continue;

                        const double decay = cell.inherited_decay_scale();
                        if (decay < s.scaffold_inherited_min_decay_scale)
                            continue;

                        auto vit = vinfo_idx_by_cell.find(ck);
                        if (vit == vinfo_idx_by_cell.end()) continue;
                        const VertexRecord& vr = mesh.verts[vit->second];
                        Vec3 n = vr.normal;
                        if (!normalized_or_zero(n)) continue;

                        const double sr = std::sqrt(std::max(0.0, vr.residual));
                        if (s.scaffold_inherited_sign_max_sqrt_residual > 0.0 &&
                            (!std::isfinite(sr) || sr > s.scaffold_inherited_sign_max_sqrt_residual))
                            continue;

                        double conf = std::clamp(vr.normal_consistency, 0.0, 1.0);
                        if (vr.scan_boundary_ratio > 0.95) conf *= 0.5;
                        if (conf < 0.10) continue;

                        double sd = (p - vr.position).dot(n);
                        double w = s.scaffold_inherited_sign_weight *
                                   decay * cell.weight_inherited * conf;
                        if (w <= 1e-12) continue;
                        info.signed_sum += sd * w;
                        info.evidence_weight += w;
                    }
                }
            }
        }
        return info;
    }

    MeshData build_corner_dc_mesh(int min_last_hit_scan = -1,
                                  int max_last_hit_scan = -1) const {
        MeshData mesh;
        auto recs = build_vertex_table();
        mesh.verts.reserve(recs.size());

        std::unordered_map<VoxKey, int, VoxHash> vinfo_idx_by_cell;
        vinfo_idx_by_cell.reserve(recs.size() * 2 + 1);
        for (const auto& r : recs) {
            if (!dc_vertex_ok(r, min_last_hit_scan, max_last_hit_scan)) continue;
            int ni = (int)mesh.verts.size();
            mesh.verts.push_back(r);
            vinfo_idx_by_cell[r.key] = ni;
        }
        if (mesh.verts.size() < 3) return mesh;

        std::unordered_map<VoxKey, CornerSign, VoxHash> corner_cache;
        corner_cache.reserve(cells.size() / 8 + mesh.verts.size() * 2 + 1);
        auto corner_sign_of = [&](const VoxKey& c) -> CornerSign {
            auto it = corner_cache.find(c);
            if (it != corner_cache.end()) return it->second;
            CornerSign cs = compute_corner_sign(c, mesh, vinfo_idx_by_cell);
            corner_cache[c] = cs;
            return cs;
        };

        auto idx_of = [&](const VoxKey& k) -> int {
            auto it = vinfo_idx_by_cell.find(k);
            return (it == vinfo_idx_by_cell.end()) ? -1 : it->second;
        };

        std::unordered_set<FaceKey, FaceKeyHash> face_set;
        face_set.reserve(mesh.verts.size() * 3);

        // Active edge anchors: process edges around surface cells and their
        // immediate free neighbors, instead of scanning all ray-carved cells.
        // This avoids O(number_of_free_cells) meshing on large maps.
        std::unordered_set<VoxKey, VoxHash> anchors;
        anchors.reserve(vinfo_idx_by_cell.size() * 8 + 1);
        for (const auto& kv : vinfo_idx_by_cell) {
            const VoxKey& k = kv.first;
            // Any of the 8 cells around a surface voxel can be the lex-min
            // anchor of a grid edge whose 4-cell ring includes that voxel.
            for (int dx = -1; dx <= 0; dx++)
                for (int dy = -1; dy <= 0; dy++)
                    for (int dz = -1; dz <= 0; dz++)
                        anchors.insert(VoxKey{(int32_t)(k.i + dx),
                                              (int32_t)(k.j + dy),
                                              (int32_t)(k.k + dz)});
        }

        for (const VoxKey& k : anchors) {
            for (int axis = 0; axis < 3; axis++) {
                int u_axis = (axis + 1) % 3;
                int v_axis = (axis + 2) % 3;

                VoxKey corner_A = add_key2(k, u_axis, 1, v_axis, 1);
                VoxKey corner_B = add_key_axis(corner_A, axis, 1);

                CornerSign sa = corner_sign_of(corner_A);
                CornerSign sb = corner_sign_of(corner_B);
                if (!sa.defined() || !sb.defined()) continue;
                double eps_a = 0.01 * s.voxel_size * sa.evidence_weight;
                double eps_b = 0.01 * s.voxel_size * sb.evidence_weight;
                int sgn_a = sa.sign(eps_a);
                int sgn_b = sb.sign(eps_b);
                if (sgn_a == 0 || sgn_b == 0) continue;
                if (sgn_a == sgn_b) continue;

                VoxKey c00 = k;
                VoxKey c10 = add_key_axis(k, u_axis, 1);
                VoxKey c11 = add_key2(k, u_axis, 1, v_axis, 1);
                VoxKey c01 = add_key_axis(k, v_axis, 1);

                int i00 = idx_of(c00);
                int i10 = idx_of(c10);
                int i11 = idx_of(c11);
                int i01 = idx_of(c01);
                int n_have = (i00>=0) + (i10>=0) + (i11>=0) + (i01>=0);
                if (n_have < 3) continue;

                int axis_sign = (sgn_a < 0) ? +1 : -1;
                Vec3 expected = axis_vec(axis, axis_sign);

                if (n_have == 4) {
                    double d_a = (mesh.verts[i00].position - mesh.verts[i11].position).squaredNorm();
                    double d_b = (mesh.verts[i10].position - mesh.verts[i01].position).squaredNorm();
                    if (d_a <= d_b) {
                        add_oriented_triangle(mesh, face_set, i00, i10, i11, expected);
                        add_oriented_triangle(mesh, face_set, i00, i11, i01, expected);
                    } else {
                        add_oriented_triangle(mesh, face_set, i00, i10, i01, expected);
                        add_oriented_triangle(mesh, face_set, i10, i11, i01, expected);
                    }
                } else {
                    int idxs[4] = {i00, i10, i11, i01};
                    int tri[3]; int n = 0;
                    for (int ii = 0; ii < 4; ii++) if (idxs[ii] >= 0) tri[n++] = idxs[ii];
                    add_oriented_triangle(mesh, face_set, tri[0], tri[1], tri[2], expected);
                }
            }
        }
        return mesh;
    }

    // ---- Phase 3E corner_dc_plus: boundary-guided adaptive QEM grow ---- //

    static double point_segment_distance(const Vec3& p, const Vec3& a, const Vec3& b) {
        Vec3 ab = b - a;
        double d2 = ab.squaredNorm();
        if (d2 < 1e-18) return (p - a).norm();
        double t = std::clamp((p - a).dot(ab) / d2, 0.0, 1.0);
        return (p - (a + t * ab)).norm();
    }

    double cdp_adaptive_radius(const VertexRecord& r) const {
        if (!std::isfinite(r.residual)) return 0.0;
        double sr = std::sqrt(std::max(0.0, r.residual));
        bool weak = (r.confidence_tier != 0);
        if (!weak) {
            if (r.hit_count < s.mesh_min_hit_count) return 0.0;
            if (sr > s.cdp_max_sqrt_residual) return 0.0;
            if (r.normal_consistency < s.cdp_min_consistency) return 0.0;
        } else {
            // Inherited scaffold vertices may have zero real hits. They are
            // allowed as weak CDP targets only if their inherited QEM residual
            // and normal consistency pass the same conservative gates.
            if (r.confidence_tier != 6 && r.hit_count < 1) return 0.0;
            if (sr > std::max(s.cdp_max_sqrt_residual, s.seed_l1_max_sqrt_residual)) return 0.0;
            if (r.normal_consistency < std::min(s.cdp_min_consistency, s.seed_l1_min_consistency)) return 0.0;
        }
        if (r.scan_boundary_ratio > s.cdp_max_boundary_ratio) return 0.0;

        bool flat = std::strcmp(r.kind, "flat") == 0;
        double factor = 0.0;
        if (flat && (weak || r.normal_consistency >= 0.94)) factor = s.cdp_flat_radius_factor;
        else if (s.cdp_allow_curve) factor = s.cdp_curve_radius_factor;
        else return 0.0;
        factor = std::min(factor, s.cdp_max_radius_factor);
        return factor * s.voxel_size;
    }

    bool cdp_normals_compatible(const VertexRecord& a,
                                const VertexRecord& b,
                                const VertexRecord& c) const {
        Vec3 na = a.normal, nb = b.normal, nc = c.normal;
        if (!normalized_or_zero(na) || !normalized_or_zero(nb) || !normalized_or_zero(nc)) return false;
        bool all_flat = std::strcmp(a.kind, "flat") == 0 &&
                        std::strcmp(b.kind, "flat") == 0 &&
                        std::strcmp(c.kind, "flat") == 0;
        double thr = all_flat ? s.cdp_planar_normal_dot : s.cdp_curve_normal_dot;
        if (!all_flat && !s.cdp_allow_curve) return false;
        return std::abs(na.dot(nb)) >= thr &&
               std::abs(na.dot(nc)) >= thr &&
               std::abs(nb.dot(nc)) >= thr;
    }

    bool cdp_free_samples_ok(const MeshData& mesh, int a, int b, int c) const {
        if (!s.cdp_reject_free_samples) return true;
        const Vec3& pa = mesh.verts[a].position;
        const Vec3& pb = mesh.verts[b].position;
        const Vec3& pc = mesh.verts[c].position;
        Vec3 samples[7] = {
            (pa + pb + pc) / 3.0,
            0.5 * (pa + pb),
            0.5 * (pb + pc),
            0.5 * (pc + pa),
            0.60 * pa + 0.20 * pb + 0.20 * pc,
            0.20 * pa + 0.60 * pb + 0.20 * pc,
            0.20 * pa + 0.20 * pb + 0.60 * pc
        };
        for (const Vec3& x : samples) {
            auto it = cells.find(voxel_index(x));
            if (it != cells.end() && it->second.label(s) == VoxelCell::Label::FREE)
                return false;
        }
        return true;
    }

    bool eogm_samples_ok_for_triangle(const Vec3& pa, const Vec3& pb, const Vec3& pc,
                                      double max_free, double max_conflict,
                                      double min_plaus_surface) const {
        if (!s.enable_eogm) return true;
        Vec3 samples[7] = {
            (pa + pb + pc) / 3.0,
            0.5 * (pa + pb), 0.5 * (pb + pc), 0.5 * (pc + pa),
            0.60 * pa + 0.20 * pb + 0.20 * pc,
            0.20 * pa + 0.60 * pb + 0.20 * pc,
            0.20 * pa + 0.20 * pb + 0.60 * pc
        };
        int seen = 0;
        double plaus_sum = 0.0;
        for (const Vec3& x : samples) {
            auto it = cells.find(voxel_index(x));
            if (it == cells.end()) continue; // unknown is allowed, but does not add support
            const VoxelCell& vc = it->second;
            if (vc.eogm_bel_free() > max_free) return false;
            if (vc.eogm_conflict_mass() > max_conflict) return false;
            if (vc.label(s) == VoxelCell::Label::FREE) return false;
            plaus_sum += vc.eogm_plaus_surface();
            seen++;
        }
        if (seen == 0) return true;
        return (plaus_sum / (double)seen) >= min_plaus_surface;
    }

    bool cdp_merged_qem_ok(const MeshData& mesh, int a, int b, int c) const {
        int ids[3] = {a, b, c};
        Mat3 A = Mat3::Zero();
        Vec3 bb = Vec3::Zero();
        double cc = 0.0;
        double wsum = 0.0;
        Vec3 anchor = Vec3::Zero();
        for (int id : ids) {
            const VertexRecord& r = mesh.verts[id];
            auto it = cells.find(r.key);
            if (it == cells.end()) return false;
            const VoxelCell& vc = it->second;
            A.noalias() += vc.A_eff();
            bb.noalias() += vc.b_eff();
            cc += vc.c_eff();
            wsum += vc.weight_eff();
            anchor += r.position;
        }
        if (wsum < 1e-12) return false;
        anchor /= 3.0;

        Mat3 Areg = A + s.lambda_p * Mat3::Identity();
        Vec3 x = Areg.ldlt().solve(bb + s.lambda_p * anchor);
        double f = x.transpose() * A * x;
        f += cc;
        f -= 2.0 * bb.dot(x);
        double sqrt_res = std::sqrt(std::max(0.0, f / wsum));
        if (sqrt_res > s.cdp_max_merged_sqrt_residual) return false;

        Eigen::SelfAdjointEigenSolver<Mat3> eig(A);
        Vec3 n = eig.eigenvectors().col(2); // largest eigenvalue direction, i.e. dominant plane normal
        if (!normalized_or_zero(n)) return false;
        double max_dist = s.cdp_max_point_plane_dist_factor * s.voxel_size;
        for (int id : ids) {
            double pd = std::abs((mesh.verts[id].position - x).dot(n));
            if (pd > max_dist) return false;
        }
        return true;
    }

    bool cdp_triangle_ok(const MeshData& mesh, int a, int b, int c) const {
        if (a == b || a == c || b == c) return false;
        const VertexRecord& va = mesh.verts[a];
        const VertexRecord& vb = mesh.verts[b];
        const VertexRecord& vc = mesh.verts[c];
        if (s.cdp_weak_triangles_require_confirmed) {
            bool has_weak = (va.confidence_tier != 0) || (vb.confidence_tier != 0) || (vc.confidence_tier != 0);
            bool has_confirmed = (va.confidence_tier == 0) || (vb.confidence_tier == 0) || (vc.confidence_tier == 0);
            if (has_weak && !has_confirmed) return false;
        }
        if (!cdp_normals_compatible(va, vb, vc)) return false;

        double max_edge = s.cdp_max_edge_factor * s.voxel_size;
        double eab = (va.position - vb.position).norm();
        double ebc = (vb.position - vc.position).norm();
        double eca = (vc.position - va.position).norm();
        if (eab > max_edge || ebc > max_edge || eca > max_edge) return false;

        Vec3 tri_n = (vb.position - va.position).cross(vc.position - va.position);
        double twice_area = tri_n.norm();
        if (twice_area < 2.0 * s.dc_min_area_factor * s.voxel_size * s.voxel_size) return false;
        tri_n /= twice_area;

        Vec3 avg = va.normal;
        Vec3 nb = vb.normal;
        Vec3 nc = vc.normal;
        if (!normalized_or_zero(avg) || !normalized_or_zero(nb) || !normalized_or_zero(nc)) return false;
        if (nb.dot(avg) < 0.0) nb = -nb;
        if (nc.dot(avg) < 0.0) nc = -nc;
        avg += nb + nc;
        if (!normalized_or_zero(avg)) return false;
        if (std::abs(tri_n.dot(avg)) < s.dc_triangle_normal_dot) return false;

        if (!cdp_free_samples_ok(mesh, a, b, c)) return false;
        if (s.eogm_mesh_gate &&
            !eogm_samples_ok_for_triangle(va.position, vb.position, vc.position,
                                          s.eogm_mesh_max_bel_free,
                                          s.eogm_mesh_max_conflict,
                                          s.eogm_seed_min_plaus_surface)) return false;
        if (!cdp_merged_qem_ok(mesh, a, b, c)) return false;
        return true;
    }

    void add_oriented_triangle_cdp(MeshData& mesh,
                                   std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                                   int a, int b, int c) const {
        if (!cdp_triangle_ok(mesh, a, b, c)) return;
        int fa = a, fb = b, fc = c;
        Vec3 tri_n = (mesh.verts[fb].position - mesh.verts[fa].position)
                   .cross(mesh.verts[fc].position - mesh.verts[fa].position);
        Vec3 en = mesh.verts[fa].normal;
        Vec3 nb = mesh.verts[fb].normal;
        Vec3 nc = mesh.verts[fc].normal;
        if (normalized_or_zero(en) && normalized_or_zero(nb) && normalized_or_zero(nc)) {
            if (nb.dot(en) < 0.0) nb = -nb;
            if (nc.dot(en) < 0.0) nc = -nc;
            en += nb + nc;
            if (normalized_or_zero(en) && tri_n.dot(en) < 0.0) std::swap(fb, fc);
        }
        FaceKey fk = sorted_face_key(fa, fb, fc);
        if (face_set.insert(fk).second) mesh.faces.push_back({fa, fb, fc});
    }

    bool merged_qem_plane_for_ids(const MeshData& mesh, int a, int b, int c,
                                  Vec3& out_point, Vec3& out_normal, double& out_sqrt_res,
                                  double max_sqrt_residual, double max_plane_dist_factor) const {
        int ids[3] = {a, b, c};
        Mat3 A = Mat3::Zero();
        Vec3 bb = Vec3::Zero();
        double cc = 0.0;
        double wsum = 0.0;
        Vec3 anchor = Vec3::Zero();
        for (int id : ids) {
            if (id < 0 || id >= (int)mesh.verts.size()) return false;
            const VertexRecord& r = mesh.verts[id];
            auto it = cells.find(r.key);
            if (it == cells.end()) return false;
            const VoxelCell& vc = it->second;
            A.noalias() += vc.A_eff();
            bb.noalias() += vc.b_eff();
            cc += vc.c_eff();
            wsum += vc.weight_eff();
            anchor += r.position;
        }
        if (wsum < 1e-12) return false;
        anchor /= 3.0;
        Mat3 Areg = A + s.lambda_p * Mat3::Identity();
        Vec3 x = Areg.ldlt().solve(bb + s.lambda_p * anchor);
        double f = x.transpose() * A * x;
        f += cc;
        f -= 2.0 * bb.dot(x);
        double sqrt_res = std::sqrt(std::max(0.0, f / wsum));
        if (sqrt_res > max_sqrt_residual) return false;
        Eigen::SelfAdjointEigenSolver<Mat3> eig(A);
        Vec3 n = eig.eigenvectors().col(2);
        if (!normalized_or_zero(n)) return false;
        double max_dist = max_plane_dist_factor * s.voxel_size;
        for (int id : ids) {
            double pd = std::abs((mesh.verts[id].position - x).dot(n));
            if (pd > max_dist) return false;
        }
        out_point = x;
        out_normal = n;
        out_sqrt_res = sqrt_res;
        return true;
    }

    bool generated_small_triangle_ok(const MeshData& mesh, int a, int b, int gidx) const {
        const VertexRecord& va = mesh.verts[a];
        const VertexRecord& vb = mesh.verts[b];
        const VertexRecord& vg = mesh.verts[gidx];
        double max_edge = s.cdp_max_edge_factor * s.voxel_size;
        if ((va.position - vb.position).norm() > max_edge) return false;
        if ((vb.position - vg.position).norm() > max_edge) return false;
        if ((vg.position - va.position).norm() > max_edge) return false;
        Vec3 tri_n = (vb.position - va.position).cross(vg.position - va.position);
        double twice_area = tri_n.norm();
        if (twice_area < 2.0 * s.dc_min_area_factor * s.voxel_size * s.voxel_size) return false;
        tri_n /= twice_area;
        Vec3 avg = va.normal;
        Vec3 nb = vb.normal;
        Vec3 ng = vg.normal;
        if (!normalized_or_zero(avg) || !normalized_or_zero(nb) || !normalized_or_zero(ng)) return false;
        if (nb.dot(avg) < 0.0) nb = -nb;
        if (ng.dot(avg) < 0.0) ng = -ng;
        avg += nb + ng;
        if (!normalized_or_zero(avg)) return false;
        if (std::abs(tri_n.dot(avg)) < s.dc_triangle_normal_dot) return false;
        return eogm_samples_ok_for_triangle(va.position, vb.position, vg.position,
                                            s.gen_max_bel_free, s.gen_max_conflict,
                                            s.gen_min_support_plaus_surface);
    }

    void add_oriented_triangle_generated(MeshData& mesh,
                                         std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                                         int a, int b, int c) const {
        if (a == b || a == c || b == c) return;
        int fa = a, fb = b, fc = c;
        Vec3 tri_n = (mesh.verts[fb].position - mesh.verts[fa].position)
                   .cross(mesh.verts[fc].position - mesh.verts[fa].position);
        Vec3 en = mesh.verts[fa].normal;
        Vec3 nb = mesh.verts[fb].normal;
        Vec3 nc = mesh.verts[fc].normal;
        if (normalized_or_zero(en) && normalized_or_zero(nb) && normalized_or_zero(nc)) {
            if (nb.dot(en) < 0.0) nb = -nb;
            if (nc.dot(en) < 0.0) nc = -nc;
            en += nb + nc;
            if (normalized_or_zero(en) && tri_n.dot(en) < 0.0) std::swap(fb, fc);
        }
        FaceKey fk = sorted_face_key(fa, fb, fc);
        if (face_set.insert(fk).second) mesh.faces.push_back({fa, fb, fc});
    }

    size_t apply_eogm_generative_fill_pass(MeshData& mesh,
                                           std::unordered_set<FaceKey, FaceKeyHash>& face_set) const {
        if (!s.enable_eogm_generative_fill || mesh.verts.size() < 3) return 0;
        const int N0 = (int)mesh.verts.size();
        std::vector<int> incident(N0, 0);
        std::unordered_map<uint64_t, int> edge_count;
        edge_count.reserve(mesh.faces.size() * 3 + 1);
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N0 || f.b >= N0 || f.c >= N0) continue;
            incident[f.a]++; incident[f.b]++; incident[f.c]++;
            edge_count[edge_key(f.a, f.b)]++;
            edge_count[edge_key(f.b, f.c)]++;
            edge_count[edge_key(f.c, f.a)]++;
        }

        struct BEdge { int a, b; };
        std::vector<BEdge> boundary_edges;
        boundary_edges.reserve(edge_count.size()/4 + 1);
        for (const auto& kv : edge_count) {
            if (kv.second != 1) continue;
            int a = (int)(kv.first >> 32);
            int b = (int)(kv.first & 0xffffffffu);
            if (a >= 0 && b >= 0 && a < N0 && b < N0) boundary_edges.push_back({a,b});
        }
        if (boundary_edges.empty()) return 0;

        double r_max = s.gen_max_edge_factor * s.voxel_size;
        double r2 = r_max * r_max;
        size_t added_patches = 0;
        for (const auto& be : boundary_edges) {
            if ((int)added_patches >= s.gen_max_faces) break;
            const Vec3& pa = mesh.verts[be.a].position;
            const Vec3& pb = mesh.verts[be.b].position;
            int best = -1;
            double best_score = 1e100;
            for (int q = 0; q < N0; q++) {
                if (q == be.a || q == be.b) continue;
                if (incident[q] >= s.cdp_min_incident_faces) continue;
                const Vec3& pq = mesh.verts[q].position;
                if ((pq - pa).squaredNorm() > r2 || (pq - pb).squaredNorm() > r2) continue;
                if (mesh.verts[be.a].confidence_tier != 0 &&
                    mesh.verts[be.b].confidence_tier != 0 &&
                    mesh.verts[q].confidence_tier != 0 &&
                    s.gen_min_support_confirmed > 0) continue;
                if (!cdp_normals_compatible(mesh.verts[be.a], mesh.verts[be.b], mesh.verts[q])) continue;
                if (!eogm_samples_ok_for_triangle(pa, pb, pq, s.gen_max_bel_free,
                                                  s.gen_max_conflict,
                                                  s.gen_min_support_plaus_surface)) continue;
                Vec3 plane_p, plane_n; double sqrt_res = 0.0;
                if (!merged_qem_plane_for_ids(mesh, be.a, be.b, q, plane_p, plane_n, sqrt_res,
                                              s.gen_max_merged_sqrt_residual,
                                              s.gen_max_point_plane_dist_factor)) continue;
                double score = point_segment_distance(pq, pa, pb) + 2.0 * sqrt_res;
                if (score < best_score) { best_score = score; best = q; }
            }
            if (best < 0) continue;

            Vec3 plane_p, plane_n; double sqrt_res = 0.0;
            if (!merged_qem_plane_for_ids(mesh, be.a, be.b, best, plane_p, plane_n, sqrt_res,
                                          s.gen_max_merged_sqrt_residual,
                                          s.gen_max_point_plane_dist_factor)) continue;
            Vec3 centroid = (mesh.verts[be.a].position + mesh.verts[be.b].position + mesh.verts[best].position) / 3.0;
            Vec3 gpos = centroid - ((centroid - plane_p).dot(plane_n)) * plane_n;
            VertexRecord gr = mesh.verts[be.a];
            gr.key = voxel_index(gpos);
            gr.position = gpos;
            gr.normal = plane_n;
            gr.kind = "generated";
            gr.residual = sqrt_res * sqrt_res;
            gr.normal_consistency = std::min({mesh.verts[be.a].normal_consistency, mesh.verts[be.b].normal_consistency, mesh.verts[best].normal_consistency});
            gr.scan_boundary_ratio = 0.0;
            gr.weight_sum = 0.0;
            gr.hit_count = 0;
            gr.miss_count = 0;
            gr.last_hit_scan = std::max({mesh.verts[be.a].last_hit_scan, mesh.verts[be.b].last_hit_scan, mesh.verts[best].last_hit_scan});
            gr.eogm_bel_surface = 0.0;
            gr.eogm_bel_free = 0.0;
            gr.eogm_plaus_surface = 1.0;
            gr.eogm_unknown = 1.0;
            gr.eogm_conflict = 0.0;
            gr.prob_plane_sigma = 0.0;
            gr.prob_plane_radius = 0.0;
            gr.prob_plane_min_eigen = 0.0;
            gr.prob_plane_mid_eigen = 0.0;
            gr.prob_plane_max_eigen = 0.0;
            gr.prob_plane_normal_cov_trace = 0.0;
            gr.prob_plane_points = 0;
            gr.prob_plane_valid = 0;
            gr.prob_plane_is_planar = 0;
            gr.confidence_tier = 4;
            int gidx = (int)mesh.verts.size();
            mesh.verts.push_back(gr);

            bool ok1 = generated_small_triangle_ok(mesh, be.a, be.b, gidx);
            bool ok2 = generated_small_triangle_ok(mesh, be.b, best, gidx);
            bool ok3 = generated_small_triangle_ok(mesh, best, be.a, gidx);
            if (!(ok1 && ok2 && ok3)) {
                mesh.verts.pop_back();
                continue;
            }
            add_oriented_triangle_generated(mesh, face_set, be.a, be.b, gidx);
            add_oriented_triangle_generated(mesh, face_set, be.b, best, gidx);
            add_oriented_triangle_generated(mesh, face_set, best, be.a, gidx);
            added_patches++;
        }
        if (added_patches > 0) {
            std::printf("  [eogm_gen_fill] patches=%zu generated_vertices=%zu faces=%zu\n",
                        added_patches, mesh.verts.size() - (size_t)N0, mesh.faces.size());
        }
        return added_patches;
    }


    // ---- Hierarchical mesh-only scaffold fill ----------------------------- //
    // This is the direct superset of the earlier single-level planar scaffold:
    // it creates provisional tier-5 vertices in L0 child voxels of every valid
    // scaffold parent, then tiles them on the ordinary L0 lattice. Level 1
    // (factor=2) mimics the old 0.20 m -> 0.10 m mesh-only scaffold fill. Higher
    // levels are applied after lower levels and only fill child keys that still
    // have no real/generated/scaffold vertex, so L2/L3 add recall instead of
    // replacing the L1 result.

    bool scaffold_triangle_ok(const MeshData& mesh, int a, int b, int c,
                              const Vec3& expected_normal) const {
        const Vec3& pa = mesh.verts[a].position;
        const Vec3& pb = mesh.verts[b].position;
        const Vec3& pc = mesh.verts[c].position;
        double maxe = s.scaffold_max_edge_factor * s.voxel_size;
        if ((pa - pb).norm() > maxe || (pb - pc).norm() > maxe || (pc - pa).norm() > maxe)
            return false;

        Vec3 tn = (pb - pa).cross(pc - pa);
        double twice_area = tn.norm();
        if (twice_area < 2.0 * s.dc_min_area_factor * s.voxel_size * s.voxel_size) return false;
        tn /= twice_area;

        Vec3 en = expected_normal;
        if (!normalized_or_zero(en)) return false;
        if (std::abs(tn.dot(en)) < s.dc_triangle_normal_dot) return false;

        Vec3 samples[4] = {(pa + pb + pc) / 3.0,
                           0.5 * (pa + pb),
                           0.5 * (pb + pc),
                           0.5 * (pc + pa)};
        for (const Vec3& xs : samples) {
            auto it = cells.find(voxel_index(xs));
            if (it == cells.end()) continue;
            const VoxelCell& vc = it->second;
            if (vc.label(s) == VoxelCell::Label::FREE) return false;
            if (s.enable_eogm) {
                if (vc.eogm_bel_free() > s.scaffold_max_bel_free) return false;
                if (vc.eogm_conflict_mass() > s.scaffold_max_conflict) return false;
            }
        }
        return true;
    }

    void rebuild_mesh_key_index(const MeshData& mesh,
                                std::unordered_map<VoxKey, int, VoxHash>& idx_by_key) const {
        idx_by_key.clear();
        idx_by_key.reserve(mesh.verts.size() * 2 + 1);
        // First pass: real/promoted/inherited vertices win over mesh-only scaffold.
        for (int i = 0; i < (int)mesh.verts.size(); ++i) {
            if (mesh.verts[i].confidence_tier == 4 || mesh.verts[i].confidence_tier == 5) continue;
            if (idx_by_key.find(mesh.verts[i].key) == idx_by_key.end())
                idx_by_key[mesh.verts[i].key] = i;
        }
        // Second pass: existing mesh-only scaffold fills only still-empty keys.
        for (int i = 0; i < (int)mesh.verts.size(); ++i) {
            if (mesh.verts[i].confidence_tier != 5) continue;
            if (idx_by_key.find(mesh.verts[i].key) == idx_by_key.end())
                idx_by_key[mesh.verts[i].key] = i;
        }
    }

    size_t apply_hierarchical_scaffold_fill_pass(
            MeshData& mesh,
            std::unordered_set<FaceKey, FaceKeyHash>& face_set) const {
        if (!hierarchical_scaffold_fill_enabled()) return 0;

        std::unordered_map<uint64_t, int> edge_count;
        edge_count.reserve(mesh.faces.size() * 3 + 1);
        for (const auto& fc : mesh.faces) {
            if (fc.a < 0 || fc.b < 0 || fc.c < 0 ||
                fc.a >= (int)mesh.verts.size() || fc.b >= (int)mesh.verts.size() || fc.c >= (int)mesh.verts.size())
                continue;
            edge_count[edge_key(fc.a, fc.b)]++;
            edge_count[edge_key(fc.b, fc.c)]++;
            edge_count[edge_key(fc.c, fc.a)]++;
        }

        std::unordered_map<VoxKey, int, VoxHash> idx_by_key;
        rebuild_mesh_key_index(mesh, idx_by_key);

        size_t total_added_faces = 0;
        size_t total_added_vertices = 0;
        const int max_level = std::clamp(s.scaffold_max_level, 1, 6);

        for (int level = 1; level <= max_level; ++level) {
            const int factor = scaffold_factor_for_level(level);
            // During an incremental remesh, only aggregate parents overlapping
            // the active region; otherwise aggregate the whole map (full build).
            auto agg = active_region_ ? build_scaffold_aggregates_for_region(level)
                                      : build_scaffold_aggregates(level);
            if (agg.empty()) continue;

            std::unordered_map<VoxKey, int, VoxHash> level_scaffold_idx_by_key;
            std::unordered_map<VoxKey, std::pair<int,int>, VoxHash> axes_by_key;
            level_scaffold_idx_by_key.reserve(agg.size() * 4 + 1);
            axes_by_key.reserve(agg.size() * 4 + 1);

            size_t level_valid_parents = 0;
            size_t level_created_vertices = 0;

            for (const auto& kv : agg) {
                const VoxKey& pk = kv.first;
                const ScaffoldAggregate& a = kv.second;
                Vec3 n, x;
                double d = 0.0;
                if (!scaffold_parent_plane(level, pk, a, n, d, x)) continue;
                level_valid_parents++;

                int depth_axis = 0;
                double best = std::abs(n[0]);
                for (int ax = 1; ax < 3; ++ax) {
                    if (std::abs(n[ax]) > best) { best = std::abs(n[ax]); depth_axis = ax; }
                }
                int u_axis = (depth_axis + 1) % 3;
                int v_axis = (depth_axis + 2) % 3;
                const double reach_scale = 0.5 * s.voxel_size *
                    (std::abs(n[0]) + std::abs(n[1]) + std::abs(n[2]));

                for (int dx = 0; dx < factor; ++dx)
                for (int dy = 0; dy < factor; ++dy)
                for (int dz = 0; dz < factor; ++dz) {
                    VoxKey ck{(int32_t)(pk.i * factor + dx),
                              (int32_t)(pk.j * factor + dy),
                              (int32_t)(pk.k * factor + dz)};

                    if (idx_by_key.find(ck) != idx_by_key.end()) continue; // real/L1 scaffold wins
                    // Keep scaffold fill local during a persistent incremental
                    // remesh: only tile child cells inside the active region.
                    if (active_region_ && active_region_->find(ck) == active_region_->end()) continue;
                    if (!scaffold_child_visibility_ok(ck)) continue;

                    Vec3 ctr = voxel_center(ck);
                    if (s.scaffold_fill_require_plane_crossing && std::abs(n.dot(ctr) + d) > reach_scale)
                        continue;

                    Vec3 p = ctr - (n.dot(ctr) + d) * n;
                    if (s.clamp_to_voxel) {
                        Vec3 lo, hi; voxel_bounds(ck, lo, hi);
                        p = p.cwiseMax(lo).cwiseMin(hi);
                    }

                    VertexRecord gr{};
                    gr.key = ck;
                    gr.position = p;
                    gr.normal = n;
                    gr.kind = "scaffold";
                    gr.residual = 0.0;
                    gr.normal_consistency = 1.0;
                    gr.scan_boundary_ratio = 0.0;
                    gr.eval2_over_eval1 = 0.0;
                    gr.eval3_over_eval1 = 0.0;
                    gr.weight_sum = 0.0;
                    gr.hit_count = 0;
                    gr.miss_count = 0;
                    gr.last_hit_scan = -1;
                    gr.eogm_bel_surface = 0.0;
                    gr.eogm_bel_free = 0.0;
                    gr.eogm_plaus_surface = 1.0;
                    gr.eogm_unknown = 1.0;
                    gr.eogm_conflict = 0.0;
                    gr.prob_plane_sigma = 0.0;
                    gr.prob_plane_radius = 0.0;
                    gr.prob_plane_min_eigen = 0.0;
                    gr.prob_plane_mid_eigen = 0.0;
                    gr.prob_plane_max_eigen = 0.0;
                    gr.prob_plane_normal_cov_trace = 0.0;
                    gr.prob_plane_points = 0;
                    gr.prob_plane_valid = 0;
                    gr.prob_plane_is_planar = 1;
                    gr.confidence_tier = 5;

                    int gi = (int)mesh.verts.size();
                    mesh.verts.push_back(gr);
                    idx_by_key[ck] = gi;
                    level_scaffold_idx_by_key[ck] = gi;
                    axes_by_key[ck] = {u_axis, v_axis};
                    level_created_vertices++;
                }
            }

            if (axes_by_key.empty()) {
                if (level_valid_parents > 0) {
                    std::printf("  [hier_scaffold_fill] L%d factor=%d parents=%zu new_vertices=0 faces=0\n",
                                level, factor, level_valid_parents);
                }
                continue;
            }

            auto resolve = [&](const VoxKey& k) -> int {
                auto it = idx_by_key.find(k);
                return it == idx_by_key.end() ? -1 : it->second;
            };

            auto add_tri = [&](int a, int b, int c, const Vec3& en) -> bool {
                if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) return false;
                if (!triangle_edges_can_accept(edge_count, a, b, c)) return false;
                bool any_scaffold = mesh.verts[a].confidence_tier == 5 ||
                                    mesh.verts[b].confidence_tier == 5 ||
                                    mesh.verts[c].confidence_tier == 5;
                if (!any_scaffold) return false;
                if (!scaffold_triangle_ok(mesh, a, b, c, en)) return false;

                int fa = a, fb = b, fc = c;
                Vec3 tn = (mesh.verts[fb].position - mesh.verts[fa].position)
                        .cross(mesh.verts[fc].position - mesh.verts[fa].position);
                Vec3 e = en;
                if (normalized_or_zero(e) && tn.dot(e) < 0.0) std::swap(fb, fc);
                FaceKey fk = sorted_face_key(fa, fb, fc);
                if (!face_set.insert(fk).second) return false;
                mesh.faces.push_back({fa, fb, fc});
                register_triangle_edges(edge_count, fa, fb, fc);
                return true;
            };

            size_t level_added_faces = 0;
            for (const auto& kv : axes_by_key) {
                const VoxKey& ck = kv.first;
                int u_axis = kv.second.first;
                int v_axis = kv.second.second;
                VoxKey k00 = ck;
                VoxKey k10 = add_key_axis(ck, u_axis, 1);
                VoxKey k01 = add_key_axis(ck, v_axis, 1);
                VoxKey k11 = add_key2(ck, u_axis, 1, v_axis, 1);
                int i00 = resolve(k00), i10 = resolve(k10), i01 = resolve(k01), i11 = resolve(k11);
                if (i00 < 0 || i10 < 0 || i01 < 0 || i11 < 0) continue;

                Vec3 en = mesh.verts[i00].normal;
                double da = (mesh.verts[i00].position - mesh.verts[i11].position).squaredNorm();
                double db = (mesh.verts[i10].position - mesh.verts[i01].position).squaredNorm();
                if (da <= db) {
                    if (add_tri(i00, i10, i11, en)) level_added_faces++;
                    if (add_tri(i00, i11, i01, en)) level_added_faces++;
                } else {
                    if (add_tri(i00, i10, i01, en)) level_added_faces++;
                    if (add_tri(i10, i11, i01, en)) level_added_faces++;
                }
            }

            total_added_faces += level_added_faces;
            total_added_vertices += level_created_vertices;
            std::printf("  [hier_scaffold_fill] L%d factor=%d parents=%zu new_vertices=%zu faces=%zu\n",
                        level, factor, level_valid_parents, level_created_vertices, level_added_faces);
        }

        // Remove mesh-only scaffold vertices that failed to receive faces. Keep
        // real, weak, inherited, and EOGM-generated vertices unchanged.
        std::vector<int> incident(mesh.verts.size(), 0);
        for (const auto& fc : mesh.faces) {
            if (fc.a >= 0 && fc.a < (int)incident.size()) incident[fc.a]++;
            if (fc.b >= 0 && fc.b < (int)incident.size()) incident[fc.b]++;
            if (fc.c >= 0 && fc.c < (int)incident.size()) incident[fc.c]++;
        }
        std::vector<int> remap(mesh.verts.size(), -1);
        std::vector<VertexRecord> kept;
        kept.reserve(mesh.verts.size());
        for (int i = 0; i < (int)mesh.verts.size(); ++i) {
            if (mesh.verts[i].confidence_tier == 5 && incident[i] == 0) continue;
            remap[i] = (int)kept.size();
            kept.push_back(mesh.verts[i]);
        }
        if (kept.size() != mesh.verts.size()) {
            for (auto& fc : mesh.faces) {
                fc.a = remap[fc.a];
                fc.b = remap[fc.b];
                fc.c = remap[fc.c];
            }
            mesh.verts.swap(kept);
        }

        if (total_added_faces > 0 || total_added_vertices > 0) {
            std::printf("  [hier_scaffold_fill] total_new_vertices=%zu total_faces=%zu total_verts=%zu\n",
                        total_added_vertices, total_added_faces, mesh.verts.size());
        }
        return total_added_faces;
    }

    MeshData build_corner_dc_plus_mesh(int min_last_hit_scan = -1,
                                       int max_last_hit_scan = -1) const {
        // Pass A: confirmed-only corner_dc baseline. Weak tiers are appended
        // only after baseline connectivity has been created, so they cannot
        // alter corner signs or ordinary DC quads.
        MeshData mesh = build_corner_dc_mesh(min_last_hit_scan, max_last_hit_scan);

        if (s.cdp_use_weak_vertices) {
            std::unordered_set<VoxKey, VoxHash> present;
            present.reserve(mesh.verts.size() * 2 + 1);
            for (const auto& r : mesh.verts) present.insert(r.key);
            auto all_recs = build_vertex_table();
            for (const auto& r : all_recs) {
                if (r.confidence_tier == 0) continue;
                if (present.find(r.key) != present.end()) continue;
                if (min_last_hit_scan >= 0 && r.last_hit_scan < min_last_hit_scan) continue;
                if (max_last_hit_scan >= 0 && r.last_hit_scan > max_last_hit_scan) continue;
                mesh.verts.push_back(r);
                present.insert(r.key);
            }
        }

        const int N = (int)mesh.verts.size();
        if (N < 3 && !hierarchical_scaffold_fill_enabled()) return mesh;

        std::unordered_set<FaceKey, FaceKeyHash> face_set;
        face_set.reserve(mesh.faces.size() * 2 + 1024);
        for (const auto& f : mesh.faces) face_set.insert(sorted_face_key(f.a, f.b, f.c));

        size_t total_added = 0;
        for (int iter = 0; iter < std::max(1, s.cdp_iters); iter++) {
            std::vector<int> incident(N, 0);
            std::unordered_map<uint64_t, int> edge_count;
            edge_count.reserve(mesh.faces.size() * 3 + 1);
            for (const auto& f : mesh.faces) {
                if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
                incident[f.a]++; incident[f.b]++; incident[f.c]++;
                edge_count[edge_key(f.a, f.b)]++;
                edge_count[edge_key(f.b, f.c)]++;
                edge_count[edge_key(f.c, f.a)]++;
            }

            struct BoundaryEdge { int a, b; double radius; };
            std::vector<BoundaryEdge> boundary_edges;
            boundary_edges.reserve(edge_count.size() / 4 + 1);
            std::vector<char> is_boundary(N, 0);
            for (const auto& kv : edge_count) {
                if (kv.second != 1) continue;
                int a = (int)(kv.first >> 32);
                int b = (int)(kv.first & 0xffffffffu);
                if (a < 0 || b < 0 || a >= N || b >= N) continue;
                double ra = cdp_adaptive_radius(mesh.verts[a]);
                double rb = cdp_adaptive_radius(mesh.verts[b]);
                double r = std::min(ra, rb);
                if (r <= 0.0) continue;
                boundary_edges.push_back({a, b, r});
                is_boundary[a] = is_boundary[b] = 1;
            }
            if (boundary_edges.empty()) break;

            // Spatial hash of weak/unmeshed target QEM vertices.
            std::unordered_map<VoxKey, std::vector<int>, VoxHash> weak_by_key;
            weak_by_key.reserve((size_t)N / 4 + 1);
            size_t n_weak = 0;
            for (int i = 0; i < N; i++) {
                if (incident[i] >= s.cdp_min_incident_faces) continue;
                if (is_boundary[i]) continue; // fill inward/outward from boundary to currently unconnected support
                if (cdp_adaptive_radius(mesh.verts[i]) <= 0.0) continue;
                weak_by_key[mesh.verts[i].key].push_back(i);
                n_weak++;
            }
            if (n_weak == 0) break;

            size_t added_this_iter = 0;
            double max_radius = s.cdp_max_radius_factor * s.voxel_size;
            int max_vox_rad = std::max(1, (int)std::ceil(max_radius / s.voxel_size) + 1);

            for (const auto& be : boundary_edges) {
                const Vec3& pa = mesh.verts[be.a].position;
                const Vec3& pb = mesh.verts[be.b].position;
                Vec3 mid = 0.5 * (pa + pb);
                VoxKey mk = voxel_index(mid);

                struct Cand { double score; int idx; };
                std::vector<Cand> cands;
                for (int dx = -max_vox_rad; dx <= max_vox_rad; dx++) {
                    for (int dy = -max_vox_rad; dy <= max_vox_rad; dy++) {
                        for (int dz = -max_vox_rad; dz <= max_vox_rad; dz++) {
                            VoxKey qk{(int32_t)(mk.i + dx), (int32_t)(mk.j + dy), (int32_t)(mk.k + dz)};
                            auto it = weak_by_key.find(qk);
                            if (it == weak_by_key.end()) continue;
                            for (int q : it->second) {
                                const Vec3& pq = mesh.verts[q].position;
                                double dseg = point_segment_distance(pq, pa, pb);
                                if (dseg > be.radius) continue;
                                double da = (pq - pa).norm();
                                double db = (pq - pb).norm();
                                double maxe = s.cdp_max_edge_factor * s.voxel_size;
                                if (da > maxe || db > maxe) continue;
                                double score = dseg + 0.15 * (da + db);
                                cands.push_back({score, q});
                            }
                        }
                    }
                }
                if (cands.empty()) continue;
                std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.score < b.score; });
                if ((int)cands.size() > s.cdp_max_candidates_per_edge)
                    cands.resize(s.cdp_max_candidates_per_edge);

                // Add at most one best triangle per boundary edge per iteration.
                for (const Cand& cand : cands) {
                    size_t before = mesh.faces.size();
                    add_oriented_triangle_cdp(mesh, face_set, be.a, be.b, cand.idx);
                    if (mesh.faces.size() > before) {
                        added_this_iter++;
                        break;
                    }
                }
            }
            total_added += added_this_iter;
            std::printf("  [corner_dc_plus] iter=%d boundary_edges=%zu weak_targets=%zu added=%zu\n",
                        iter, boundary_edges.size(), n_weak, added_this_iter);
            if (added_this_iter == 0) break;
        }
        if (s.enable_eogm_generative_fill) {
            size_t gen_added = apply_eogm_generative_fill_pass(mesh, face_set);
            total_added += gen_added * 3;
        }
        if (hierarchical_scaffold_fill_enabled()) {
            size_t scaffold_added = apply_hierarchical_scaffold_fill_pass(mesh, face_set);
            total_added += scaffold_added;
        }
        if (s.enable_component_growth) {
            // PlanarMesh-like stages 4-6 around the growth pass:
            // delete contradicted faces, shrink over-long high-curvature edges,
            // grow boundary components, then re-validate and optionally thin.
            total_added += apply_component_refinement_pipeline(mesh, face_set);
        }
        if (total_added > 0) {
            std::printf("  [corner_dc_plus] total_added=%zu final_faces=%zu\n",
                        total_added, mesh.faces.size());
        }
        return mesh;
    }



    // ---- Component-growth topology pass ----------------------------------- //
    //
    // This pass adds a PlanarMesh-like "Grow" concept without changing the
    // fixed voxel-QEM storage. It starts from an existing conservative mesh
    // (usually corner_dc_plus + optional scaffold), extracts connected face
    // components, estimates a local QEM/normal support for each component, and
    // grows boundary edges toward low-degree/uncovered voxel-QEM vertices.
    //
    // The important distinction from the smooth mesher is that this is not a
    // blind local fan triangulation. Growth is seeded only from current mesh
    // boundary edges, is component-plane gated, uses adaptive boundary radii,
    // and keeps manifold/EOGM/QEM checks.

    struct ComponentGrowComp {
        std::vector<int> verts;
        std::vector<std::pair<int,int>> boundary_edges;
        Vec3 point = Vec3::Zero();
        Vec3 normal = Vec3::Zero();
        double sqrt_residual = 0.0;
        bool valid = false;
        int face_count = 0;
    };

    struct DSU {
        std::vector<int> p;
        explicit DSU(int n=0) : p(n) { std::iota(p.begin(), p.end(), 0); }
        int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
        void unite(int a, int b) { a = find(a); b = find(b); if (a != b) p[b] = a; }
    };

    void refresh_persistent_components_from_mesh(const MeshData& mesh,
                                                 const char* reason,
                                                 bool clear_dirty_after) const {
        if (!s.component_growth_persistent_state) return;
        const int N = (int)mesh.verts.size();
        if (N == 0) {
            persistent_components.clear();
            persistent_component_by_key.clear();
            if (clear_dirty_after && s.component_growth_clear_dirty_after_mesh) dirty_component_voxel_keys.clear();
            return;
        }

        std::vector<int> incident(N, 0);
        std::unordered_map<uint64_t, int> edge_count;
        edge_count.reserve(mesh.faces.size() * 3 + 1);
        DSU dsu(N);
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
            incident[f.a]++; incident[f.b]++; incident[f.c]++;
            edge_count[edge_key(f.a, f.b)]++; edge_count[edge_key(f.b, f.c)]++; edge_count[edge_key(f.c, f.a)]++;
            dsu.unite(f.a, f.b); dsu.unite(f.b, f.c); dsu.unite(f.c, f.a);
        }

        std::unordered_map<int, ComponentGrowComp> comps;
        comps.reserve(mesh.faces.size() / 4 + 1);
        for (int i = 0; i < N; ++i) if (incident[i] > 0) comps[dsu.find(i)].verts.push_back(i);
        for (const auto& f : mesh.faces) { if (f.a >= 0 && f.a < N) comps[dsu.find(f.a)].face_count++; }
        for (const auto& kv : edge_count) {
            if (kv.second != 1) continue;
            int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
            if (a < 0 || b < 0 || a >= N || b >= N) continue;
            if (dsu.find(a) == dsu.find(b)) comps[dsu.find(a)].boundary_edges.push_back({a,b});
        }

        std::unordered_map<int, PersistentQEMSurfaceComponent> next_components;
        std::unordered_map<VoxKey, int, VoxHash> next_by_key;
        size_t boundary_vertices = 0;
        for (auto& kv : comps) {
            ComponentGrowComp& c = kv.second;
            if (c.face_count <= 0 || c.verts.empty()) continue;
            std::unordered_set<VoxKey, VoxHash> keys;
            keys.reserve(c.verts.size() * 2 + 1);
            for (int vid : c.verts) if (vid >= 0 && vid < N) keys.insert(mesh.verts[vid].key);
            int pid = persistent_id_by_majority_overlap(keys);
            if (pid < 0) pid = next_persistent_component_id++;

            PersistentQEMSurfaceComponent pc;
            pc.id = pid; pc.vertex_keys = keys; pc.face_count = c.face_count; pc.last_refresh_scan = scan_count;
            for (const VoxKey& k : keys) { if (component_key_is_dirty(k)) pc.dirty = true; next_by_key[k] = pid; }

            component_growth_component_plane(mesh, c.verts, c);
            pc.point = c.point; pc.normal = c.normal; pc.sqrt_residual = c.sqrt_residual;

            pc.owned_faces.reserve(c.face_count);
            for (const auto& f : mesh.faces) {
                if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
                if (dsu.find(f.a) != kv.first) continue;
                pc.owned_faces.push_back({mesh.verts[f.a].key, mesh.verts[f.b].key, mesh.verts[f.c].key});
            }
            pc.owned_edges.reserve(edge_count.size());
            for (const auto& ekv : edge_count) {
                int a = (int)(ekv.first >> 32), b = (int)(ekv.first & 0xffffffffu);
                if (a < 0 || b < 0 || a >= N || b >= N) continue;
                if (dsu.find(a) != kv.first || dsu.find(b) != kv.first) continue;
                pc.owned_edges.push_back({mesh.verts[a].key, mesh.verts[b].key});
            }
            for (const auto& be : c.boundary_edges) {
                if (be.first < 0 || be.second < 0 || be.first >= N || be.second >= N) continue;
                const VertexRecord& va = mesh.verts[be.first];
                const VertexRecord& vb = mesh.verts[be.second];
                pc.boundary_vertex_keys.insert(va.key); pc.boundary_vertex_keys.insert(vb.key);
                pc.boundary_radius[va.key] = std::max(pc.boundary_radius[va.key], component_growth_radius_for_vertex(va));
                pc.boundary_radius[vb.key] = std::max(pc.boundary_radius[vb.key], component_growth_radius_for_vertex(vb));
                if (component_key_is_dirty(va.key) || component_key_is_dirty(vb.key)) pc.dirty = true;
            }
            boundary_vertices += pc.boundary_vertex_keys.size();
            next_components[pid] = std::move(pc);
        }

        persistent_components.swap(next_components);
        persistent_component_by_key.swap(next_by_key);
        persistent_component_refresh_count++;
        if (clear_dirty_after && s.component_growth_clear_dirty_after_mesh) dirty_component_voxel_keys.clear();

        if (s.enable_component_growth) {
            std::printf("  [component_state] reason=%s components=%zu boundary_vertices=%zu dirty_keys=%zu refresh=%ld\n",
                        reason ? reason : "mesh", persistent_components.size(), boundary_vertices,
                        dirty_component_voxel_keys.size(), persistent_component_refresh_count);
        }
    }

    double component_growth_radius_for_vertex(const VertexRecord& r) const {
        double rad = s.component_growth_boundary_radius_factor * s.voxel_size;
        if (std::strcmp(r.kind, "flat") == 0) {
            rad = std::max(rad, s.component_growth_flat_radius_factor * s.voxel_size);
        } else {
            rad *= 0.80;
        }
        if (r.confidence_tier != 0) rad *= 0.85;
        if (r.normal_consistency > 0.0 && r.normal_consistency < 0.85) rad *= 0.70;
        rad = std::clamp(rad, 1.0 * s.voxel_size,
                         std::max(1.0, s.component_growth_max_radius_factor) * s.voxel_size);
        return rad;
    }

    bool component_growth_vertex_visible_ok(const VertexRecord& r) const {
        if (s.enable_eogm) {
            if (r.eogm_bel_free > s.component_growth_max_bel_free) return false;
            if (r.eogm_conflict > s.component_growth_max_conflict) return false;
            // Mesh-only scaffold vertices often have plaus=1/unknown=1. This gate
            // mainly blocks real/inherited vertices with strong contrary evidence.
            if (r.eogm_plaus_surface > 0.0 && r.eogm_plaus_surface < s.component_growth_min_plaus_surface)
                return false;
        }
        return true;
    }


    struct QEMRankGrowthInfo {
        int rank = 1;              // 1=flat/plane, 2=edge, 3=corner/junction
        Vec3 normal = Vec3::Zero();
        Vec3 edge_dir = Vec3::Zero();
        bool valid = false;
    };

    QEMRankGrowthInfo qem_rank_info_for_vertex(const VertexRecord& r) const {
        QEMRankGrowthInfo out;
        if (std::strcmp(r.kind, "edge") == 0) out.rank = 2;
        else if (std::strcmp(r.kind, "corner") == 0) out.rank = 3;
        else out.rank = 1;

        Vec3 rn = r.normal;
        if (normalized_or_zero(rn)) out.normal = rn;

        auto it = cells.find(r.key);
        if (it != cells.end() && it->second.weight_eff() > 1e-12) {
            auto info = it->second.eigen_analysis(s);
            // QEM A is a sum of tangent-plane normal outer-products.
            // Rank-1: largest eigenvector is the plane normal.
            // Rank-2: smallest eigenvector is the edge tangent/null direction.
            Vec3 n = info.evecs.col(0);
            if (normalized_or_zero(n)) out.normal = n;
            Vec3 e = info.evecs.col(2);
            if (normalized_or_zero(e)) out.edge_dir = e;
        }
        if (normalized_or_zero(out.normal)) { out.valid = true; return out; }
        if (out.rank == 2 && normalized_or_zero(out.edge_dir)) { out.valid = true; return out; }
        // A corner without a stable single normal can still be used as a local
        // terminal junction; the component normal will be used by the caller.
        if (out.rank >= 3) { out.valid = true; return out; }
        return out;
    }

    double qem_rank_radius_for_vertex(const VertexRecord& r) const {
        double base = component_growth_radius_for_vertex(r);
        QEMRankGrowthInfo qi = qem_rank_info_for_vertex(r);
        if (qi.rank == 2) base *= 0.80;
        if (qi.rank >= 3) base *= std::clamp(s.component_growth_corner_radius_factor, 0.10, 1.0);
        return std::clamp(base, 0.50 * s.voxel_size,
                          std::max(1.0, s.component_growth_max_radius_factor) * s.voxel_size);
    }

    bool component_growth_qem_rank_rrs_accept(const VertexRecord& boundary,
                                              const VertexRecord& target,
                                              const ComponentGrowComp& comp) const {
        QEMRankGrowthInfo bq = qem_rank_info_for_vertex(boundary);
        QEMRankGrowthInfo tq = qem_rank_info_for_vertex(target);
        if (!bq.valid || !tq.valid) return false;

        Vec3 d = target.position - boundary.position;
        double dist = d.norm();
        double radius = qem_rank_radius_for_vertex(boundary);
        if (dist > radius) return false;

        Vec3 cn = comp.normal;
        if (!normalized_or_zero(cn)) cn = bq.normal;
        if (!normalized_or_zero(cn)) return false;

        Vec3 tn = target.normal;
        if (!normalized_or_zero(tn)) return false;
        // Corner targets may have mixed normals; use a softer transition test.
        double trans_dot = (tq.rank >= 3 || bq.rank >= 3)
            ? s.component_growth_rank_transition_normal_dot
            : s.component_growth_component_normal_dot;
        if (std::abs(tn.dot(cn)) < trans_dot) return false;

        if (bq.rank == 1) {
            // Rank-1 planar boundary: query region is a disk in the local plane.
            Vec3 n = bq.normal;
            if (!normalized_or_zero(n)) n = cn;
            double normal_dist = std::abs(d.dot(n));
            if (normal_dist > s.component_growth_plane_dist_factor * s.voxel_size) return false;
            Vec3 tang = d - d.dot(n) * n;
            if (tang.norm() > radius) return false;
            // Flat patches can stitch to edge/corner vertices, but the transition
            // must remain close to the supporting plane.
            return true;
        }

        if (bq.rank == 2) {
            // Rank-2 edge boundary: growth is primarily along the QEM null line.
            // It can connect to flat/corner vertices only if they lie close to that line.
            Vec3 e = bq.edge_dir;
            if (!normalized_or_zero(e)) return false;
            double along = std::abs(d.dot(e));
            Vec3 off = d - d.dot(e) * e;
            double line_tol = std::max(0.25 * s.voxel_size,
                s.component_growth_edge_line_dist_factor * s.voxel_size);
            if (off.norm() > line_tol) return false;
            if (along > radius) return false;
            return true;
        }

        // Rank-3/corner boundary: terminal junction. It may connect locally, but
        // it should not drive growth beyond a small neighborhood.
        return dist <= std::clamp(s.component_growth_corner_radius_factor, 0.10, 1.0) * radius;
    }

    bool component_growth_edge_target_rrs_accept(const MeshData& mesh,
                                                 int a, int b, int q,
                                                 const ComponentGrowComp& comp,
                                                 double radius_a,
                                                 double radius_b,
                                                 double dseg) const {
        if (!s.component_growth_use_rrs_boundary_search) {
            return dseg <= std::min(radius_a, radius_b);
        }
        if (a < 0 || b < 0 || q < 0 ||
            a >= (int)mesh.verts.size() || b >= (int)mesh.verts.size() || q >= (int)mesh.verts.size()) return false;
        const VertexRecord& va = mesh.verts[a];
        const VertexRecord& vb = mesh.verts[b];
        const VertexRecord& vq = mesh.verts[q];
        if (s.component_growth_qem_rank_rrs) {
            return component_growth_qem_rank_rrs_accept(va, vq, comp) ||
                   component_growth_qem_rank_rrs_accept(vb, vq, comp);
        }
        double da = (vq.position - va.position).norm();
        double db = (vq.position - vb.position).norm();
        return da <= radius_a || db <= radius_b;
    }

    bool component_growth_component_plane(const MeshData& mesh,
                                          const std::vector<int>& ids,
                                          ComponentGrowComp& comp) const {
        if (ids.empty()) return false;
        Mat3 A = Mat3::Zero();
        Vec3 b = Vec3::Zero();
        double c = 0.0;
        double wsum = 0.0;
        Vec3 anchor = Vec3::Zero();
        Vec3 nsum = Vec3::Zero();
        double nweight = 0.0;

        for (int id : ids) {
            if (id < 0 || id >= (int)mesh.verts.size()) continue;
            const VertexRecord& r = mesh.verts[id];
            anchor += r.position;
            Vec3 rn = r.normal;
            if (normalized_or_zero(rn)) {
                // Sign-align to the first accumulated normal.
                if (nweight > 0.0 && rn.dot(nsum) < 0.0) rn = -rn;
                double w = std::max(0.05, r.weight_sum);
                nsum += w * rn;
                nweight += w;
            }
            auto it = cells.find(r.key);
            if (it == cells.end()) continue;
            const VoxelCell& vc = it->second;
            if (vc.weight_eff() <= 1e-12) continue;
            A.noalias() += vc.A_eff();
            b.noalias() += vc.b_eff();
            c += vc.c_eff();
            wsum += vc.weight_eff();
        }

        anchor /= (double)std::max(1, (int)ids.size());
        Vec3 x = anchor;
        Vec3 n = Vec3::Zero();
        double sr = 0.0;

        if (wsum > 1e-12) {
            Mat3 Areg = A + s.lambda_p * Mat3::Identity();
            x = Areg.ldlt().solve(b + s.lambda_p * anchor);
            double f = x.transpose() * A * x;
            f += c;
            f -= 2.0 * b.dot(x);
            sr = std::sqrt(std::max(0.0, f / wsum));
            if (sr > s.component_growth_max_merged_sqrt_residual) return false;

            Eigen::SelfAdjointEigenSolver<Mat3> eig(A);
            if (eig.info() == Eigen::Success) n = eig.eigenvectors().col(2); // largest QEM eigendir ≈ normal
        }

        if (!normalized_or_zero(n)) {
            n = nsum;
            if (!normalized_or_zero(n)) return false;
        }
        if (nweight > 0.0 && n.dot(nsum) < 0.0) n = -n;

        comp.point = x;
        comp.normal = n;
        comp.sqrt_residual = sr;
        comp.valid = true;
        return true;
    }

    bool component_growth_triangle_ok(const MeshData& mesh,
                                      int a, int b, int c,
                                      const Vec3& component_point,
                                      const Vec3& component_normal,
                                      const std::unordered_map<uint64_t, int>& edge_count) const {
        if (a < 0 || b < 0 || c < 0 ||
            a >= (int)mesh.verts.size() || b >= (int)mesh.verts.size() || c >= (int)mesh.verts.size()) return false;
        if (a == b || a == c || b == c) return false;
        if (!triangle_edges_can_accept(edge_count, a, b, c)) return false;

        const VertexRecord& va = mesh.verts[a];
        const VertexRecord& vb = mesh.verts[b];
        const VertexRecord& vc = mesh.verts[c];
        if (!component_growth_vertex_visible_ok(va) ||
            !component_growth_vertex_visible_ok(vb) ||
            !component_growth_vertex_visible_ok(vc)) return false;

        const double max_edge = s.component_growth_max_edge_factor * s.voxel_size;
        double eab = (va.position - vb.position).norm();
        double ebc = (vb.position - vc.position).norm();
        double eca = (vc.position - va.position).norm();
        if (eab > max_edge || ebc > max_edge || eca > max_edge) return false;

        Vec3 tn = (vb.position - va.position).cross(vc.position - va.position);
        double twice_area = tn.norm();
        if (twice_area < 2.0 * s.dc_min_area_factor * s.voxel_size * s.voxel_size) return false;
        tn /= twice_area;

        Vec3 cn = component_normal;
        if (!normalized_or_zero(cn)) return false;
        if (std::abs(tn.dot(cn)) < s.dc_triangle_normal_dot) return false;

        Vec3 na = va.normal, nb = vb.normal, nc = vc.normal;
        if (!normalized_or_zero(na) || !normalized_or_zero(nb) || !normalized_or_zero(nc)) return false;
        if (std::abs(na.dot(cn)) < s.component_growth_component_normal_dot) return false;
        if (std::abs(nb.dot(cn)) < s.component_growth_component_normal_dot) return false;
        if (std::abs(nc.dot(cn)) < s.component_growth_component_normal_dot) return false;
        if (std::abs(na.dot(nb)) < s.component_growth_normal_dot) return false;
        if (std::abs(nb.dot(nc)) < s.component_growth_normal_dot) return false;
        if (std::abs(nc.dot(na)) < s.component_growth_normal_dot) return false;

        const double max_pd = s.component_growth_plane_dist_factor * s.voxel_size;
        if (std::abs(cn.dot(va.position - component_point)) > max_pd) return false;
        if (std::abs(cn.dot(vb.position - component_point)) > max_pd) return false;
        if (std::abs(cn.dot(vc.position - component_point)) > max_pd) return false;

        if (s.component_growth_reject_free_samples &&
            !eogm_samples_ok_for_triangle(va.position, vb.position, vc.position,
                                          s.component_growth_max_bel_free,
                                          s.component_growth_max_conflict,
                                          s.component_growth_min_plaus_surface)) return false;

        if (s.component_growth_require_confirmed_anchor) {
            const bool has_confirmed = (va.confidence_tier == 0) || (vb.confidence_tier == 0) || (vc.confidence_tier == 0);
            if (!has_confirmed) return false;
        }

        if (s.component_growth_require_merged_qem_when_available) {
            // Enforce merged QEM when all three vertices have backing cells. Mesh-only
            // scaffold vertices may not, in which case the component plane gates above
            // are the controlling geometry prior.
            bool all_have_cells = cells.find(va.key) != cells.end() &&
                                  cells.find(vb.key) != cells.end() &&
                                  cells.find(vc.key) != cells.end();
            if (all_have_cells) {
                Vec3 pp, pn; double sr = 0.0;
                if (!merged_qem_plane_for_ids(mesh, a, b, c, pp, pn, sr,
                                              s.component_growth_max_merged_sqrt_residual,
                                              s.component_growth_max_point_plane_dist_factor)) return false;
            }
        }
        return true;
    }

    size_t apply_component_growth_pass(MeshData& mesh,
                                       std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                                       bool force=false) const {
        if ((!force && !s.enable_component_growth) || mesh.verts.size() < 3) return 0;

        size_t total_added = 0;
        const int iters = std::max(1, s.component_growth_iters);
        for (int iter = 0; iter < iters; ++iter) {
            const int N = (int)mesh.verts.size();
            if (N < 3) break;

            std::vector<int> incident(N, 0);
            std::unordered_map<uint64_t, int> edge_count;
            edge_count.reserve(mesh.faces.size() * 3 + 1);
            DSU dsu(N);
            for (const auto& f : mesh.faces) {
                if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
                incident[f.a]++; incident[f.b]++; incident[f.c]++;
                edge_count[edge_key(f.a, f.b)]++;
                edge_count[edge_key(f.b, f.c)]++;
                edge_count[edge_key(f.c, f.a)]++;
                dsu.unite(f.a, f.b); dsu.unite(f.b, f.c); dsu.unite(f.c, f.a);
            }
            if (mesh.faces.empty()) break;

            std::unordered_map<int, ComponentGrowComp> comps;
            comps.reserve(mesh.faces.size() / 4 + 1);
            for (int i = 0; i < N; ++i) if (incident[i] > 0) {
                int r = dsu.find(i);
                comps[r].verts.push_back(i);
            }
            for (const auto& f : mesh.faces) {
                if (f.a < 0 || f.a >= N) continue;
                comps[dsu.find(f.a)].face_count++;
            }
            for (const auto& kv : edge_count) {
                if (kv.second != 1) continue;
                int a = (int)(kv.first >> 32);
                int b = (int)(kv.first & 0xffffffffu);
                if (a < 0 || b < 0 || a >= N || b >= N) continue;
                int ra = dsu.find(a), rb = dsu.find(b);
                if (ra != rb) continue;
                comps[ra].boundary_edges.push_back({a,b});
            }

            // Low-degree/uncovered vertices are growth targets. They may be real,
            // weak, inherited, or mesh-only scaffold vertices, but they must pass
            // visibility gates. In dirty-only mode, targets are limited to the
            // dirty voxel neighborhood accumulated since the previous mesh build.
            std::unordered_map<VoxKey, std::vector<int>, VoxHash> cand_by_key;
            cand_by_key.reserve((size_t)N / 2 + 1);
            size_t candidate_count = 0;
            for (int i = 0; i < N; ++i) {
                if (incident[i] >= s.component_growth_target_min_incident_faces) continue;
                if (!component_growth_vertex_visible_ok(mesh.verts[i])) continue;
                if (s.component_growth_dirty_only && !dirty_component_voxel_keys.empty() &&
                    !component_key_is_dirty(mesh.verts[i].key)) continue;
                cand_by_key[mesh.verts[i].key].push_back(i);
                candidate_count++;
            }
            if (candidate_count == 0) break;

            size_t added_iter = 0;
            const double max_radius = std::max(1.0, s.component_growth_max_radius_factor) * s.voxel_size;
            const int max_vox_rad = std::max(1, (int)std::ceil(max_radius / s.voxel_size) + 1);

            for (auto& ckvc : comps) {
                int root = ckvc.first;
                ComponentGrowComp& comp = ckvc.second;
                if (comp.face_count < s.component_growth_min_component_faces) continue;
                if (comp.boundary_edges.empty()) continue;
                if (!current_component_is_dirty(mesh, comp.verts, comp.boundary_edges)) continue;
                if (!component_growth_component_plane(mesh, comp.verts, comp)) continue;

                for (const auto& be : comp.boundary_edges) {
                    int a = be.first, b = be.second;
                    if (a < 0 || b < 0 || a >= N || b >= N) continue;
                    if (!component_growth_vertex_visible_ok(mesh.verts[a]) ||
                        !component_growth_vertex_visible_ok(mesh.verts[b])) continue;

                    const Vec3& pa = mesh.verts[a].position;
                    const Vec3& pb = mesh.verts[b].position;
                    Vec3 mid = 0.5 * (pa + pb);
                    VoxKey mk = voxel_index(mid);
                    const double radius_a = s.component_growth_qem_rank_rrs
                        ? qem_rank_radius_for_vertex(mesh.verts[a])
                        : component_growth_radius_for_vertex(mesh.verts[a]);
                    const double radius_b = s.component_growth_qem_rank_rrs
                        ? qem_rank_radius_for_vertex(mesh.verts[b])
                        : component_growth_radius_for_vertex(mesh.verts[b]);
                    const double grow_radius = std::min(radius_a, radius_b);

                    struct Cand { double score; int idx; };
                    std::vector<Cand> cands;
                    for (int dx = -max_vox_rad; dx <= max_vox_rad; ++dx)
                    for (int dy = -max_vox_rad; dy <= max_vox_rad; ++dy)
                    for (int dz = -max_vox_rad; dz <= max_vox_rad; ++dz) {
                        VoxKey qk{(int32_t)(mk.i + dx), (int32_t)(mk.j + dy), (int32_t)(mk.k + dz)};
                        auto it = cand_by_key.find(qk);
                        if (it == cand_by_key.end()) continue;
                        for (int q : it->second) {
                            if (q == a || q == b) continue;
                            if (incident[q] >= s.component_growth_target_min_incident_faces) continue;
                            if (incident[q] > 0 && dsu.find(q) == root) continue; // already inside this component
                            if (!triangle_edges_can_accept(edge_count, a, b, q)) continue;
                            const Vec3& pq = mesh.verts[q].position;
                            double dseg = point_segment_distance(pq, pa, pb);
                            double da = (pq - pa).norm();
                            double db = (pq - pb).norm();
                            if (!component_growth_edge_target_rrs_accept(mesh, a, b, q, comp,
                                                                          radius_a, radius_b, dseg)) continue;
                            double maxe = s.component_growth_max_edge_factor * s.voxel_size;
                            if (da > maxe || db > maxe) continue;
                            double pd = std::abs(comp.normal.dot(pq - comp.point));
                            if (pd > s.component_growth_plane_dist_factor * s.voxel_size) continue;
                            Vec3 qn = mesh.verts[q].normal;
                            if (!normalized_or_zero(qn)) continue;
                            if (std::abs(qn.dot(comp.normal)) < s.component_growth_component_normal_dot) continue;
                            double score = dseg + 0.10 * (da + db) + 2.0 * pd;
                            cands.push_back({score, q});
                        }
                    }
                    if (cands.empty()) continue;
                    std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y){ return x.score < y.score; });
                    if ((int)cands.size() > s.component_growth_max_candidates_per_edge)
                        cands.resize(s.component_growth_max_candidates_per_edge);

                    for (const Cand& cand : cands) {
                        int q = cand.idx;
                        if (!component_growth_triangle_ok(mesh, a, b, q, comp.point, comp.normal, edge_count)) continue;
                        int fa = a, fb = b, fc = q;
                        Vec3 tri_n = (mesh.verts[fb].position - mesh.verts[fa].position)
                                   .cross(mesh.verts[fc].position - mesh.verts[fa].position);
                        Vec3 en = comp.normal;
                        if (normalized_or_zero(en) && tri_n.dot(en) < 0.0) std::swap(fb, fc);
                        FaceKey fk = sorted_face_key(fa, fb, fc);
                        if (!face_set.insert(fk).second) continue;
                        mesh.faces.push_back({fa, fb, fc});
                        register_triangle_edges(edge_count, fa, fb, fc);
                        incident[fa]++; incident[fb]++; incident[fc]++;
                        added_iter++;
                        break; // one growth triangle per boundary edge per iter
                    }
                }
            }

            total_added += added_iter;
            std::printf("  [component_growth] iter=%d components=%zu candidates=%zu added=%zu\n",
                        iter, comps.size(), candidate_count, added_iter);
            if (added_iter == 0) break;
        }

        if (total_added > 0) {
            std::printf("  [component_growth] total_added=%zu final_faces=%zu\n",
                        total_added, mesh.faces.size());
        }
        return total_added;
    }

    void rebuild_face_set_from_mesh(const MeshData& mesh,
                                    std::unordered_set<FaceKey, FaceKeyHash>& face_set) const {
        face_set.clear();
        face_set.reserve(mesh.faces.size() * 2 + 1024);
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 ||
                f.a >= (int)mesh.verts.size() || f.b >= (int)mesh.verts.size() || f.c >= (int)mesh.verts.size()) continue;
            face_set.insert(sorted_face_key(f.a, f.b, f.c));
        }
    }

    bool component_face_dirty_or_global(const MeshData& mesh, const MeshFace& f, bool dirty_only) const {
        if (!dirty_only) return true;
        if (dirty_component_voxel_keys.empty()) return true;
        if (f.a >= 0 && f.a < (int)mesh.verts.size() && component_key_is_dirty(mesh.verts[f.a].key)) return true;
        if (f.b >= 0 && f.b < (int)mesh.verts.size() && component_key_is_dirty(mesh.verts[f.b].key)) return true;
        if (f.c >= 0 && f.c < (int)mesh.verts.size() && component_key_is_dirty(mesh.verts[f.c].key)) return true;
        return false;
    }

    bool component_face_free_space_contradicted(const MeshData& mesh, const MeshFace& f) const {
        if (f.a < 0 || f.b < 0 || f.c < 0 ||
            f.a >= (int)mesh.verts.size() || f.b >= (int)mesh.verts.size() || f.c >= (int)mesh.verts.size()) return true;
        const Vec3& pa = mesh.verts[f.a].position;
        const Vec3& pb = mesh.verts[f.b].position;
        const Vec3& pc = mesh.verts[f.c].position;
        Vec3 samples[7] = {
            (pa + pb + pc) / 3.0,
            0.5 * (pa + pb), 0.5 * (pb + pc), 0.5 * (pc + pa),
            0.60 * pa + 0.20 * pb + 0.20 * pc,
            0.20 * pa + 0.60 * pb + 0.20 * pc,
            0.20 * pa + 0.20 * pb + 0.60 * pc
        };
        int seen = 0;
        double plaus_sum = 0.0;
        for (const Vec3& x : samples) {
            auto it = cells.find(voxel_index(x));
            if (it == cells.end()) continue;
            const VoxelCell& vc = it->second;
            if (vc.label(s) == VoxelCell::Label::FREE) return true;
            if (s.enable_eogm) {
                if (vc.eogm_bel_free() > s.component_fis_delete_max_bel_free) return true;
                if (vc.eogm_conflict_mass() > s.component_fis_delete_max_conflict) return true;
                plaus_sum += vc.eogm_plaus_surface();
                seen++;
            }
        }
        if (s.enable_eogm && seen > 0 &&
            (plaus_sum / (double)seen) < s.component_fis_delete_min_plaus_surface) return true;
        return false;
    }

    size_t apply_component_fis_delete_pass(MeshData& mesh,
                                           std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                                           const char* reason) const {
        if (!s.enable_component_fis_delete || mesh.faces.empty()) return 0;
        std::vector<MeshFace> kept;
        kept.reserve(mesh.faces.size());
        size_t removed = 0;
        for (const MeshFace& f : mesh.faces) {
            bool check = component_face_dirty_or_global(mesh, f, s.component_fis_delete_dirty_only);
            if (check && component_face_free_space_contradicted(mesh, f)) {
                removed++;
                continue;
            }
            kept.push_back(f);
        }
        if (removed > 0) {
            mesh.faces.swap(kept);
            rebuild_face_set_from_mesh(mesh, face_set);
            std::printf("  [component_fis_delete] reason=%s removed=%zu faces=%zu\n",
                        reason ? reason : "", removed, mesh.faces.size());
        }
        return removed;
    }

    bool component_edge_exceeds_rank_radius(const MeshData& mesh, int a, int b) const {
        if (a < 0 || b < 0 || a >= (int)mesh.verts.size() || b >= (int)mesh.verts.size()) return true;
        const VertexRecord& va = mesh.verts[a];
        const VertexRecord& vb = mesh.verts[b];
        double d = (va.position - vb.position).norm();
        double ra = s.component_growth_qem_rank_rrs ? qem_rank_radius_for_vertex(va) : component_growth_radius_for_vertex(va);
        double rb = s.component_growth_qem_rank_rrs ? qem_rank_radius_for_vertex(vb) : component_growth_radius_for_vertex(vb);
        double allowed = s.component_shrink_edge_over_radius * std::min(ra, rb);
        allowed = std::min(allowed, s.component_growth_max_edge_factor * s.voxel_size);
        return d > allowed;
    }

    size_t apply_component_radius_shrink_pass(MeshData& mesh,
                                              std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                                              const char* reason) const {
        if (!s.enable_component_radius_shrink || mesh.faces.empty()) return 0;
        std::vector<int> incident(mesh.verts.size(), 0);
        DSU dsu((int)mesh.verts.size());
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 ||
                f.a >= (int)mesh.verts.size() || f.b >= (int)mesh.verts.size() || f.c >= (int)mesh.verts.size()) continue;
            incident[f.a]++; incident[f.b]++; incident[f.c]++;
            dsu.unite(f.a, f.b); dsu.unite(f.b, f.c); dsu.unite(f.c, f.a);
        }
        std::unordered_map<int, int> comp_faces;
        for (const auto& f : mesh.faces) if (f.a >= 0 && f.a < (int)mesh.verts.size()) comp_faces[dsu.find(f.a)]++;

        std::vector<MeshFace> kept;
        kept.reserve(mesh.faces.size());
        size_t removed = 0;
        for (const MeshFace& f : mesh.faces) {
            bool check = component_face_dirty_or_global(mesh, f, s.component_radius_shrink_dirty_only);
            if (check) {
                int root = (f.a >= 0 && f.a < (int)mesh.verts.size()) ? dsu.find(f.a) : -1;
                int fc = root >= 0 ? comp_faces[root] : 0;
                bool too_long = fc >= s.component_shrink_min_component_faces &&
                    (component_edge_exceeds_rank_radius(mesh, f.a, f.b) ||
                     component_edge_exceeds_rank_radius(mesh, f.b, f.c) ||
                     component_edge_exceeds_rank_radius(mesh, f.c, f.a));
                if (too_long) { removed++; continue; }
            }
            kept.push_back(f);
        }
        if (removed > 0) {
            mesh.faces.swap(kept);
            rebuild_face_set_from_mesh(mesh, face_set);
            std::printf("  [component_shrink] reason=%s removed=%zu faces=%zu\n",
                        reason ? reason : "", removed, mesh.faces.size());
        }
        return removed;
    }

    size_t compact_mesh_vertices(MeshData& mesh,
                                 std::unordered_set<FaceKey, FaceKeyHash>& face_set) const {
        if (!s.component_compact_after_topology_ops || mesh.verts.empty()) return 0;
        std::vector<char> used(mesh.verts.size(), 0);
        for (const auto& f : mesh.faces) {
            if (f.a >= 0 && f.a < (int)used.size()) used[f.a] = 1;
            if (f.b >= 0 && f.b < (int)used.size()) used[f.b] = 1;
            if (f.c >= 0 && f.c < (int)used.size()) used[f.c] = 1;
        }
        std::vector<int> remap(mesh.verts.size(), -1);
        std::vector<VertexRecord> new_verts;
        new_verts.reserve(mesh.verts.size());
        for (int i = 0; i < (int)mesh.verts.size(); ++i) {
            if (!used[i]) continue;
            remap[i] = (int)new_verts.size();
            new_verts.push_back(mesh.verts[i]);
        }
        if (new_verts.size() == mesh.verts.size()) return 0;
        for (auto& f : mesh.faces) {
            f.a = remap[f.a]; f.b = remap[f.b]; f.c = remap[f.c];
        }
        size_t removed = mesh.verts.size() - new_verts.size();
        mesh.verts.swap(new_verts);
        rebuild_face_set_from_mesh(mesh, face_set);
        return removed;
    }

    size_t apply_component_adaptive_simplification_pass(MeshData& mesh,
                                                        std::unordered_set<FaceKey, FaceKeyHash>& face_set) const {
        if (!s.enable_component_adaptive_simplification || mesh.faces.empty()) return 0;
        const int N = (int)mesh.verts.size();
        std::unordered_map<uint64_t, int> edge_count;
        edge_count.reserve(mesh.faces.size() * 3 + 1);
        DSU dsu(N);
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
            edge_count[edge_key(f.a, f.b)]++;
            edge_count[edge_key(f.b, f.c)]++;
            edge_count[edge_key(f.c, f.a)]++;
            dsu.unite(f.a, f.b); dsu.unite(f.b, f.c); dsu.unite(f.c, f.a);
        }
        std::unordered_map<int, std::vector<int>> faces_by_comp;
        for (int fi = 0; fi < (int)mesh.faces.size(); ++fi) {
            const auto& f = mesh.faces[fi];
            if (f.a >= 0 && f.a < N) faces_by_comp[dsu.find(f.a)].push_back(fi);
        }
        std::vector<char> keep(mesh.faces.size(), 1);
        size_t removed = 0;
        for (auto& kv : faces_by_comp) {
            const std::vector<int>& flist = kv.second;
            if ((int)flist.size() < s.component_simplify_min_component_faces) continue;
            std::vector<int> vids;
            vids.reserve(flist.size() * 3);
            std::unordered_set<int> seen;
            for (int fi : flist) {
                const auto& f = mesh.faces[fi];
                if (seen.insert(f.a).second) vids.push_back(f.a);
                if (seen.insert(f.b).second) vids.push_back(f.b);
                if (seen.insert(f.c).second) vids.push_back(f.c);
            }
            ComponentGrowComp comp;
            if (!component_growth_component_plane(mesh, vids, comp)) continue;
            // Only thin strongly planar components.
            int flat_votes = 0;
            for (int v : vids) if (std::strcmp(mesh.verts[v].kind, "flat") == 0) flat_votes++;
            if (flat_votes < (int)(0.80 * (double)vids.size())) continue;

            std::vector<Vec3> kept_centroids;
            kept_centroids.reserve(flist.size());
            double rad = std::max(0.5 * s.voxel_size,
                                  s.component_simplify_centroid_radius_factor * s.voxel_size);
            for (int fi : flist) {
                const auto& f = mesh.faces[fi];
                bool boundary_face = edge_use_count(edge_count, f.a, f.b) == 1 ||
                                     edge_use_count(edge_count, f.b, f.c) == 1 ||
                                     edge_use_count(edge_count, f.c, f.a) == 1;
                if (s.component_simplify_preserve_boundary_faces && boundary_face) {
                    kept_centroids.push_back((mesh.verts[f.a].position + mesh.verts[f.b].position + mesh.verts[f.c].position) / 3.0);
                    continue;
                }
                Vec3 tn = (mesh.verts[f.b].position - mesh.verts[f.a].position)
                        .cross(mesh.verts[f.c].position - mesh.verts[f.a].position);
                if (!normalized_or_zero(tn) || std::abs(tn.dot(comp.normal)) < s.component_simplify_normal_dot) continue;
                Vec3 ctr = (mesh.verts[f.a].position + mesh.verts[f.b].position + mesh.verts[f.c].position) / 3.0;
                bool redundant = false;
                for (const Vec3& kc : kept_centroids) {
                    if ((ctr - kc).norm() < rad) { redundant = true; break; }
                }
                if (redundant) { keep[fi] = 0; removed++; }
                else kept_centroids.push_back(ctr);
            }
        }
        if (removed > 0) {
            std::vector<MeshFace> new_faces;
            new_faces.reserve(mesh.faces.size() - removed);
            for (int i = 0; i < (int)mesh.faces.size(); ++i) if (keep[i]) new_faces.push_back(mesh.faces[i]);
            mesh.faces.swap(new_faces);
            rebuild_face_set_from_mesh(mesh, face_set);
            size_t removed_verts = compact_mesh_vertices(mesh, face_set);
            std::printf("  [component_simplify] removed_faces=%zu removed_unused_vertices=%zu faces=%zu verts=%zu\n",
                        removed, removed_verts, mesh.faces.size(), mesh.verts.size());
        }
        return removed;
    }


    bool persistent_face_key_retired(const VoxKey& k) const {
        auto it = cells.find(k);
        if (it == cells.end()) return true;
        const VoxelCell& c = it->second;
        if (c.label(s) == VoxelCell::Label::FREE) return true;
        if (!cell_vertex_export_ok(c)) return true;
        if (s.enable_eogm) {
            if (c.eogm_bel_free() > s.component_persistent_reuse_max_bel_free) return true;
            if (c.eogm_conflict_mass() > s.component_persistent_reuse_max_conflict) return true;
            if (c.eogm_plaus_surface() < s.component_persistent_reuse_min_plaus_surface) return true;
        }
        return false;
    }

    bool persistent_face_should_retain(const PersistentComponentFace& pf) const {
        const bool dirty = component_key_is_dirty(pf.a) || component_key_is_dirty(pf.b) || component_key_is_dirty(pf.c);
        if (dirty && !s.component_persistent_reuse_dirty_faces) return false;
        if (persistent_face_key_retired(pf.a) || persistent_face_key_retired(pf.b) || persistent_face_key_retired(pf.c)) return false;
        return true;
    }

    size_t emit_retained_persistent_faces(MeshData& mesh,
                                          std::unordered_set<FaceKey, FaceKeyHash>& face_set,
                                          const char* reason) const {
        if (!s.component_growth_persistent_state || !s.component_persistent_reuse_faces) return 0;
        if (persistent_components.empty()) return 0;

        std::unordered_map<VoxKey, int, VoxHash> idx_by_key;
        idx_by_key.reserve(mesh.verts.size() * 2 + 1);
        for (int i = 0; i < (int)mesh.verts.size(); ++i) idx_by_key[mesh.verts[i].key] = i;

        std::unordered_map<uint64_t, int> edge_count;
        edge_count.reserve(mesh.faces.size() * 3 + 1);
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 ||
                f.a >= (int)mesh.verts.size() || f.b >= (int)mesh.verts.size() || f.c >= (int)mesh.verts.size()) continue;
            edge_count[edge_key(f.a, f.b)]++;
            edge_count[edge_key(f.b, f.c)]++;
            edge_count[edge_key(f.c, f.a)]++;
        }

        size_t added = 0, skipped_retired = 0, skipped_missing = 0, skipped_manifold = 0;
        for (const auto& kv : persistent_components) {
            const PersistentQEMSurfaceComponent& pc = kv.second;
            for (const PersistentComponentFace& pf : pc.owned_faces) {
                if (!persistent_face_should_retain(pf)) { skipped_retired++; continue; }
                auto ia = idx_by_key.find(pf.a);
                auto ib = idx_by_key.find(pf.b);
                auto ic = idx_by_key.find(pf.c);
                if (ia == idx_by_key.end() || ib == idx_by_key.end() || ic == idx_by_key.end()) {
                    skipped_missing++;
                    if (s.component_persistent_reuse_require_all_vertices) continue;
                    else continue; // placeholder for future direct per-key vertex materialization
                }
                int a = ia->second, b = ib->second, c = ic->second;
                if (a == b || a == c || b == c) continue;
                FaceKey fk = sorted_face_key(a, b, c);
                if (face_set.find(fk) != face_set.end()) continue;
                if (!triangle_edges_can_accept(edge_count, a, b, c)) { skipped_manifold++; continue; }

                // Use current QEM-resolved vertex positions from mesh. Only topology is retained.
                Vec3 avg_n = mesh.verts[a].normal;
                Vec3 nb = mesh.verts[b].normal;
                Vec3 nc = mesh.verts[c].normal;
                if (!normalized_or_zero(avg_n) || !normalized_or_zero(nb) || !normalized_or_zero(nc)) continue;
                if (nb.dot(avg_n) < 0.0) nb = -nb;
                if (nc.dot(avg_n) < 0.0) nc = -nc;
                avg_n += nb + nc;
                if (!normalized_or_zero(avg_n)) continue;

                Vec3 tri_n = (mesh.verts[b].position - mesh.verts[a].position)
                           .cross(mesh.verts[c].position - mesh.verts[a].position);
                if (!normalized_or_zero(tri_n)) continue;
                int fa = a, fb = b, fc = c;
                if (tri_n.dot(avg_n) < 0.0) std::swap(fb, fc);

                // Keep retained faces under the same broad visibility/size constraints used by component growth.
                if (!component_growth_vertex_visible_ok(mesh.verts[fa]) ||
                    !component_growth_vertex_visible_ok(mesh.verts[fb]) ||
                    !component_growth_vertex_visible_ok(mesh.verts[fc])) continue;
                const double max_edge = s.component_growth_max_edge_factor * s.voxel_size;
                if ((mesh.verts[fa].position - mesh.verts[fb].position).norm() > max_edge ||
                    (mesh.verts[fb].position - mesh.verts[fc].position).norm() > max_edge ||
                    (mesh.verts[fc].position - mesh.verts[fa].position).norm() > max_edge) continue;

                face_set.insert(sorted_face_key(fa, fb, fc));
                mesh.faces.push_back({fa, fb, fc});
                register_triangle_edges(edge_count, fa, fb, fc);
                added++;
            }
        }
        if (added > 0 || (s.enable_component_growth && (skipped_retired > 0 || skipped_missing > 0))) {
            std::printf("  [component_reuse] reason=%s retained_faces=%zu skipped_retired=%zu skipped_missing=%zu skipped_manifold=%zu faces=%zu\n",
                        reason ? reason : "", added, skipped_retired, skipped_missing, skipped_manifold, mesh.faces.size());
        }
        return added;
    }

    size_t apply_component_refinement_pipeline(MeshData& mesh,
                                               std::unordered_set<FaceKey, FaceKeyHash>& face_set) const {
        size_t changed = 0;
        changed += apply_component_fis_delete_pass(mesh, face_set, "pre_growth");
        changed += apply_component_radius_shrink_pass(mesh, face_set, "pre_growth");
        changed += apply_component_growth_pass(mesh, face_set);
        changed += apply_component_fis_delete_pass(mesh, face_set, "post_growth");
        changed += apply_component_radius_shrink_pass(mesh, face_set, "post_growth");
        changed += apply_component_adaptive_simplification_pass(mesh, face_set);
        if (changed > 0) {
            size_t removed_verts = compact_mesh_vertices(mesh, face_set);
            if (removed_verts > 0) std::printf("  [component_compact] removed_unused_vertices=%zu\n", removed_verts);
        }
        return changed;
    }

    // ---- Optional edge-aware QEM vertex smoothing ---- //
    //
    // This pass is deliberately conservative. High residual alone does not
    // cause smoothing: residual can come from noise, a real edge/corner, sparse
    // support, or a mixed-surface voxel. We only smooth vertices that pass
    // confidence/feature/EOGM gates, and by default we protect confirmed flat
    // rank-1 vertices so walls/floors are not planarized or tangentially slid.
    // The default correction is normal-only and QEM-projected.

    static void add_unique_adj_edge(std::vector<std::vector<int>>& adj,
                                    std::unordered_set<uint64_t>& edges,
                                    int a, int b) {
        if (a == b) return;
        uint64_t ek = edge_key(a, b);
        if (!edges.insert(ek).second) return;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    bool vertex_smooth_kind_protected(const VertexRecord& r) const {
        if (!s.vertex_smooth_preserve_edges) return false;
        if (!r.kind) return false;
        // Preserve QEM edge/corner modes. Rank-1 flat modes are handled by the
        // confirmed-flat gate below, not by this feature gate.
        return std::strcmp(r.kind, "edge") == 0 || std::strcmp(r.kind, "corner") == 0;
    }

    bool vertex_smooth_visibility_ok(const Vec3& a, const Vec3& b) const {
        if (!s.enable_eogm) return true;
        Vec3 samples[3] = {
            0.5 * (a + b),
            0.67 * a + 0.33 * b,
            0.33 * a + 0.67 * b
        };
        for (const Vec3& x : samples) {
            auto it = cells.find(voxel_index(x));
            if (it == cells.end()) continue;
            const VoxelCell& vc = it->second;
            if (vc.label(s) == VoxelCell::Label::FREE) return false;
            if (vc.eogm_bel_free() > s.vertex_smooth_max_bel_free) return false;
            if (vc.eogm_conflict_mass() > s.vertex_smooth_max_conflict) return false;
        }
        return true;
    }

    bool vertex_smoothing_candidate_ok(const VertexRecord& r, int degree) const {
        if (!s.enable_vertex_smoothing) return false;
        if (degree < s.vertex_smooth_min_degree) return false;
        if (r.scan_boundary_ratio > s.vertex_smooth_max_boundary_ratio) return false;
        if (s.enable_eogm) {
            if (r.eogm_bel_free > s.vertex_smooth_max_bel_free) return false;
            if (r.eogm_conflict > s.vertex_smooth_max_conflict) return false;
        }
        if (r.normal_consistency < s.vertex_smooth_min_normal_consistency) return false;
        if (vertex_smooth_kind_protected(r)) return false;

        const bool weak_or_generated = (r.confidence_tier != 0);
        if (weak_or_generated) return true;

        // Confirmed vertices are protected by default. This prevents the
        // smoother from flattening/planarizing ordinary rank-1 walls/floors.
        if (!s.vertex_smooth_apply_to_confirmed) return false;
        if (r.hit_count < s.vertex_smooth_min_hit_count) return false;
        if (!s.vertex_smooth_allow_confirmed_flat && r.kind && std::strcmp(r.kind, "flat") == 0)
            return false;
        if (!std::isfinite(r.residual)) return false;
        if (std::sqrt(std::max(0.0, r.residual)) < s.vertex_smooth_min_sqrt_residual)
            return false;
        return true;
    }

    Vec3 qem_project_smooth_candidate(const VertexRecord& r,
                                      const Vec3& old_pos,
                                      const Vec3& candidate) const {
        if (!s.vertex_smooth_qem_project) return candidate;
        auto it = cells.find(r.key);
        if (it == cells.end()) return candidate; // generated/virtual-only fallback
        const VoxelCell& vc = it->second;
        if (vc.weight_eff() < 1e-12) return candidate;

        const double la = std::max(1e-12, s.lambda_p * s.vertex_smooth_anchor_scale);
        const double lc = std::max(1e-12, s.lambda_p * s.vertex_smooth_candidate_scale);
        Mat3 Areg = vc.A_eff() + (la + lc) * Mat3::Identity();
        Vec3 rhs = vc.b_eff() + la * old_pos + lc * candidate;
        Vec3 x = Areg.ldlt().solve(rhs);
        if (s.clamp_to_voxel) {
            Vec3 lo, hi;
            voxel_bounds(r.key, lo, hi);
            x = x.cwiseMax(lo).cwiseMin(hi);
        }
        return x;
    }

    bool qem_accept_smoothed_position(const VertexRecord& r,
                                      const Vec3& old_pos,
                                      const Vec3& new_pos) const {
        auto it = cells.find(r.key);
        if (it == cells.end()) return true; // generated/virtual-only fallback
        const VoxelCell& vc = it->second;
        if (vc.weight_eff() < 1e-12) return true;
        double ro = vc.residual_per_observation(old_pos);
        double rn = vc.residual_per_observation(new_pos);
        if (!std::isfinite(ro) || !std::isfinite(rn)) return false;
        double tol = s.vertex_smooth_residual_tol_factor * std::max(1e-6, s.range_precision);
        return rn <= ro + tol * tol;
    }

    int smooth_mesh_vertices(MeshData& mesh) const {
        if (!s.enable_vertex_smoothing || s.vertex_smooth_iters <= 0) return 0;
        const int N = (int)mesh.verts.size();
        if (N == 0 || mesh.faces.empty()) return 0;

        std::vector<std::vector<int>> adj(N);
        std::unordered_set<uint64_t> edges;
        edges.reserve(mesh.faces.size() * 3 + 1);
        for (const auto& f : mesh.faces) {
            if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
            add_unique_adj_edge(adj, edges, f.a, f.b);
            add_unique_adj_edge(adj, edges, f.b, f.c);
            add_unique_adj_edge(adj, edges, f.c, f.a);
        }

        int total_moved = 0;
        const double radius = std::max(1e-9, s.vertex_smooth_radius_factor * s.voxel_size);
        const double sigma_sp = std::max(1e-9, s.vertex_smooth_sigma_spatial_factor * s.voxel_size);
        const double sigma_n = std::max(1e-9, s.vertex_smooth_sigma_normal);
        const double sigma_plane = std::max(1e-9, s.vertex_smooth_sigma_plane_factor * s.voxel_size);
        const double max_move = std::max(0.0, s.vertex_smooth_max_move_factor * s.voxel_size);

        for (int iter = 0; iter < s.vertex_smooth_iters; ++iter) {
            std::vector<Vec3> old_pos(N);
            for (int i = 0; i < N; ++i) old_pos[i] = mesh.verts[i].position;

            std::vector<Vec3> new_pos = old_pos;
            std::vector<char> moved(N, 0);

            #ifdef HAS_OPENMP
            #pragma omp parallel for schedule(static, 256)
            #endif
            for (int i = 0; i < N; ++i) {
                const VertexRecord& ri = mesh.verts[i];
                if (!vertex_smoothing_candidate_ok(ri, (int)adj[i].size())) continue;
                Vec3 ni = ri.normal;
                if (!normalized_or_zero(ni)) continue;

                Vec3 acc = Vec3::Zero();
                double wsum = 0.0;
                for (int j : adj[i]) {
                    if (j < 0 || j >= N || j == i) continue;
                    const VertexRecord& rj = mesh.verts[j];
                    Vec3 nj = rj.normal;
                    if (!normalized_or_zero(nj)) continue;
                    double ndot = std::abs(ni.dot(nj));
                    if (ndot < s.vertex_smooth_normal_dot) continue;

                    const Vec3 dij = old_pos[j] - old_pos[i];
                    double dist = dij.norm();
                    if (dist < 1e-12 || dist > radius) continue;
                    if (!vertex_smooth_visibility_ok(old_pos[i], old_pos[j])) continue;

                    // Bilateral weights: spatial falloff, normal edge stop,
                    // and point-plane edge stop in both tangent planes.
                    double wsp = std::exp(-(dist * dist) / (2.0 * sigma_sp * sigma_sp));
                    double dn = 1.0 - ndot;
                    double wn = std::exp(-(dn * dn) / (2.0 * sigma_n * sigma_n));
                    double pi = std::abs(ni.dot(dij));
                    double pj = std::abs(nj.dot(-dij));
                    double pd = std::max(pi, pj);
                    double wp = std::exp(-(pd * pd) / (2.0 * sigma_plane * sigma_plane));
                    double wc = std::clamp(rj.normal_consistency, 0.05, 1.0);
                    double w = wsp * wn * wp * wc;
                    if (w <= 1e-12) continue;
                    acc.noalias() += w * old_pos[j];
                    wsum += w;
                }
                if (wsum <= 1e-12) continue;

                Vec3 avg = acc / wsum;
                Vec3 disp = avg - old_pos[i];
                if (s.vertex_smooth_normal_only) {
                    disp = ni * disp.dot(ni); // suppress tangential sliding/planarization
                }
                Vec3 candidate = old_pos[i] + disp;
                Vec3 projected = qem_project_smooth_candidate(ri, old_pos[i], candidate);

                Vec3 step = projected - old_pos[i];
                double step_len = step.norm();
                if (max_move > 0.0 && step_len > max_move) {
                    projected = old_pos[i] + (max_move / std::max(step_len, 1e-12)) * step;
                    step = projected - old_pos[i];
                    step_len = step.norm();
                }
                if (step_len < 1e-9) continue;
                if (!qem_accept_smoothed_position(ri, old_pos[i], projected)) continue;
                new_pos[i] = projected;
                moved[i] = 1;
            }

            int moved_iter = 0;
            for (int i = 0; i < N; ++i) {
                if (!moved[i]) continue;
                mesh.verts[i].position = new_pos[i];
                auto it = cells.find(mesh.verts[i].key);
                if (it != cells.end() && it->second.weight_eff() > 1e-12) {
                    mesh.verts[i].residual = it->second.residual_per_observation(new_pos[i]);
                }
                moved_iter++;
            }
            total_moved += moved_iter;
            if (moved_iter == 0) break;
        }
        return total_moved;
    }

    MeshData build_component_growth_mesh(int min_last_hit_scan = -1,
                                         int max_last_hit_scan = -1) const {
        MeshData mesh = build_corner_dc_plus_mesh(min_last_hit_scan, max_last_hit_scan);
        std::unordered_set<FaceKey, FaceKeyHash> face_set;
        face_set.reserve(mesh.faces.size() * 2 + 1024);
        for (const auto& f : mesh.faces) face_set.insert(sorted_face_key(f.a, f.b, f.c));
        if (!s.enable_component_growth)
            apply_component_growth_pass(mesh, face_set, true);
        return mesh;
    }

    // Pure mode dispatch: build a mesh from the current cells. Honors the
    // active_region_ restriction (set by the persistent incremental updater)
    // via build_vertex_table(), so the very same builders can produce either a
    // full mesh or a local patch. No persistence bookkeeping happens here.
    MeshData dispatch_mesh_builder(int min_last_hit_scan = -1,
                                   int max_last_hit_scan = -1) const {
        std::string mode = s.mesh_mode;
        for (char& ch : mode) ch = (char)std::tolower((unsigned char)ch);
        MeshData mesh;
        if (mode == "corner_dc_plus" || mode == "cornerdc_plus" ||
            mode == "corner_plus" || mode == "cdp") {
            mesh = build_corner_dc_plus_mesh(min_last_hit_scan, max_last_hit_scan);
        } else if (mode == "component_growth" || mode == "component_grow" || mode == "grow") {
            mesh = build_component_growth_mesh(min_last_hit_scan, max_last_hit_scan);
        } else if (mode == "corner_dc" || mode == "cornerdc" || mode == "corner") {
            mesh = build_corner_dc_mesh(min_last_hit_scan, max_last_hit_scan);
        } else if (mode == "dual" || mode == "dc") {
            mesh = build_grid_edge_dc_mesh(min_last_hit_scan, max_last_hit_scan);
        } else if (mode == "surface_net" || mode == "surfacenet") {
            // Older face-iteration surface-net builder. Kept for A/B
            // comparison with the grid-edge DC above; produces redundant
            // overlapping quads on tilted surfaces.
            mesh = build_open_dual_contour_mesh(min_last_hit_scan, max_last_hit_scan);
        } else if (mode == "hybrid") {
            // Prefer grid-edge DC where free/surface evidence is sufficient,
            // but fall back to the smooth local fan mesher if DC emits no faces.
            // This is useful during incremental debugging: a zero-face DC snapshot
            // no longer makes the viewer look broken.
            mesh = build_corner_dc_plus_mesh(min_last_hit_scan, max_last_hit_scan);
            if (mesh.faces.empty()) mesh = build_local_smooth_mesh(min_last_hit_scan, max_last_hit_scan);
        } else {
            mesh = build_local_smooth_mesh(min_last_hit_scan, max_last_hit_scan);
        }

        // corner_dc_plus / hybrid / component_growth already run the scaffold
        // fill and (when enabled) the component-growth pipeline internally via
        // build_corner_dc_plus_mesh. For the remaining modes (smooth, dual,
        // corner_dc, surface_net) append the same two passes here so inherited
        // (virtual) scaffold vertices get tiled AND --enable_component_growth
        // works on top of any base mesher, not just corner_dc_plus.
        const bool mode_runs_passes_internally =
            (mode == "corner_dc_plus" || mode == "cornerdc_plus" || mode == "corner_plus" || mode == "cdp" ||
             mode == "component_growth" || mode == "component_grow" || mode == "grow" || mode == "hybrid");
        if (!mode_runs_passes_internally && mesh.verts.size() >= 1) {
            std::unordered_set<FaceKey, FaceKeyHash> fset;
            rebuild_face_set_from_mesh(mesh, fset);
            if (hierarchical_scaffold_fill_enabled())
                apply_hierarchical_scaffold_fill_pass(mesh, fset);
            if (s.enable_component_growth) {
                // force=true: run the boundary QEM grow pass even though this base
                // mode did not invoke the refinement pipeline.
                apply_component_growth_pass(mesh, fset, true);
            }
        }
        return mesh;
    }

    // Halo (in voxels) to grow around the dirty region before a local re-mesh.
    // Derived from the largest triangle-edge factor of the active mesher so
    // that triangles straddling the dirty/clean seam can still find and bind to
    // their clean neighbors during the local rebuild.
    int persistent_mesh_halo() const {
        if (s.persistent_mesh_halo_voxels > 0) return s.persistent_mesh_halo_voxels;
        double f = std::max(std::max(s.mesh_max_edge_factor, s.dc_max_edge_factor),
                            std::max(s.cdp_max_edge_factor, s.component_growth_max_edge_factor));
        if (!std::isfinite(f) || f <= 0.0) f = 2.0;
        return std::max(2, (int)std::ceil(f) + 1);
    }

    // Is this voxel key still a valid surface vertex *right now*? Uses the
    // mesher's own emission gate so the persistent mesh can never disagree with
    // what a fresh build would produce. This is what retires faces over cells
    // that silently became FREE through ray-carving (miss-only cells are not
    // marked growth-dirty, so we cannot rely on the dirty set for deletion).
    bool persistent_vertex_key_alive(const VoxKey& k) const {
        auto it = cells.find(k);
        if (it == cells.end()) return false;
        const VoxelCell& c = it->second;
        if (c.label(s) == VoxelCell::Label::FREE) return false;
        if (!cell_vertex_export_ok(c)) return false;
        return true;
    }

    // PlanarMesh-style local re-triangulation with topology persistence.
    // Re-meshes only the changed voxels (dirty set) plus a halo, then splices
    // the result into the persistent mesh:
    //   * clean faces persist UNLESS contradicted (a vertex cell became FREE /
    //     invalid), giving PlanarMesh-like recall;
    //   * faces touching the dirty region are replaced by the freshly meshed
    //     ones, and dirty vertices are re-solved by QEM;
    //   * still-plausible dirty faces are retained as gap-fill where the local
    //     mesher produced nothing, so re-meshing never tears holes;
    //   * a manifold edge-incidence guard rejects any face that would push an
    //     edge above two incident faces, keeping seams 2-manifold.
    void update_persistent_mesh_dirty_region() const {
        // Nothing changed at all (no hits and no carving): skip entirely.
        if (dirty_component_voxel_keys.empty() && !mesh_free_carve_pending_) return;

        // Voxels whose geometry/topology may have changed this round.
        std::unordered_set<VoxKey, VoxHash> dirty = dirty_component_voxel_keys;

        // Grow the halo so seam triangles reconnect to clean neighbors.
        const int halo = persistent_mesh_halo();
        std::unordered_set<VoxKey, VoxHash> region;
        const size_t side = (size_t)(2 * halo + 1);
        region.reserve(dirty.size() * side * side * side);
        for (const VoxKey& k : dirty) {
            for (int dx = -halo; dx <= halo; ++dx)
            for (int dy = -halo; dy <= halo; ++dy)
            for (int dz = -halo; dz <= halo; ++dz)
                region.insert(VoxKey{(int32_t)(k.i + dx), (int32_t)(k.j + dy), (int32_t)(k.k + dz)});
        }

        // Re-mesh just the region using the unchanged QEM mesher.
        active_region_ = &region;
        MeshData local = dispatch_mesh_builder(-1, -1);
        active_region_ = nullptr;

        auto is_dirty = [&](const VoxKey& k) { return dirty.find(k) != dirty.end(); };

        MeshData merged;
        merged.verts.reserve(persistent_mesh_.verts.size() + local.verts.size());
        merged.faces.reserve(persistent_mesh_.faces.size() + local.faces.size());
        std::unordered_map<VoxKey, int, VoxHash> idx_by_key;
        idx_by_key.reserve((persistent_mesh_.verts.size() + local.verts.size()) * 2 + 16);
        auto get_or_add = [&](const VertexRecord& r) -> int {
            auto it = idx_by_key.find(r.key);
            if (it != idx_by_key.end()) return it->second;
            int ni = (int)merged.verts.size();
            merged.verts.push_back(r);
            idx_by_key[r.key] = ni;
            return ni;
        };

        // Cache aliveness once per key touched in this update.
        std::unordered_map<VoxKey, bool, VoxHash> alive_cache;
        alive_cache.reserve((persistent_mesh_.verts.size() + local.verts.size()) + 16);
        auto alive = [&](const VoxKey& k) -> bool {
            auto it = alive_cache.find(k);
            if (it != alive_cache.end()) return it->second;
            bool a = persistent_vertex_key_alive(k);
            alive_cache.emplace(k, a);
            return a;
        };

        // Vertices:
        //   * clean persistent vertices persist only if still alive (this drops
        //     ghost vertices over cells carved to FREE without being marked dirty);
        //   * dirty cells are re-solved from the local build;
        //   * any extra region vertices the local build introduced (scaffold /
        //     generated) are pulled in.
        for (const auto& r : persistent_mesh_.verts)
            if (!is_dirty(r.key) && alive(r.key)) get_or_add(r);
        for (const auto& r : local.verts)
            if (is_dirty(r.key)) get_or_add(r);
        for (const auto& r : local.verts)
            if (region.find(r.key) != region.end() && idx_by_key.find(r.key) == idx_by_key.end())
                get_or_add(r);

        std::unordered_set<FaceKey, FaceKeyHash> fset;
        fset.reserve((persistent_mesh_.faces.size() + local.faces.size()) * 2 + 16);
        // Manifold edge-incidence guard: never let an undirected edge carry more
        // than two faces. This keeps the dirty/clean seam 2-manifold even though
        // the retained and freshly meshed faces were triangulated independently.
        std::unordered_map<uint64_t, int> edge_use;
        edge_use.reserve((persistent_mesh_.faces.size() + local.faces.size()) * 3 + 16);
        auto add_face = [&](const VoxKey& ka, const VoxKey& kb, const VoxKey& kc) -> bool {
            auto ia = idx_by_key.find(ka), ib = idx_by_key.find(kb), ic = idx_by_key.find(kc);
            if (ia == idx_by_key.end() || ib == idx_by_key.end() || ic == idx_by_key.end()) return false;
            int a = ia->second, b = ib->second, c = ic->second;
            if (a == b || a == c || b == c) return false;
            FaceKey fk = sorted_face_key(a, b, c);
            if (fset.find(fk) != fset.end()) return false;            // duplicate triangle
            uint64_t e0 = edge_key(a, b), e1 = edge_key(b, c), e2 = edge_key(c, a);
            auto used = [&](uint64_t e) { auto it = edge_use.find(e); return it == edge_use.end() ? 0 : it->second; };
            if (used(e0) >= 2 || used(e1) >= 2 || used(e2) >= 2) return false; // would break 2-manifold
            fset.insert(fk);
            edge_use[e0]++; edge_use[e1]++; edge_use[e2]++;
            merged.faces.push_back({a, b, c});
            return true;
        };

        size_t kept = 0, remeshed = 0, retained_dirty = 0, retired = 0;
        // (1) Settled clean faces first (highest priority) — but retire any whose
        //     cell silently became FREE/invalid since they were created.
        for (const auto& f : persistent_mesh_.faces) {
            const VoxKey& ka = persistent_mesh_.verts[f.a].key;
            const VoxKey& kb = persistent_mesh_.verts[f.b].key;
            const VoxKey& kc = persistent_mesh_.verts[f.c].key;
            if (is_dirty(ka) || is_dirty(kb) || is_dirty(kc)) continue; // handled in passes (2)/(3)
            if (!alive(ka) || !alive(kb) || !alive(kc)) { retired++; continue; } // vertex cell contradicted
            if (s.persistent_retire_free_faces &&
                component_face_free_space_contradicted(persistent_mesh_, f)) { retired++; continue; } // interior crosses free space
            if (add_face(ka, kb, kc)) kept++;
        }
        // (2) Freshly meshed faces that touch the dirty region (authoritative for
        //     the changed area).
        for (const auto& f : local.faces) {
            const VoxKey& ka = local.verts[f.a].key;
            const VoxKey& kb = local.verts[f.b].key;
            const VoxKey& kc = local.verts[f.c].key;
            if (!(is_dirty(ka) || is_dirty(kb) || is_dirty(kc))) continue; // clean face already handled
            if (add_face(ka, kb, kc)) remeshed++;
        }
        // (3) Recall guard: retain still-plausible OLD dirty faces only where the
        //     local mesher left a gap. The manifold guard means these can only
        //     fill holes (they cannot create non-manifold seams), so re-meshing a
        //     region never tears the previously accepted surface.
        for (const auto& f : persistent_mesh_.faces) {
            const VoxKey& ka = persistent_mesh_.verts[f.a].key;
            const VoxKey& kb = persistent_mesh_.verts[f.b].key;
            const VoxKey& kc = persistent_mesh_.verts[f.c].key;
            if (!(is_dirty(ka) || is_dirty(kb) || is_dirty(kc))) continue; // only the dirty ones here
            if (!alive(ka) || !alive(kb) || !alive(kc)) { retired++; continue; } // vertex cell contradicted
            if (s.persistent_retire_free_faces &&
                component_face_free_space_contradicted(persistent_mesh_, f)) { retired++; continue; } // interior crosses free space
            if (add_face(ka, kb, kc)) retained_dirty++;
        }

        persistent_mesh_ = std::move(merged);
        persistent_mesh_update_count++;

        // Seam-aware hole closing across the persistent<->local boundary.
        size_t seam_closed = 0;
        if (s.enable_seam_closing) seam_closed = close_seam_holes(persistent_mesh_, region, dirty);

        std::printf("  [persistent_mesh] update #%ld: dirty=%zu region=%zu halo=%d "
                    "local(v=%zu,f=%zu) kept_clean=%zu remeshed=%zu retained_dirty=%zu retired=%zu "
                    "seam_closed=%zu -> mesh(v=%zu,f=%zu)\n",
                    persistent_mesh_update_count, dirty.size(), region.size(), halo,
                    local.verts.size(), local.faces.size(), kept, remeshed, retained_dirty, retired,
                    seam_closed, persistent_mesh_.verts.size(), persistent_mesh_.faces.size());

        dirty_component_voxel_keys.clear();
        mesh_free_carve_pending_ = false;
    }

    // PlanarMesh-style angular hole closing restricted to the persistent<->local
    // seam. For each boundary vertex whose voxel key lies in the freshly meshed
    // region, gather its boundary-edge neighbors, project them into the vertex
    // tangent plane, angle-sort them, and close every sufficiently small angular
    // gap by adding the triangle (v, n_i, n_{i+1}) -- synthesizing the bridging
    // edge n_i--n_{i+1}. A manifold edge-incidence guard means a gap is only
    // closed when both of its edges still have a free face slot, so this can fill
    // holes but never create a non-manifold seam. Cascaded over a few iterations
    // because closing one triangle can expose the next gap.
    //
    // To avoid patching ordinary interior holes inside the local remesh, a gap
    // is only closed when its triangle straddles the actual seam: it must mix at
    // least one vertex from a freshly re-meshed (dirty) cell with at least one
    // from the retained clean mesh. Triangles whose interior crosses observed
    // free space are vetoed as well.
    size_t close_seam_holes(MeshData& mesh,
                            const std::unordered_set<VoxKey, VoxHash>& region,
                            const std::unordered_set<VoxKey, VoxHash>& dirty) const {
        if (mesh.verts.size() < 3 || mesh.faces.empty()) return 0;
        auto is_dirty = [&](const VoxKey& k) { return dirty.find(k) != dirty.end(); };
        const double max_edge = std::max(s.mesh_max_edge_factor, s.cdp_max_edge_factor) * s.voxel_size;
        const double max_fan_angle = std::clamp(s.mesh_max_fan_angle, 0.1, 6.283185307179586);
        const int iters = std::max(1, s.seam_close_iters);

        size_t total_added = 0;
        for (int iter = 0; iter < iters; ++iter) {
            const int N = (int)mesh.verts.size();

            // Edge incidence + current triangle set.
            std::unordered_map<uint64_t, int> edge_count;
            edge_count.reserve(mesh.faces.size() * 3 + 1);
            std::unordered_set<FaceKey, FaceKeyHash> face_set;
            face_set.reserve(mesh.faces.size() * 2 + 16);
            for (const auto& f : mesh.faces) {
                if (f.a < 0 || f.b < 0 || f.c < 0 || f.a >= N || f.b >= N || f.c >= N) continue;
                edge_count[edge_key(f.a, f.b)]++;
                edge_count[edge_key(f.b, f.c)]++;
                edge_count[edge_key(f.c, f.a)]++;
                face_set.insert(sorted_face_key(f.a, f.b, f.c));
            }

            // Boundary neighbors: endpoints joined by a once-used (boundary) edge.
            std::unordered_map<int, std::vector<int>> bnbr;
            for (const auto& kv : edge_count) {
                if (kv.second != 1) continue;
                int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
                if (a < 0 || b < 0 || a >= N || b >= N) continue;
                bnbr[a].push_back(b);
                bnbr[b].push_back(a);
            }
            if (bnbr.empty()) break;

            size_t added_iter = 0;
            for (auto& kv : bnbr) {
                int vi = kv.first;
                // Only work the seam: the freshly re-meshed neighborhood.
                if (region.find(mesh.verts[vi].key) == region.end()) continue;
                std::vector<int>& nbrs = kv.second;
                if (nbrs.size() < 2) continue;

                Vec3 nv = mesh.verts[vi].normal;
                if (!normalized_or_zero(nv)) continue;
                Vec3 ref = (std::abs(nv.z()) < 0.9) ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
                Vec3 u = ref.cross(nv); if (!normalized_or_zero(u)) continue;
                Vec3 w = nv.cross(u);   if (!normalized_or_zero(w)) continue;

                struct AN { double ang; int idx; };
                std::vector<AN> ring;
                ring.reserve(nbrs.size());
                for (int j : nbrs) {
                    Vec3 d = mesh.verts[j].position - mesh.verts[vi].position;
                    double x = d.dot(u), y = d.dot(w);
                    if (x * x + y * y < 1e-18) continue;
                    ring.push_back({std::atan2(y, x), j});
                }
                if (ring.size() < 2) continue;
                std::sort(ring.begin(), ring.end(), [](const AN& a, const AN& b){ return a.ang < b.ang; });

                int K = (int)ring.size();
                for (int t = 0; t < K; ++t) {
                    int j = ring[t].idx;
                    int k = ring[(t + 1) % K].idx;
                    if (j == k || j == vi || k == vi) continue;
                    double gap = ring[(t + 1) % K].ang - ring[t].ang;
                    if (t + 1 == K) gap += 6.283185307179586;
                    if (gap > max_fan_angle) continue;

                    // Seam-only: the triangle must straddle the dirty/clean
                    // boundary -- at least one re-meshed (dirty) vertex AND at
                    // least one retained clean vertex -- otherwise it is an
                    // ordinary interior hole the local mesher already judged.
                    {
                        bool any_dirty = is_dirty(mesh.verts[vi].key) || is_dirty(mesh.verts[j].key) || is_dirty(mesh.verts[k].key);
                        bool any_clean = !is_dirty(mesh.verts[vi].key) || !is_dirty(mesh.verts[j].key) || !is_dirty(mesh.verts[k].key);
                        if (!(any_dirty && any_clean)) continue;
                    }

                    // Geometry / QEM gates (reuse the smooth mesher's triangle test).
                    if ((mesh.verts[vi].position - mesh.verts[j].position).norm() > max_edge) continue;
                    if ((mesh.verts[vi].position - mesh.verts[k].position).norm() > max_edge) continue;
                    if ((mesh.verts[j].position  - mesh.verts[k].position).norm() > max_edge) continue;
                    if (!mesh_triangle_ok(mesh.verts[vi], mesh.verts[j], mesh.verts[k])) continue;

                    // Do not bridge across observed free space.
                    if (s.persistent_retire_free_faces &&
                        component_face_free_space_contradicted(mesh, MeshFace{vi, j, k})) continue;

                    // Manifold guard: only close if both bridged edges have a slot.
                    if (!triangle_edges_can_accept(edge_count, vi, j, k)) continue;

                    FaceKey fk = sorted_face_key(vi, j, k);
                    if (face_set.find(fk) != face_set.end()) continue;

                    // Orient against the averaged vertex normal.
                    int fa = vi, fb = j, fc = k;
                    Vec3 tri_n = (mesh.verts[fb].position - mesh.verts[fa].position)
                               .cross(mesh.verts[fc].position - mesh.verts[fa].position);
                    if (normalized_or_zero(nv) && tri_n.dot(nv) < 0.0) std::swap(fb, fc);

                    mesh.faces.push_back({fa, fb, fc});
                    face_set.insert(fk);
                    register_triangle_edges(edge_count, fa, fb, fc);
                    added_iter++;
                }
            }
            total_added += added_iter;
            if (added_iter == 0) break;
        }
        return total_added;
    }

    MeshData build_mesh_for_mode(int min_last_hit_scan = -1,
                                 int max_last_hit_scan = -1) const {
        // Legacy one-shot rebuild path. Retained only as an internal fallback;
        // the persistent incremental mesher below is the default behavior.
        if (!s.persistent_incremental_mesh) {
            active_region_ = nullptr;
            MeshData mesh = dispatch_mesh_builder(min_last_hit_scan, max_last_hit_scan);
            if (s.component_growth_persistent_state && s.component_persistent_reuse_faces) {
                std::unordered_set<FaceKey, FaceKeyHash> reuse_face_set;
                rebuild_face_set_from_mesh(mesh, reuse_face_set);
                size_t reused = emit_retained_persistent_faces(mesh, reuse_face_set, "build_mesh");
                if (reused > 0) {
                    apply_component_fis_delete_pass(mesh, reuse_face_set, "post_reuse");
                    apply_component_radius_shrink_pass(mesh, reuse_face_set, "post_reuse");
                    if (s.component_compact_after_topology_ops) compact_mesh_vertices(mesh, reuse_face_set);
                }
            }
            refresh_persistent_components_from_mesh(mesh, "build_mesh", true);
            return mesh;
        }

        // Persistent incremental mesh: one mesh kept alive across scans.
        if (min_last_hit_scan >= 0 || max_last_hit_scan >= 0) {
            std::printf("  [persistent_mesh] note: last-hit-scan window [%d,%d] ignored; "
                        "exporting the single persistent mesh.\n",
                        min_last_hit_scan, max_last_hit_scan);
        }

        if (!persistent_mesh_initialized_) {
            // First export: build the whole mesh once, then maintain it.
            active_region_ = nullptr;
            persistent_mesh_ = dispatch_mesh_builder(-1, -1);
            persistent_mesh_initialized_ = true;
            dirty_component_voxel_keys.clear();
            mesh_free_carve_pending_ = false;
            refresh_persistent_components_from_mesh(persistent_mesh_, "persistent_full_build", false);
            std::printf("  [persistent_mesh] full build: %zu verts %zu faces\n",
                        persistent_mesh_.verts.size(), persistent_mesh_.faces.size());
            return persistent_mesh_;
        }

        update_persistent_mesh_dirty_region();
        refresh_persistent_components_from_mesh(persistent_mesh_, "persistent_incremental", false);
        return persistent_mesh_;
    }

    void export_mesh_ply(const std::string& mesh_path,
                         int min_last_hit_scan = -1,
                         int max_last_hit_scan = -1) const {
        double t0 = now_sec();
        MeshData mesh = build_mesh_for_mode(min_last_hit_scan, max_last_hit_scan);
        // Vertex smoothing is an OUTPUT-ONLY cosmetic pass: it runs on this
        // returned copy and is deliberately NOT written back into persistent_mesh_.
        // Therefore we must NOT refresh persistent component state from the
        // smoothed geometry, or component planes/boundaries would drift away
        // from the canonical (unsmoothed) persistent mesh. Persistent component
        // state is already refreshed inside build_mesh_for_mode() from the
        // canonical mesh.
        int smoothed_vertices = smooth_mesh_vertices(mesh);

        // Write atomically: the live browser may poll while this file is being
        // produced. Writing to a temporary path and then renaming prevents the
        // viewer from loading a partial PLY header/body.
        const std::string tmp_path = mesh_path + ".tmp_" + mesh_run_id;
        {
            std::ofstream out(tmp_path);
            if (!out) {
                std::fprintf(stderr, "Cannot write mesh temp file %s\n", tmp_path.c_str());
                return;
            }
            out << "ply\nformat ascii 1.0\n"
                << "comment mesh_mode " << s.mesh_mode << "\n"
                << "comment vertex_smoothing " << (s.enable_vertex_smoothing ? "on" : "off")
                << " moved " << smoothed_vertices
                << " iters " << s.vertex_smooth_iters
                << " normal_only " << (s.vertex_smooth_normal_only ? "yes" : "no")
                << " qem_project " << (s.vertex_smooth_qem_project ? "yes" : "no") << "\n"
                << "comment open-scene mesh from voxel-QEM vertices\n"
                << "comment voxel_size " << s.voxel_size << "\n"
                << "comment scans " << scan_count << "\n"
                << "comment mesh_run_id " << mesh_run_id << "\n";
            if (min_last_hit_scan >= 0 || max_last_hit_scan >= 0) {
                out << "comment last_hit_scan_window " << min_last_hit_scan << " " << max_last_hit_scan << "\n";
            }
            out << "element vertex " << mesh.verts.size() << "\n"
                << "property float x\nproperty float y\nproperty float z\n"
                << "property float nx\nproperty float ny\nproperty float nz\n"
                << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                << "property float residual\n"
                << "property float consistency\n"
                << "property int last_scan\n"
                << "property uchar kind_id\n"
                << "property float scan_boundary\n"
                << "property float eogm_bel_surface\n"
                << "property float eogm_bel_free\n"
                << "property float eogm_conflict\n"
                << "property uchar confidence_tier\n"
                << "element face " << mesh.faces.size() << "\n"
                << "property list uchar int vertex_indices\n"
                << "end_header\n";
            for (const auto& r : mesh.verts) {
                int rr, gg, bb, kid;
                rgb_for_kind(r.kind, rr, gg, bb, kid);
                out << r.position[0] << ' ' << r.position[1] << ' ' << r.position[2]
                    << ' ' << r.normal[0] << ' ' << r.normal[1] << ' ' << r.normal[2]
                    << ' ' << rr << ' ' << gg << ' ' << bb
                    << ' ' << r.residual
                    << ' ' << r.normal_consistency
                    << ' ' << r.last_hit_scan
                    << ' ' << kid
                    << ' ' << r.scan_boundary_ratio
                    << ' ' << r.eogm_bel_surface
                    << ' ' << r.eogm_bel_free
                    << ' ' << r.eogm_conflict
                    << ' ' << r.confidence_tier << '\n';
            }
            for (const auto& f : mesh.faces) {
                out << "3 " << f.a << ' ' << f.b << ' ' << f.c << '\n';
            }
        }

        std::error_code ec;
        fs::rename(tmp_path, mesh_path, ec);
        if (ec) {
            std::error_code rm_ec;
            fs::remove(mesh_path, rm_ec);
            ec.clear();
            fs::rename(tmp_path, mesh_path, ec);
        }
        if (ec) {
            std::fprintf(stderr, "Cannot atomically publish mesh %s from %s: %s\n",
                         mesh_path.c_str(), tmp_path.c_str(), ec.message().c_str());
            return;
        }

        double dt = now_sec() - t0;
        std::printf("Exported mesh: %zu vertices, %zu faces -> %s  %.2fs  vertex_smooth_moved=%d\n",
                    mesh.verts.size(), mesh.faces.size(), mesh_path.c_str(), dt, smoothed_vertices);
        if (mesh.faces.empty()) {
            std::fprintf(stderr,
                "  [MESH WARN] mesh has 0 faces. If using --mesh_mode dual with sparse/free-space data, try "
                "--mesh_mode hybrid or --mesh_mode smooth, or add --dc_no_require_free when running with --no_carve.\n");
        }
    }

    void maybe_dump_mesh_snapshot(int scan_idx) const {
        if (s.mesh_every <= 0) return;
        std::string dir = !s.mesh_dump_dir.empty() ? s.mesh_dump_dir : s.dump_dir;
        if (dir.empty()) return;
        if ((scan_idx + 1) % std::max(1, s.mesh_every) != 0) return;
        std::error_code ec;
        fs::create_directories(dir, ec);
        int lo = -1, hi = -1;
        if (s.mesh_submap_scans > 0) {
            lo = std::max(0, scan_idx - s.mesh_submap_scans + 1);
            hi = scan_idx;
        }
        char name[256];
        // Include a per-process run id so old snapshots from previous runs are
        // not overwritten and browser/live polling cannot confuse stale files
        // with new ones. The viewer understands both legacy and unique names.
        if (s.mesh_submap_scans > 0)
            std::snprintf(name, sizeof(name), "mesh_submap_%05d_%05d_%s.ply", lo, hi, mesh_run_id.c_str());
        else
            std::snprintf(name, sizeof(name), "mesh_%05d_%s.ply", scan_idx, mesh_run_id.c_str());
        export_mesh_ply(dir + "/" + name, lo, hi);
    }

    // ---- Visualization dump (per-scan PLY snapshots + manifest) ---- //
    //
    // Writes one PLY per kept scan into dump_dir. The PLY contains:
    //   - position (xyz)
    //   - mean normal (nx/ny/nz)
    //   - RGB pre-coloured by classification (flat / edge / corner) so any
    //     standard PLY viewer renders something sensible
    //   - custom properties for the JS viewer:
    //       residual    (float)  per-observation QEM residual at v*
    //       consistency (float)  normal_consistency, low = multi-modal
    //       last_scan   (int)    last HIT scan for this vertex (backward-compatible property name)
    //       kind_id     (uchar)  0=flat, 1=edge, 2=corner
    //       scan_boundary (float) fraction of hit weight marked by scan-line depth jumps
    //
    // manifest.json is rewritten on every call so the viewer can poll it
    // and visualise a still-running session in "live" mode.

    static void rgb_for_kind(const char* kind, int& r, int& g, int& b, int& kid) {
        if      (std::strcmp(kind, "edge")   == 0) { r=240; g=180; b= 30; kid=1; }
        else if (std::strcmp(kind, "corner") == 0) { r=230; g= 50; b= 50; kid=2; }
        else if (std::strcmp(kind, "generated") == 0) { r=160; g= 80; b=255; kid=4; }
        else                                       { r=200; g=200; b=200; kid=0; }
    }

    void write_manifest() {
        std::string path = s.dump_dir + "/manifest.json";
        std::ofstream out(path);
        out << "{\n";
        out << "  \"voxel_size\": " << s.voxel_size << ",\n";
        out << "  \"n_scans_seen\": " << scan_count << ",\n";
        if (s.dump_roi_radius > 0) {
            out << "  \"roi\": {\"c\": ["
                << s.dump_roi_center[0] << ", "
                << s.dump_roi_center[1] << ", "
                << s.dump_roi_center[2] << "], \"r\": "
                << s.dump_roi_radius << "},\n";
        } else {
            out << "  \"roi\": null,\n";
        }
        out << "  \"scans\": [";
        for (size_t i = 0; i < dump_manifest.size(); i++) {
            const auto& d = dump_manifest[i];
            if (i > 0) out << ",";
            out << "\n    {\"scan\": " << d.scan_idx
                << ", \"ply\": \"" << d.ply << "\""
                << ", \"n_vertices\": " << d.n_vertices
                << ", \"n_new\": " << d.n_new_this_scan
                << ", \"dump_sec\": " << d.dump_time_sec << "}";
        }
        out << "\n  ]\n}\n";
    }

    void dump_scan_viz(int scan_idx) {
        if (!dump_active()) return;
        // Throttle by dump_every. Counter advances on every call so the
        // schedule is over scans, not over kept dumps.
        bool keep = (dump_pass_count % std::max(1, s.dump_every)) == 0;
        dump_pass_count++;
        if (!keep) return;

        double t0 = now_sec();
        std::error_code ec;
        fs::create_directories(s.dump_dir, ec);

        // Build current vertex table, filter by ROI, count new-this-scan.
        auto recs = build_vertex_table();
        std::vector<VertexRecord> kept;
        kept.reserve(recs.size());
        int n_new = 0;
        for (auto& r : recs) {
            if (!in_roi(r.position)) continue;
            kept.push_back(r);
            auto it = cells.find(r.key);
            // "New this scan" uses last_hit_scan -- a cell that only saw
            // misses this scan is not a new vertex.
            if (it != cells.end() && it->second.last_hit_scan == scan_idx) n_new++;
        }
        if ((int)kept.size() > s.dump_max_verts) {
            kept.resize(s.dump_max_verts);
        }

        char ply_name[64];
        std::snprintf(ply_name, sizeof(ply_name), "scan_%05d.ply", scan_idx);
        std::string ply_path = s.dump_dir + "/" + ply_name;
        {
            std::ofstream out(ply_path);
            int M = (int)kept.size();
            out << "ply\n"
                << "format ascii 1.0\n"
                << "comment qem_voxel_map snapshot scan=" << scan_idx << "\n"
                << "comment voxel_size " << s.voxel_size << "\n"
                << "element vertex " << M << "\n"
                << "property float x\nproperty float y\nproperty float z\n"
                << "property float nx\nproperty float ny\nproperty float nz\n"
                << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                << "property float residual\n"
                << "property float consistency\n"
                << "property int last_scan\n"
                << "property uchar kind_id\n"
                << "property float scan_boundary\n"
                << "property float eogm_bel_surface\n"
                << "property float eogm_bel_free\n"
                << "property float eogm_conflict\n"
                << "property uchar confidence_tier\n"
                << "end_header\n";
            for (auto& r : kept) {
                int rr, gg, bb, kid;
                rgb_for_kind(r.kind, rr, gg, bb, kid);
                auto it = cells.find(r.key);
                // PLY column "last_scan" carries the last HIT scan (vertex
                // age), not the last evidence scan. The property name stays
                // "last_scan" for backward compatibility with the viewer.
                int last_scan = (it != cells.end()) ? it->second.last_hit_scan : -1;
                out << r.position[0] << ' ' << r.position[1] << ' ' << r.position[2]
                    << ' ' << r.normal[0] << ' ' << r.normal[1] << ' ' << r.normal[2]
                    << ' ' << rr << ' ' << gg << ' ' << bb
                    << ' ' << r.residual
                    << ' ' << r.normal_consistency
                    << ' ' << last_scan
                    << ' ' << kid
                    << ' ' << r.scan_boundary_ratio
                    << ' ' << r.confidence_tier
                    << '\n';
            }
        }

        double dt = now_sec() - t0;
        dump_manifest.push_back({scan_idx, ply_name, (int)kept.size(), n_new, dt});
        write_manifest();
        std::printf("    [VIZ] scan %05d -> %s  verts=%d  new=%d  %.2fs\n",
                    scan_idx, ply_name, (int)kept.size(), n_new, dt);
    }

    // ---- Summary printing ----

    void print_summary() const {
        long n_surface = 0, n_free = 0, n_unknown = 0;
        long flat_cells = 0, edge_cells = 0, corner_cells = 0;
        long promoted_seed_cells = 0;
        long promoted_seed_observations = 0;
        long weak_promoted_vertices = 0;
        long parent_supported_vertices = 0;
        long inherited_vertices = 0;
        long inherited_with_real_qem = 0;
        for (auto& [k, v] : cells) {
            if (v.promoted_seed_count > 0) {
                promoted_seed_cells++;
                promoted_seed_observations += v.promoted_seed_count;
            }
            int tier = cell_vertex_tier(v);
            if (tier == 1) weak_promoted_vertices++;
            else if (tier == 2) parent_supported_vertices++;
            else if (tier == 6) inherited_vertices++;
            if (v.has_inherited() && v.weight_sum > 1e-12) inherited_with_real_qem++;
            auto lbl = v.label(s);
            if (lbl == VoxelCell::Label::SURFACE) {
                n_surface++;
                if (v.hit_count >= s.min_hit_count_vertex) {
                    auto info = v.eigen_analysis(s);
                    if (std::strcmp(info.kind, "flat")   == 0) flat_cells++;
                    else if (std::strcmp(info.kind, "edge")   == 0) edge_cells++;
                    else if (std::strcmp(info.kind, "corner") == 0) corner_cells++;
                }
            } else if (lbl == VoxelCell::Label::FREE) {
                n_free++;
            } else {
                n_unknown++;
            }
        }
        std::printf("\n%s\n", std::string(60, '=').c_str());
        std::printf("Voxel QEM map summary\n");
        std::printf("  voxel_size = %.3f  | %d scans processed\n",
                    s.voxel_size, scan_count);
        std::printf("  cells: %zu total | surface=%ld free=%ld unknown=%ld\n",
                    cells.size(), n_surface, n_free, n_unknown);
        std::printf("  surface cells with hit_count >= %d: flat=%ld edge=%ld corner=%ld\n",
                    s.min_hit_count_vertex, flat_cells, edge_cells, corner_cells);
        std::printf("  seed layer: pending_cells=%zu l1_parents=%zu promoted_cells=%ld promoted_obs=%ld\n",
                    seed_cells.size(), seed_l1_cells.size(), promoted_seed_cells, promoted_seed_observations);
        long high_free_surface = 0, high_conflict_surface = 0;
        for (auto& [kk, vv] : cells) {
            if (vv.label(s) == VoxelCell::Label::SURFACE) {
                if (vv.eogm_bel_free() > s.eogm_mesh_max_bel_free) high_free_surface++;
                if (vv.eogm_conflict_mass() > s.eogm_mesh_max_conflict) high_conflict_surface++;
            }
        }
        std::printf("  vertex tiers: weak_promoted=%ld parent_supported=%ld inherited=%ld inherited_with_real_qem=%ld\n",
                    weak_promoted_vertices, parent_supported_vertices,
                    inherited_vertices, inherited_with_real_qem);
        std::printf("  EOGM risk among surface cells: high_free=%ld high_conflict=%ld\n",
                    high_free_surface, high_conflict_surface);
        std::printf("  totals: hits=%ld seed_hits=%ld promoted_seed_cells_total=%ld l1_parent_supported_total=%ld misses=%ld grazing_rejected=%ld\n",
                    n_hits_total, n_seed_hits_total, n_seed_promoted_total, n_seed_l1_parent_supported_total, n_misses_total, n_grazing_total);
        std::printf("%s\n", std::string(60, '=').c_str());
    }
};

// ========================================================================= //
// 10. argv helpers (robust against missing values / bad numerics)            //
// ========================================================================= //

namespace argv_util {
    // Same robust-arg pattern qem_mesh.cpp uses: a missing value or a
    // numeric-conversion failure prints a meaningful error and exits with
    // code 2, rather than crashing with an uncaught std::invalid_argument.
    static int    g_i    = 1;
    static int    g_argc = 0;
    static char** g_argv = nullptr;

    void init(int argc, char** argv) { g_argc = argc; g_argv = argv; g_i = 1; }

    std::string next_raw(const std::string& flag) {
        if (g_i + 1 >= g_argc) {
            std::fprintf(stderr, "Error: missing value for %s\n", flag.c_str());
            std::exit(2);
        }
        std::string v = g_argv[++g_i];
        if (v.size() >= 2 && v[0] == '-' && v[1] == '-') {
            std::fprintf(stderr, "Error: %s expected a value but got '%s' "
                                  "(looks like another flag)\n",
                         flag.c_str(), v.c_str());
            std::exit(2);
        }
        return v;
    }
    double next_double(const std::string& flag) {
        std::string v = next_raw(flag);
        try { return std::stod(v); }
        catch (const std::exception&) {
            std::fprintf(stderr, "Error: bad numeric value '%s' for %s\n",
                         v.c_str(), flag.c_str());
            std::exit(2);
        }
    }
    int next_int(const std::string& flag) {
        std::string v = next_raw(flag);
        try { return std::stoi(v); }
        catch (const std::exception&) {
            std::fprintf(stderr, "Error: bad integer value '%s' for %s\n",
                         v.c_str(), flag.c_str());
            std::exit(2);
        }
    }
}  // namespace argv_util

// ========================================================================= //
// 11. SIGINT handling: save partial map on Ctrl-C                            //
// ========================================================================= //

namespace {
    VoxelQEMMap* g_map = nullptr;
    std::string  g_output;
    double       g_t0 = 0.0;
    volatile bool g_interrupted = false;
}

static void on_sigint(int) {
    if (g_interrupted) {
        std::printf("\nForce quit.\n");
        std::_Exit(1);
    }
    g_interrupted = true;
    std::printf("\n[SIGINT] Saving map...\n");
    if (g_map) {
        g_map->print_summary();
        g_map->export_ply_and_csv(g_output);
    }
    std::printf("Total: %.1fs\n", now_sec() - g_t0);
    std::_Exit(0);
}

// ========================================================================= //
// 12. Usage                                                                  //
// ========================================================================= //

static void print_usage() {
    std::cerr <<
"Usage: qem_voxel_map --pcd_folder DIR --pose_file FILE [options]\n\n"
"  Core inputs:\n"
"    --pcd_folder DIR         folder of *.pcd files (binary or ASCII)\n"
"    --pose_file FILE         poses; format inferred from extension:\n"
"                                .tum    -> TUM   (default)\n"
"                                .g2o    -> G2O\n"
"                                .slam   -> G2O\n"
"                                .csv    -> CSV\n"
"    --output PATH            output .ply path (default: map.ply)\n"
"                             metadata is written to <stem>_meta.csv\n\n"
"  Discretisation & weighting:\n"
"    --voxel_size F           voxel edge length, metres (default 0.1)\n"
"    --range_precision F      sigma_range, metres (default 0.015)\n"
"    --disable_prob_planes   disable probabilistic plane covariance/gates\n"
"    --bearing_sigma_rad F   LiDAR bearing std-dev in radians (default 0.0015)\n"
"    --pose_trans_sigma F    fallback pose translation std-dev, metres (default 0.02)\n"
"    --pose_rot_sigma_rad F  fallback pose rotation std-dev, radians (default 0.002)\n"
"    --prob_gate_sigma F     adaptive gate width in sigma units (default 3.0)\n"
"    --prob_qem_ref_sigma F  reference sigma for inverse-variance QEM weights\n"
"    --prob_min_sigma F      lower clamp for probabilistic sigma, metres\n"
"    --prob_max_sigma F      upper clamp for probabilistic sigma, metres\n"
"    --min_incident_cos F     grazing threshold |n.dir| (default 0.30)\n"
"    --process_every_n N      input subsampling factor (default 2)\n"
"    --estimate_normals_k N   k for PCA fallback (default 20)\n"
"    --disable_nvt           disable NVT/BEO normal denoising\n"
"    --nvt_all_normals       run NVT even when PCD already supplies normals\n"
"    --nvt_k N               neighbors for NVT voting (default 16)\n"
"    --nvt_iters N           NVT/BEO iterations (default 1; >1 may smooth edges)\n"
"    --nvt_rho_cos F         normal agreement gate (default 0.85)\n"
"    --nvt_tau F             BEO eigenvalue threshold (default 0.15)\n"
"    --nvt_damping F         damping d in d*n + T*n (default 3.0)\n"
"    --nvt_min_conf F        min raw normal confidence (default 0.10)\n"
"    --nvt_conf_soft_floor F final QEM multiplier floor (default 0.50)\n"
"    --nvt_max_radius F      max NVT voting radius, metres (default 5*voxel_size)\n"
"    --disable_scanline      disable scan-line/range-image gates\n"
"    --disable_scanline_normals  do not use scan-line normals before PCA\n"
"    --scanline_rows N       fallback elevation rows if no organized/ring field\n"
"    --scanline_cols N       azimuth columns for ring/elevation fallback\n"
"    --scanline_gate_rows N  row window for depth-jump gating (default 1)\n"
"    --scanline_gate_cols N  column window for depth-jump gating (default 2)\n"
"    --scanline_jump_abs F   depth jump abs threshold, metres (default 0.30)\n"
"    --scanline_jump_rel F   depth jump relative threshold (default 0.03)\n"
"    --scanline_boundary_conf F  QEM weight multiplier at depth jumps (default 0.65)\n"
"    --sensor pandar_qt64    enable PandarQT64 pseudo-ring geometry for unorganized scans\n"
"    --pandar_qt64           same as --sensor pandar_qt64\n"
"    --pandar_qt64_calib P   optional unit angle correction file: channel,elevation[,azimuth]\n"
"    --pandar_qt64_uniform   use uniform vertical fallback instead of design channel table\n"
"    --pandar_qt64_infer     infer 64 elevation clusters from each scan instead of design table\n\n"
"  Weak/seed hypothesis voxels:\n"
"    --disable_seed_voxels   drop weak/grazing observations as before\n"
"    --seed_min_incident_cos F  minimum incidence to keep a weak endpoint (default 0.08)\n"
"    --seed_hit_weight_scale F  weak endpoint QEM weight scale (default 0.25)\n"
"    --seed_min_hit_weight F    weak endpoint QEM weight floor (default 0.005)\n"
"    --seed_carve_rays       allow weak endpoints to carve misses, scaled by --seed_miss_weight_scale\n"
"    --seed_miss_weight_scale F weak miss scale when seed_carve_rays is on (default 0.05)\n"
"    --no_seed_ray_normal_fallback keep ray-normal fallback points in the main map\n"
"    --seed_export_min_hits N pending seed PLY export threshold (default 1)\n"
"    --seed_promote_min_hits N self-evidence hits to promote seed cell (default 3)\n"
"    --seed_promote_neighbor_min_hits N hits for neighbor-supported promotion (default 1)\n"
"    --seed_promote_min_neighbors N confirmed neighbors needed for promotion (default 2)\n"
"    --seed_promote_min_consistency F min normal consistency for seed promotion (default 0.70)\n"
"    --seed_promote_max_sqrt_residual F max seed sqrt residual in metres (default 0.10)\n"
"    --seed_promote_neighbor_normal_dot F neighbor normal gate (default 0.85)\n"
"    --seed_promote_neighbor_plane_dist_factor F plane distance / voxel_size (default 1.5)\n"
"    --seed_max_ray_normal_fraction F max ray-normal fallback fraction for ordinary L0 seed\n"
"                              maturation (default 0.50; L1 has a separate stricter gate)\n"
"    --seed_promote_cluster_min_size N same-resolution seed cluster size (default 3)\n"
"    --seed_promote_cluster_radius_voxels N cluster search radius in L0 voxels (default 1)\n"
"    --seed_promote_cluster_normal_dot F cluster normal compatibility (default 0.85)\n"
"    --seed_promote_cluster_plane_dist_factor F cluster plane distance / voxel_size (default 1.0)\n"
"    --seed_promote_cluster_merged_max_sqrt_residual F merged cluster residual, m (default 0.08)\n"
"    --no_export_seed_vertices do not write <output>_seeds.ply/csv\n"
"    --disable_seed_l1_parents disable 0.20m parent-supported seed validation\n"
"    --seed_l1_min_unique_scans N scans contributing to L1 parent (default 2)\n"
"    --seed_l1_min_children N direct 0.10m children in L1 parent (default 2)\n"
"    --seed_l1_min_consistency F parent normal consistency (default 0.70)\n"
"    --seed_l1_max_sqrt_residual F parent merged-QEM sqrt residual, m (default 0.08)\n"
"    --seed_l1_max_ray_normal_fraction F max ray-normal fallback fraction (default 0.35)\n"
"    --seed_l1_child_plane_dist F child-to-parent tangent plane distance, m (default 0.10)\n"
"    --seed_l1_child_normal_dot F child/parent abs normal dot (default 0.80)\n"
"    --seed_l1_child_min_occupancy F block child in carved-free territory (default -0.35)\n"
"    --no_export_weak_vertices do not export weak/promoted vertices in main map PLY/CSV\n"
"    --cdp_no_weak_vertices keep corner_dc_plus from using weak tiers in adaptive fill\n"
"    --cdp_allow_weak_only_triangles allow adaptive triangles made entirely of weak vertices\n\n"
"  Hierarchical scaffold / mesh-only fill:\n"
"    --enable_hierarchical_scaffold enable L1/L2/... parent-QEM virtual support and mesh-only fill\n"
"    --enable_planar_scaffold alias for hierarchical mesh-only scaffold fill\n"
"    --enable_planar_scaffold_inherit alias for --enable_hierarchical_scaffold\n"
"    --disable_hierarchical_scaffold_fill disable mesh-only scaffold tiling while keeping inheritance\n"
"    --enable_hierarchical_scaffold_fill enable mesh-only scaffold tiling\n"
"    --scaffold_base_factor N hierarchy base factor (default 2: L1=2x, L2=4x)\n"
"    --scaffold_max_level N max scaffold level above L0 (default 2)\n"
"    --scaffold_min_parent_children N base real L0 children required per parent\n"
"    --scaffold_min_parent_children_per_level N extra support per coarser level\n"
"    --scaffold_min_child_fraction F optional support fraction of parent block\n"
"    --scaffold_max_merged_sqrt_residual F parent merged-QEM residual gate, m\n"
"    --scaffold_max_rank_ratio F parent rank-1 flatness gate lambda_mid/lambda_big\n"
"    --scaffold_max_edge_factor F max scaffold triangle edge / voxel_size\n"
"    --scaffold_min_normal_consistency F parent/source consistency gate\n"
"    --scaffold_max_bel_free F EOGM free-belief veto for scaffold support\n"
"    --scaffold_max_conflict F EOGM conflict veto for scaffold support\n"
"    --scaffold_inherit_weight F L1 virtual QEM weight in crossed L0 child voxels\n"
"    --scaffold_inherit_level_decay F virtual QEM multiplier per coarser level\n"
"    --scaffold_inherit_decay_weight_ref F real-weight scale for gradual virtual-QEM decay\n"
"    --scaffold_skip_real_children do not inject virtual QEM into children with real QEM\n"
"    --scaffold_fill_require_plane_crossing mesh-only fill only creates child vertices crossed by the parent plane\n"
"    --scaffold_fill_all_children mimic old single scaffold: valid parent may fill all child slots\n"
"    --disable_scaffold_inherited_corner_sign keep inherited UNKNOWN cells out of corner_dc signs\n"
"    --enable_scaffold_inherited_corner_sign allow inherited cells to create weak corner signs\n"
"    --scaffold_inherited_sign_weight F weak sign weight multiplier for inherited cells (default 0.25)\n"
"    --scaffold_inherited_min_decay_scale F min surviving virtual-QEM scale for signs (default 0.02)\n"
"    --scaffold_inherited_sign_max_sqrt_residual F inherited sign residual gate, m (default 0.060)\n\n"
"  Component growth / PlanarMesh-like topology expansion:\n"
"    --enable_component_growth grow mesh components from boundary edges toward low-degree QEM/scaffold vertices\n"
"    --disable_component_growth disable component growth\n"
"    --component_growth_iters N growth iterations after corner_dc_plus/scaffold (default 1)\n"
"    --component_growth_min_component_faces N ignore tiny seed components below this face count\n"
"    --component_growth_target_min_incident_faces N vertices below this degree are growth targets\n"
"    --component_growth_max_candidates_per_edge N candidate cap per boundary edge\n"
"    --component_growth_boundary_radius_factor F base boundary radius / voxel_size\n"
"    --component_growth_flat_radius_factor F flat-component boundary radius / voxel_size\n"
"    --component_growth_max_radius_factor F search radius cap / voxel_size\n"
"    --component_growth_max_edge_factor F max growth triangle edge / voxel_size\n"
"    --component_growth_normal_dot F vertex-vertex normal compatibility\n"
"    --component_growth_component_normal_dot F vertex-component normal compatibility\n"
"    --component_growth_plane_dist_factor F candidate distance to component plane / voxel_size\n"
"    --component_growth_max_merged_sqrt_residual F merged triangle QEM residual gate, m\n"
"    --component_growth_max_point_plane_dist_factor F merged-QEM point-plane distance / voxel_size\n"
"    --component_growth_max_bel_free F EOGM free-belief veto\n"
"    --component_growth_max_conflict F EOGM conflict veto\n"
"    --component_growth_min_plaus_surface F EOGM surface plausibility gate\n"
"    --component_growth_allow_weak_only_triangles allow triangles without any confirmed vertex\n"
"    --component_growth_no_free_reject disable EOGM/free-space samples for growth triangles\n"
"    --enable_component_persistent_state cache stable component IDs/owned faces/boundary radii after mesh builds\n"
"    --disable_component_persistent_state disable persistent component cache\n"
"    --component_growth_dirty_only grow only components/targets touched by dirty voxel updates\n"
"    --component_growth_all_components grow all components each mesh build\n"
"    --component_growth_dirty_radius_voxels N dirty-neighborhood radius in voxels for incremental component growth\n"
"    --component_growth_rrs_boundary_search require target to lie inside a boundary vertex radius\n"
"    --component_growth_edge_search use old boundary-edge distance search instead of RRS-like boundary radii\n"
"    --component_growth_qem_rank_rrs use QEM rank-aware RRS: flat=disk, edge=line, corner=local\n"
"    --component_growth_sphere_rrs use old spherical boundary radius test\n"
"    --component_growth_edge_line_dist_factor F edge/rank-2 line distance tolerance / voxel_size\n"
"    --component_growth_corner_radius_factor F corner/rank-3 radius multiplier\n"
"    --enable_component_fis_delete enable FIS-like EOGM/free-space face deletion\n"
"    --disable_component_fis_delete disable FIS-like deletion\n"
"    --enable_component_radius_shrink enable rank-radius edge shrink/delete\n"
"    --disable_component_radius_shrink disable radius shrink/delete\n"
"    --enable_component_adaptive_simplification enable conservative flat-component face thinning\n"
"    --disable_component_adaptive_simplification disable component simplification\n\n"
"  EOGM / generative completion:\n"
"    --disable_eogm          disable Dempster-Shafer/EOGM evidence accumulation/gates\n"
"    --seed_evidence_mode MODE  old | eogm (default old). Promotion uses either old\n"
"                              hit/miss occupancy gates OR EOGM plausibility/free/conflict gates,\n"
"                              never both stacked. Geometry/QEM gates remain mandatory.\n"
"    --eogm_mesh_gate       also use EOGM as a face veto in ordinary corner_dc_plus meshing\n"
"                              (default off; generative fill still uses EOGM vetoes).\n"
"    --eogm_seed_min_plaus_surface F min Pl(surface)=mS+mU for seed promotion (default 0.45)\n"
"    --eogm_seed_max_bel_free F max Bel(free)=mF for seed promotion (default 0.70)\n"
"    --eogm_seed_max_conflict F max retained conflict for seed promotion (default 0.65)\n"
"    --enable_eogm_generative_fill enable conservative mesh-only virtual fill patches\n"
"    --gen_max_edge_factor F max real-support triangle edge / voxel_size (default 8.0)\n"
"    --gen_max_merged_sqrt_residual F max merged QEM residual for generated patch (default 0.060)\n"
"    --gen_max_bel_free F EOGM free-belief veto for generated samples (default 0.35)\n"
"    --gen_max_conflict F EOGM conflict veto for generated samples (default 0.45)\n\n"
"  Free-space carving:\n"
"    --no_carve               disable ray carving (faster, no free/unknown)\n"
"    --max_ray_voxels N       per-ray traversal cap (default 2000)\n\n"
"  Labelling:\n"
"    --thresh_surface F       surface threshold on occupancy score (def 0.10)\n"
"    --thresh_free F          free threshold (default -0.50)\n"
"    --min_evidence N         hits+misses needed to label (default 2)\n"
"    --min_hit_count_vertex N hits needed to extract a vertex (default 2)\n\n"
"  Vertex solve:\n"
"    --lambda_p F             anchor regularisation (default 0.01)\n"
"    --no_clamp               disable per-voxel position clamping\n\n"
"  Run control:\n"
"    --num_scans N            limit number of scans processed (default: all)\n"
"    --snapshot_interval N    save partial output every N scans\n\n"
"  Visualization (file dumps for browser viewer):\n"
"    --dump_dir PATH          enable per-scan PLY dumps in PATH\n"
"    --dump_roi X Y Z R       OPTIONAL ROI sphere (world frame). When\n"
"                             omitted, every surface vertex is dumped\n"
"                             (subject to --dump_max_verts).\n"
"    --dump_every N           dump every Nth scan (default 1)\n"
"    --dump_max_verts N       cap vertices per dump (default 500000)\n\n"
"  Phase 3 meshing:\n"
"    --mesh_mode MODE         smooth | dual | corner_dc | corner_dc_plus | surface_net | hybrid (default smooth); add --enable_component_growth for boundary-grow pass\n"
"    --disable_persistent_mesh   rebuild the whole mesh from scratch on every export (old behavior)\n"
"    --enable_persistent_mesh    PlanarMesh-style: keep one mesh, remesh only dirty voxel regions (default on)\n"
"    --persistent_mesh_halo_voxels N   halo (voxels) re-meshed around the dirty region; 0=auto (default)\n"
"    --disable_seam_closing      do not run the seam-aware hole-closing pass after each incremental splice\n"
"    --enable_seam_closing       close holes across the persistent<->local seam (default on)\n"
"    --seam_close_iters N        cascade passes for seam hole closing (default 2)\n"
"    --disable_persistent_retire_free_faces   keep persistent faces even if their interior crosses observed free space\n"
"    --enable_persistent_retire_free_faces    retire persistent/seam faces whose centroid/edges cross FREE space (default on)\n"
"    --disable_sparse_scaffold   do not inherit virtual QEM/evidence into sparse voxels (manage --scaffold_* manually)\n"
"    --enable_sparse_scaffold    scaffold sparse regions with inherited coarse-level QEM planes (default on)\n"
"                              smooth         = local fan triangulation on smooth patches\n"
"                              dual / dc      = grid-edge DC, cell-labelled\n"
"                              corner_dc      = corner-sign DC; higher recall than dual\n"
"                              corner_dc_plus = corner_dc + RRS-like boundary QEM grow/fill\n"
"                              surface_net    = older face-iteration surface nets (axis-redundant)\n"
"                              hybrid         = corner_dc_plus first, smooth fallback if 0 faces\n"
"    --mesh_output PATH       final mesh PLY with faces; empty disables mesh export\n"
"    --mesh_dump_dir PATH     optional mesh snapshot/submap directory\n"
"    --mesh_every N           export mesh every N scans; 0 disables snapshots\n"
"    --mesh_submap_scans N    for snapshots, mesh only cells last hit in recent N scans\n"
"    --mesh_min_hit_count N   min hit_count per mesh vertex (default 3)\n"
"    --mesh_max_sqrt_residual F  residual sqrt cutoff in metres; <=0 disables\n"
"    --mesh_min_consistency F min normal consistency (default 0)\n"
"    --mesh_normal_dot F      smooth-neighbor normal abs-dot threshold (default 0.90)\n"
"    --mesh_neighbor_radius_factor F  adjacency radius / voxel_size (default 1.85)\n"
"    --mesh_max_edge_factor F max triangle edge / voxel_size (default 2.50)\n"
"    --mesh_triangle_normal_dot F min abs(face normal dot avg normal) (default 0.35)\n"
"    --mesh_max_boundary_ratio F reject mesh verts above this; 1.01 disables\n"
"    --mesh_no_free_reject    do not reject triangles whose centroid lies in FREE cells\n"
"\n"
"  Optional QEM-projected bilateral vertex smoothing (mesh output only):\n"
"    --enable_vertex_smoothing enable conservative post-mesh vertex correction\n"
"    --disable_vertex_smoothing disable vertex correction (default)\n"
"    --vertex_smooth_iters N   smoothing iterations (default 1)\n"
"    --vertex_smooth_apply_to_confirmed allow confirmed vertices to be considered\n"
"    --vertex_smooth_allow_confirmed_flat allow confirmed rank-1/flat vertices to move\n"
"    --vertex_smooth_no_preserve_edges allow edge/corner vertices to be considered\n"
"    --vertex_smooth_full_vector allow tangential movement; default is normal-only\n"
"    --vertex_smooth_no_qem_project disable QEM projection/acceptance test\n"
"    --vertex_smooth_normal_dot F edge-stop normal abs-dot threshold (default 0.85)\n"
"    --vertex_smooth_radius_factor F neighbor radius / voxel_size (default 2.0)\n"
"    --vertex_smooth_max_move_factor F max movement / voxel_size per iter (default 0.25)\n"
"    --vertex_smooth_min_sqrt_residual F confirmed-vertex residual gate, m (default 0.03)\n"
"    --vertex_smooth_max_bel_free F EOGM free-belief veto (default 0.60)\n"
"    --vertex_smooth_max_conflict F EOGM conflict veto (default 0.60)\n"
"    --dc_require_free        dual mode only: require observed SURFACE/FREE support (default)\n"
"    --dc_no_require_free     dual mode only: allow surface-only quads, useful with --no_carve\n"
"    --dc_normal_dot F        dual neighbor normal abs-dot threshold (default 0.50)\n"
"    --dc_max_edge_factor F   dual max triangle edge / voxel_size (default 2.50)\n"
"    --dc_triangle_normal_dot F dual min abs(face normal dot expected normal) (default 0.15)\n"
"    --cdp_iters N           corner_dc_plus grow iterations (default 1)\n"
"    --cdp_min_incident_faces N target verts with fewer faces than this (default 2)\n"
"    --cdp_flat_radius_factor F RRS radius for flat boundaries / voxel_size (default 6.0)\n"
"    --cdp_curve_radius_factor F radius for curved boundaries if enabled (default 3.0)\n"
"    --cdp_max_radius_factor F max RRS radius / voxel_size (default 8.0)\n"
"    --cdp_max_edge_factor F max adaptive fill edge / voxel_size (default 6.0)\n"
"    --cdp_max_candidates_per_edge N cap candidates per boundary edge (default 24)\n"
"    --cdp_planar_normal_dot F planar normal compatibility (default 0.90)\n"
"    --cdp_curve_normal_dot F curved normal compatibility (default 0.75)\n"
"    --cdp_min_consistency F min normal consistency for adaptive grow (default 0.88)\n"
"    --cdp_max_boundary_ratio F max scan-boundary ratio for grow vertices (default 0.85)\n"
"    --cdp_max_sqrt_residual F per-cell residual gate in metres (default 0.05)\n"
"    --cdp_max_merged_sqrt_residual F merged-QEM residual gate in metres (default 0.05)\n"
"    --cdp_max_point_plane_dist_factor F merged plane distance / voxel_size (default 1.0)\n"
"    --cdp_allow_curve       allow non-flat adaptive grow with looser normal gate\n"
"    --cdp_no_free_reject    do not reject adaptive fill samples in FREE cells\n\n"
"  After enabling --dump_dir, launch the viewer in a separate terminal:\n"
"    python3 viz_server.py PATH 8080\n"
"  then open http://<host>:8080/ in a browser (or SSH-tunnel localhost:8080).\n";
}

// ========================================================================= //
// 13. main                                                                   //
// ========================================================================= //

int main(int argc, char** argv) {
    std::string pcd_folder, pose_file, output = "map.ply";
    Settings settings;
    bool saw_disable_scanline = false;
    bool saw_pandar_qt64 = false;
    bool saw_pandar_uniform = false;
    bool saw_pandar_infer = false;

    argv_util::init(argc, argv);
    for (; argv_util::g_i < argc; ++argv_util::g_i) {
        std::string arg = argv[argv_util::g_i];
        if      (arg == "--pcd_folder")           pcd_folder = argv_util::next_raw(arg);
        else if (arg == "--pose_file")            pose_file  = argv_util::next_raw(arg);
        else if (arg == "--output")               output     = argv_util::next_raw(arg);
        else if (arg == "--voxel_size")           settings.voxel_size           = argv_util::next_double(arg);
        else if (arg == "--range_precision")      settings.range_precision      = argv_util::next_double(arg);
        else if (arg == "--disable_prob_planes") settings.enable_probabilistic_planes = false;
        else if (arg == "--bearing_sigma_rad")   settings.bearing_sigma_rad   = argv_util::next_double(arg);
        else if (arg == "--pose_trans_sigma")    settings.pose_trans_sigma    = argv_util::next_double(arg);
        else if (arg == "--pose_rot_sigma_rad")  settings.pose_rot_sigma_rad  = argv_util::next_double(arg);
        else if (arg == "--prob_min_sigma")      settings.prob_min_sigma      = argv_util::next_double(arg);
        else if (arg == "--prob_max_sigma")      settings.prob_max_sigma      = argv_util::next_double(arg);
        else if (arg == "--prob_qem_ref_sigma")  settings.prob_qem_ref_sigma  = argv_util::next_double(arg);
        else if (arg == "--prob_qem_min_scale")  settings.prob_qem_min_scale  = argv_util::next_double(arg);
        else if (arg == "--prob_gate_sigma")     settings.prob_gate_sigma     = argv_util::next_double(arg);
        else if (arg == "--prob_gate_min_factor") settings.prob_gate_min_factor = argv_util::next_double(arg);
        else if (arg == "--prob_gate_max_factor") settings.prob_gate_max_factor = argv_util::next_double(arg);
        else if (arg == "--prob_planar_eigen_thresh") settings.prob_planar_eigen_thresh = argv_util::next_double(arg);
        else if (arg == "--min_incident_cos")     settings.min_incident_cos     = argv_util::next_double(arg);
        else if (arg == "--process_every_n")      settings.process_every_n      = argv_util::next_int(arg);
        else if (arg == "--estimate_normals_k")   settings.estimate_normals_k   = argv_util::next_int(arg);
        else if (arg == "--disable_nvt")          settings.enable_nvt           = false;
        else if (arg == "--nvt_all_normals")      settings.nvt_only_for_pca     = false;
        else if (arg == "--nvt_k")                settings.nvt_k                = argv_util::next_int(arg);
        else if (arg == "--nvt_iters")            settings.nvt_iters            = argv_util::next_int(arg);
        else if (arg == "--nvt_rho_cos")          settings.nvt_rho_cos          = argv_util::next_double(arg);
        else if (arg == "--nvt_tau")              settings.nvt_tau              = argv_util::next_double(arg);
        else if (arg == "--nvt_damping")          settings.nvt_damping          = argv_util::next_double(arg);
        else if (arg == "--nvt_min_conf")         settings.nvt_min_conf         = argv_util::next_double(arg);
        else if (arg == "--nvt_conf_soft_floor")  settings.nvt_conf_soft_floor  = argv_util::next_double(arg);
        else if (arg == "--nvt_max_radius")       settings.nvt_max_radius       = argv_util::next_double(arg);
        else if (arg == "--disable_scanline")     { settings.enable_scanline = false; saw_disable_scanline = true; }
        else if (arg == "--disable_scanline_normals") settings.scanline_normals  = false;
        else if (arg == "--scanline_rows")        settings.scanline_rows        = argv_util::next_int(arg);
        else if (arg == "--scanline_cols")        settings.scanline_cols        = argv_util::next_int(arg);
        else if (arg == "--scanline_gate_rows")   settings.scanline_gate_rows   = argv_util::next_int(arg);
        else if (arg == "--scanline_gate_cols")   settings.scanline_gate_cols   = argv_util::next_int(arg);
        else if (arg == "--scanline_jump_abs")    settings.scanline_jump_abs    = argv_util::next_double(arg);
        else if (arg == "--scanline_jump_rel")    settings.scanline_jump_rel    = argv_util::next_double(arg);
        else if (arg == "--scanline_boundary_conf") settings.scanline_boundary_conf = argv_util::next_double(arg);
        else if (arg == "--points_registered" || arg == "--pcd_registered" || arg == "--use_pcd_viewpoint") {
            std::fprintf(stderr, "[WARN] %s ignored: this build always uses the external SLAM pose and treats PCD points as sensor-frame measurements.\n", arg.c_str());
        }
        else if (arg == "--sensor") {
            std::string sensor = argv_util::next_raw(arg);
            std::string lower = sensor;
            for (char& c : lower) c = (char)std::tolower((unsigned char)c);
            if (lower == "pandar_qt64" || lower == "pandarqt64" || lower == "qt64") {
                settings.pandar_qt64_mode = true; saw_pandar_qt64 = true;
                settings.enable_scanline = true;
                if (settings.scanline_cols <= 0) settings.scanline_cols = 600; // 10 Hz, 0.6 deg nominal
            } else {
                std::cerr << "Unknown --sensor value: " << sensor << "\n";
                return 1;
            }
        }
        else if (arg == "--pandar_qt64")          { settings.pandar_qt64_mode = true; saw_pandar_qt64 = true; settings.enable_scanline = true; if (settings.scanline_cols <= 0) settings.scanline_cols = 600; }
        else if (arg == "--pandar_qt64_calib")    settings.pandar_qt64_calib = argv_util::next_raw(arg);
        else if (arg == "--pandar_qt64_uniform")  { settings.pandar_qt64_uniform = true; settings.pandar_qt64_infer_rings = false; saw_pandar_uniform = true; }
        else if (arg == "--pandar_qt64_infer")    { settings.pandar_qt64_infer_rings = true; settings.pandar_qt64_uniform = false; saw_pandar_infer = true; }
        else if (arg == "--no_carve")             settings.carve_rays           = false;
        else if (arg == "--max_ray_voxels")       settings.max_ray_voxels       = argv_util::next_int(arg);
        else if (arg == "--thresh_surface")       settings.thresh_surface       = argv_util::next_double(arg);
        else if (arg == "--thresh_free")          settings.thresh_free          = argv_util::next_double(arg);
        else if (arg == "--min_evidence")         settings.min_evidence         = argv_util::next_int(arg);
        else if (arg == "--min_hit_count_vertex") settings.min_hit_count_vertex = argv_util::next_int(arg);
        else if (arg == "--disable_seed_voxels")  settings.enable_seed_voxels   = false;
        else if (arg == "--enable_seed_voxels")   settings.enable_seed_voxels   = true;
        else if (arg == "--seed_min_incident_cos") settings.seed_min_incident_cos = argv_util::next_double(arg);
        else if (arg == "--seed_hit_weight_scale") settings.seed_hit_weight_scale = argv_util::next_double(arg);
        else if (arg == "--seed_min_hit_weight")  settings.seed_min_hit_weight  = argv_util::next_double(arg);
        else if (arg == "--seed_carve_rays")      settings.seed_carve_rays      = true;
        else if (arg == "--seed_no_carve_rays")   settings.seed_carve_rays      = false;
        else if (arg == "--seed_miss_weight_scale") settings.seed_miss_weight_scale = argv_util::next_double(arg);
        else if (arg == "--seed_ray_normal_fallback") settings.seed_ray_normal_fallback = true;
        else if (arg == "--no_seed_ray_normal_fallback") settings.seed_ray_normal_fallback = false;
        else if (arg == "--seed_export_min_hits") settings.seed_export_min_hits = argv_util::next_int(arg);
        else if (arg == "--seed_promote_min_hits") settings.seed_promote_min_hits = argv_util::next_int(arg);
        else if (arg == "--seed_promote_neighbor_min_hits") settings.seed_promote_neighbor_min_hits = argv_util::next_int(arg);
        else if (arg == "--seed_promote_min_neighbors") settings.seed_promote_min_neighbors = argv_util::next_int(arg);
        else if (arg == "--seed_promote_min_occupancy") settings.seed_promote_min_occupancy = argv_util::next_double(arg);
        else if (arg == "--seed_promote_min_consistency") settings.seed_promote_min_consistency = argv_util::next_double(arg);
        else if (arg == "--seed_promote_max_sqrt_residual") settings.seed_promote_max_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--seed_promote_neighbor_normal_dot") settings.seed_promote_neighbor_normal_dot = argv_util::next_double(arg);
        else if (arg == "--seed_promote_neighbor_plane_dist_factor") settings.seed_promote_neighbor_plane_dist_factor = argv_util::next_double(arg);
        else if (arg == "--seed_max_ray_normal_fraction") settings.seed_max_ray_normal_fraction = argv_util::next_double(arg);
        else if (arg == "--seed_promote_cluster_min_size") settings.seed_promote_cluster_min_size = argv_util::next_int(arg);
        else if (arg == "--seed_promote_cluster_radius_voxels") settings.seed_promote_cluster_radius_voxels = argv_util::next_int(arg);
        else if (arg == "--seed_promote_cluster_normal_dot") settings.seed_promote_cluster_normal_dot = argv_util::next_double(arg);
        else if (arg == "--seed_promote_cluster_plane_dist_factor") settings.seed_promote_cluster_plane_dist_factor = argv_util::next_double(arg);
        else if (arg == "--seed_promote_cluster_merged_max_sqrt_residual") settings.seed_promote_cluster_merged_max_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--no_export_seed_vertices") settings.export_seed_vertices = false;
        else if (arg == "--disable_seed_l1_parents") settings.enable_seed_l1_parents = false;
        else if (arg == "--enable_seed_l1_parents") settings.enable_seed_l1_parents = true;
        else if (arg == "--seed_l1_min_unique_scans") settings.seed_l1_min_unique_scans = argv_util::next_int(arg);
        else if (arg == "--seed_l1_min_children") settings.seed_l1_min_children = argv_util::next_int(arg);
        else if (arg == "--seed_l1_min_consistency") settings.seed_l1_min_consistency = argv_util::next_double(arg);
        else if (arg == "--seed_l1_max_sqrt_residual") settings.seed_l1_max_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--seed_l1_max_ray_normal_fraction") settings.seed_l1_max_ray_normal_fraction = argv_util::next_double(arg);
        else if (arg == "--seed_l1_child_plane_dist") settings.seed_l1_child_plane_dist = argv_util::next_double(arg);
        else if (arg == "--seed_l1_child_normal_dot") settings.seed_l1_child_normal_dot = argv_util::next_double(arg);
        else if (arg == "--seed_l1_child_min_occupancy") settings.seed_l1_child_min_occupancy = argv_util::next_double(arg);
        else if (arg == "--no_export_weak_vertices") settings.export_weak_vertices = false;
        else if (arg == "--export_weak_vertices") settings.export_weak_vertices = true;
        else if (arg == "--lambda_p")             settings.lambda_p             = argv_util::next_double(arg);
        else if (arg == "--no_clamp")             settings.clamp_to_voxel       = false;
        else if (arg == "--num_scans")            settings.num_scans            = argv_util::next_int(arg);
        else if (arg == "--snapshot_interval")    settings.snapshot_interval    = argv_util::next_int(arg);
        // Visualization dump flags. Mirrors the --debug_dump_dir / --debug_roi
        // pattern used by qem_mesh.cpp so muscle memory carries over.
        else if (arg == "--dump_dir")             settings.dump_dir             = argv_util::next_raw(arg);
        else if (arg == "--dump_roi") {
            settings.dump_roi_center[0] = argv_util::next_double(arg);
            settings.dump_roi_center[1] = argv_util::next_double(arg);
            settings.dump_roi_center[2] = argv_util::next_double(arg);
            settings.dump_roi_radius    = argv_util::next_double(arg);
        }
        else if (arg == "--dump_every")           settings.dump_every           = argv_util::next_int(arg);
        else if (arg == "--dump_max_verts")       settings.dump_max_verts       = argv_util::next_int(arg);
        else if (arg == "--mesh_mode")            settings.mesh_mode            = argv_util::next_raw(arg);
        else if (arg == "--enable_persistent_mesh")  settings.persistent_incremental_mesh = true;
        else if (arg == "--disable_persistent_mesh") settings.persistent_incremental_mesh = false;
        else if (arg == "--persistent_mesh_halo_voxels") settings.persistent_mesh_halo_voxels = argv_util::next_int(arg);
        else if (arg == "--enable_seam_closing")  settings.enable_seam_closing = true;
        else if (arg == "--disable_seam_closing") settings.enable_seam_closing = false;
        else if (arg == "--seam_close_iters")     settings.seam_close_iters     = argv_util::next_int(arg);
        else if (arg == "--enable_persistent_retire_free_faces")  settings.persistent_retire_free_faces = true;
        else if (arg == "--disable_persistent_retire_free_faces") settings.persistent_retire_free_faces = false;
        else if (arg == "--enable_sparse_scaffold")  settings.sparse_region_scaffold = true;
        else if (arg == "--disable_sparse_scaffold") settings.sparse_region_scaffold = false;
        else if (arg == "--dc_require_free")      settings.dc_require_free      = true;
        else if (arg == "--dc_no_require_free")   settings.dc_require_free      = false;
        else if (arg == "--dc_normal_dot")        settings.dc_normal_dot        = argv_util::next_double(arg);
        else if (arg == "--dc_max_edge_factor")   settings.dc_max_edge_factor   = argv_util::next_double(arg);
        else if (arg == "--dc_min_area_factor")   settings.dc_min_area_factor   = argv_util::next_double(arg);
        else if (arg == "--dc_triangle_normal_dot") settings.dc_triangle_normal_dot = argv_util::next_double(arg);
        else if (arg == "--disable_eogm") settings.enable_eogm = false;
        else if (arg == "--enable_eogm")  settings.enable_eogm = true;
        else if (arg == "--seed_evidence_mode") {
            settings.seed_evidence_mode = argv_util::next_raw(arg);
            for (char& c : settings.seed_evidence_mode) c = (char)std::tolower((unsigned char)c);
            if (settings.seed_evidence_mode != "old" && settings.seed_evidence_mode != "eogm") {
                std::fprintf(stderr, "Error: --seed_evidence_mode must be 'old' or 'eogm'\n");
                return 2;
            }
        }
        else if (arg == "--eogm_mesh_gate") settings.eogm_mesh_gate = true;
        else if (arg == "--no_eogm_mesh_gate") settings.eogm_mesh_gate = false;
        else if (arg == "--eogm_hit_scale") settings.eogm_hit_scale = argv_util::next_double(arg);
        else if (arg == "--eogm_seed_hit_scale") settings.eogm_seed_hit_scale = argv_util::next_double(arg);
        else if (arg == "--eogm_miss_scale") settings.eogm_miss_scale = argv_util::next_double(arg);
        else if (arg == "--eogm_seed_miss_scale") settings.eogm_seed_miss_scale = argv_util::next_double(arg);
        else if (arg == "--eogm_seed_min_plaus_surface") settings.eogm_seed_min_plaus_surface = argv_util::next_double(arg);
        else if (arg == "--eogm_seed_max_bel_free") settings.eogm_seed_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--eogm_seed_max_conflict") settings.eogm_seed_max_conflict = argv_util::next_double(arg);
        else if (arg == "--eogm_cluster_min_plaus_surface") settings.eogm_cluster_min_plaus_surface = argv_util::next_double(arg);
        else if (arg == "--eogm_cluster_max_bel_free") settings.eogm_cluster_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--eogm_cluster_max_conflict") settings.eogm_cluster_max_conflict = argv_util::next_double(arg);
        else if (arg == "--eogm_mesh_max_bel_free") settings.eogm_mesh_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--eogm_mesh_max_conflict") settings.eogm_mesh_max_conflict = argv_util::next_double(arg);
        else if (arg == "--enable_eogm_generative_fill") settings.enable_eogm_generative_fill = true;
        else if (arg == "--disable_eogm_generative_fill") settings.enable_eogm_generative_fill = false;
        else if (arg == "--gen_max_faces") settings.gen_max_faces = argv_util::next_int(arg);
        else if (arg == "--gen_max_edge_factor") settings.gen_max_edge_factor = argv_util::next_double(arg);
        else if (arg == "--gen_max_merged_sqrt_residual") settings.gen_max_merged_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--gen_max_bel_free") settings.gen_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--gen_max_conflict") settings.gen_max_conflict = argv_util::next_double(arg);
        else if (arg == "--gen_min_support_plaus_surface") settings.gen_min_support_plaus_surface = argv_util::next_double(arg);
        else if (arg == "--cdp_iters")            settings.cdp_iters            = argv_util::next_int(arg);
        else if (arg == "--cdp_min_incident_faces") settings.cdp_min_incident_faces = argv_util::next_int(arg);
        else if (arg == "--cdp_flat_radius_factor") settings.cdp_flat_radius_factor = argv_util::next_double(arg);
        else if (arg == "--cdp_curve_radius_factor") settings.cdp_curve_radius_factor = argv_util::next_double(arg);
        else if (arg == "--cdp_max_radius_factor") settings.cdp_max_radius_factor = argv_util::next_double(arg);
        else if (arg == "--cdp_max_edge_factor") settings.cdp_max_edge_factor = argv_util::next_double(arg);
        else if (arg == "--cdp_max_candidates_per_edge") settings.cdp_max_candidates_per_edge = argv_util::next_int(arg);
        else if (arg == "--cdp_planar_normal_dot") settings.cdp_planar_normal_dot = argv_util::next_double(arg);
        else if (arg == "--cdp_curve_normal_dot") settings.cdp_curve_normal_dot = argv_util::next_double(arg);
        else if (arg == "--cdp_min_consistency") settings.cdp_min_consistency = argv_util::next_double(arg);
        else if (arg == "--cdp_max_boundary_ratio") settings.cdp_max_boundary_ratio = argv_util::next_double(arg);
        else if (arg == "--cdp_max_sqrt_residual") settings.cdp_max_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--cdp_max_merged_sqrt_residual") settings.cdp_max_merged_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--cdp_max_point_plane_dist_factor") settings.cdp_max_point_plane_dist_factor = argv_util::next_double(arg);
        else if (arg == "--cdp_allow_curve")      settings.cdp_allow_curve      = true;
        else if (arg == "--cdp_no_free_reject")   settings.cdp_reject_free_samples = false;
        else if (arg == "--cdp_no_weak_vertices") settings.cdp_use_weak_vertices = false;
        else if (arg == "--cdp_use_weak_vertices") settings.cdp_use_weak_vertices = true;
        else if (arg == "--cdp_allow_weak_only_triangles") settings.cdp_weak_triangles_require_confirmed = false;
        else if (arg == "--cdp_require_confirmed_in_weak_triangles") settings.cdp_weak_triangles_require_confirmed = true;
        else if (arg == "--enable_hierarchical_scaffold") settings.enable_hierarchical_scaffold = true;
        else if (arg == "--disable_hierarchical_scaffold") settings.enable_hierarchical_scaffold = false;
        else if (arg == "--enable_hierarchical_scaffold_fill") settings.enable_hierarchical_scaffold_fill = true;
        else if (arg == "--disable_hierarchical_scaffold_fill") settings.enable_hierarchical_scaffold_fill = false;
        else if (arg == "--enable_planar_scaffold") { settings.enable_planar_scaffold = true; settings.enable_hierarchical_scaffold_fill = true; if (settings.scaffold_max_level < 1) settings.scaffold_max_level = 1; }
        else if (arg == "--disable_planar_scaffold") settings.enable_planar_scaffold = false;
        else if (arg == "--enable_planar_scaffold_inherit") { settings.enable_planar_scaffold_inherit = true; settings.enable_hierarchical_scaffold = true; }
        else if (arg == "--disable_planar_scaffold_inherit") settings.enable_planar_scaffold_inherit = false;
        else if (arg == "--scaffold_l1_factor") { settings.scaffold_base_factor = argv_util::next_int(arg); if (settings.scaffold_max_level < 1) settings.scaffold_max_level = 1; }
        else if (arg == "--scaffold_base_factor") settings.scaffold_base_factor = argv_util::next_int(arg);
        else if (arg == "--scaffold_max_level") settings.scaffold_max_level = argv_util::next_int(arg);
        else if (arg == "--scaffold_min_parent_children") settings.scaffold_min_parent_children = argv_util::next_int(arg);
        else if (arg == "--scaffold_min_parent_children_per_level") settings.scaffold_min_parent_children_per_level = argv_util::next_int(arg);
        else if (arg == "--scaffold_min_child_fraction") settings.scaffold_min_child_fraction = argv_util::next_double(arg);
        else if (arg == "--scaffold_max_merged_sqrt_residual") settings.scaffold_max_merged_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--scaffold_residual_level_decay") settings.scaffold_residual_level_decay = argv_util::next_double(arg);
        else if (arg == "--scaffold_max_rank_ratio") settings.scaffold_max_rank_ratio = argv_util::next_double(arg);
        else if (arg == "--scaffold_rank_level_decay") settings.scaffold_rank_level_decay = argv_util::next_double(arg);
        else if (arg == "--scaffold_max_edge_factor") settings.scaffold_max_edge_factor = argv_util::next_double(arg);
        else if (arg == "--scaffold_min_normal_consistency") settings.scaffold_min_normal_consistency = argv_util::next_double(arg);
        else if (arg == "--scaffold_max_bel_free") settings.scaffold_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--scaffold_max_conflict") settings.scaffold_max_conflict = argv_util::next_double(arg);
        else if (arg == "--scaffold_inherit_weight") settings.scaffold_inherit_weight = argv_util::next_double(arg);
        else if (arg == "--scaffold_inherit_level_decay") settings.scaffold_inherit_level_decay = argv_util::next_double(arg);
        else if (arg == "--scaffold_inherit_decay_weight_ref") settings.scaffold_inherit_decay_weight_ref = argv_util::next_double(arg);
        else if (arg == "--scaffold_skip_real_children") settings.scaffold_inherit_into_real_cells = false;
        else if (arg == "--scaffold_inherit_into_real_cells") settings.scaffold_inherit_into_real_cells = true;
        else if (arg == "--scaffold_fill_require_plane_crossing") settings.scaffold_fill_require_plane_crossing = true;
        else if (arg == "--scaffold_fill_all_children") settings.scaffold_fill_require_plane_crossing = false;
        else if (arg == "--enable_scaffold_inherited_corner_sign") settings.scaffold_inherited_corner_sign = true;
        else if (arg == "--disable_scaffold_inherited_corner_sign") settings.scaffold_inherited_corner_sign = false;
        else if (arg == "--scaffold_inherited_sign_weight") settings.scaffold_inherited_sign_weight = argv_util::next_double(arg);
        else if (arg == "--scaffold_inherited_min_decay_scale") settings.scaffold_inherited_min_decay_scale = argv_util::next_double(arg);
        else if (arg == "--scaffold_inherited_sign_max_sqrt_residual") settings.scaffold_inherited_sign_max_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--enable_component_growth") settings.enable_component_growth = true;
        else if (arg == "--disable_component_growth") settings.enable_component_growth = false;
        else if (arg == "--component_growth_iters") settings.component_growth_iters = argv_util::next_int(arg);
        else if (arg == "--component_growth_min_component_faces") settings.component_growth_min_component_faces = argv_util::next_int(arg);
        else if (arg == "--component_growth_target_min_incident_faces") settings.component_growth_target_min_incident_faces = argv_util::next_int(arg);
        else if (arg == "--component_growth_max_candidates_per_edge") settings.component_growth_max_candidates_per_edge = argv_util::next_int(arg);
        else if (arg == "--component_growth_boundary_radius_factor") settings.component_growth_boundary_radius_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_flat_radius_factor") settings.component_growth_flat_radius_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_max_radius_factor") settings.component_growth_max_radius_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_max_edge_factor") settings.component_growth_max_edge_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_normal_dot") settings.component_growth_normal_dot = argv_util::next_double(arg);
        else if (arg == "--component_growth_component_normal_dot") settings.component_growth_component_normal_dot = argv_util::next_double(arg);
        else if (arg == "--component_growth_plane_dist_factor") settings.component_growth_plane_dist_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_max_merged_sqrt_residual") settings.component_growth_max_merged_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--component_growth_max_point_plane_dist_factor") settings.component_growth_max_point_plane_dist_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_max_bel_free") settings.component_growth_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--component_growth_max_conflict") settings.component_growth_max_conflict = argv_util::next_double(arg);
        else if (arg == "--component_growth_min_plaus_surface") settings.component_growth_min_plaus_surface = argv_util::next_double(arg);
        else if (arg == "--component_growth_no_free_reject") settings.component_growth_reject_free_samples = false;
        else if (arg == "--component_growth_free_reject") settings.component_growth_reject_free_samples = true;
        else if (arg == "--component_growth_allow_weak_only_triangles") settings.component_growth_require_confirmed_anchor = false;
        else if (arg == "--component_growth_require_confirmed_anchor") settings.component_growth_require_confirmed_anchor = true;
        else if (arg == "--component_growth_no_merged_qem_gate") settings.component_growth_require_merged_qem_when_available = false;
        else if (arg == "--component_growth_merged_qem_gate") settings.component_growth_require_merged_qem_when_available = true;
        else if (arg == "--enable_component_persistent_state") settings.component_growth_persistent_state = true;
        else if (arg == "--disable_component_persistent_state") settings.component_growth_persistent_state = false;
        else if (arg == "--enable_component_persistent_reuse_faces") settings.component_persistent_reuse_faces = true;
        else if (arg == "--disable_component_persistent_reuse_faces") settings.component_persistent_reuse_faces = false;
        else if (arg == "--component_persistent_reuse_dirty_faces") settings.component_persistent_reuse_dirty_faces = true;
        else if (arg == "--component_persistent_skip_dirty_faces") settings.component_persistent_reuse_dirty_faces = false;
        else if (arg == "--component_persistent_reuse_max_bel_free") settings.component_persistent_reuse_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--component_persistent_reuse_max_conflict") settings.component_persistent_reuse_max_conflict = argv_util::next_double(arg);
        else if (arg == "--component_persistent_reuse_min_plaus_surface") settings.component_persistent_reuse_min_plaus_surface = argv_util::next_double(arg);
        else if (arg == "--component_growth_dirty_only") settings.component_growth_dirty_only = true;
        else if (arg == "--component_growth_all_components") settings.component_growth_dirty_only = false;
        else if (arg == "--component_growth_dirty_radius_voxels") settings.component_growth_dirty_radius_voxels = argv_util::next_int(arg);
        else if (arg == "--component_growth_rrs_boundary_search") settings.component_growth_use_rrs_boundary_search = true;
        else if (arg == "--component_growth_edge_search") settings.component_growth_use_rrs_boundary_search = false;
        else if (arg == "--component_growth_qem_rank_rrs") settings.component_growth_qem_rank_rrs = true;
        else if (arg == "--component_growth_sphere_rrs") settings.component_growth_qem_rank_rrs = false;
        else if (arg == "--component_growth_edge_line_dist_factor") settings.component_growth_edge_line_dist_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_corner_radius_factor") settings.component_growth_corner_radius_factor = argv_util::next_double(arg);
        else if (arg == "--component_growth_rank_transition_normal_dot") settings.component_growth_rank_transition_normal_dot = argv_util::next_double(arg);
        else if (arg == "--enable_component_fis_delete") settings.enable_component_fis_delete = true;
        else if (arg == "--disable_component_fis_delete") settings.enable_component_fis_delete = false;
        else if (arg == "--component_fis_delete_dirty_only") settings.component_fis_delete_dirty_only = true;
        else if (arg == "--component_fis_delete_all") settings.component_fis_delete_dirty_only = false;
        else if (arg == "--component_fis_delete_max_bel_free") settings.component_fis_delete_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--component_fis_delete_max_conflict") settings.component_fis_delete_max_conflict = argv_util::next_double(arg);
        else if (arg == "--component_fis_delete_min_plaus_surface") settings.component_fis_delete_min_plaus_surface = argv_util::next_double(arg);
        else if (arg == "--enable_component_radius_shrink") settings.enable_component_radius_shrink = true;
        else if (arg == "--disable_component_radius_shrink") settings.enable_component_radius_shrink = false;
        else if (arg == "--component_radius_shrink_dirty_only") settings.component_radius_shrink_dirty_only = true;
        else if (arg == "--component_radius_shrink_all") settings.component_radius_shrink_dirty_only = false;
        else if (arg == "--component_shrink_edge_over_radius") settings.component_shrink_edge_over_radius = argv_util::next_double(arg);
        else if (arg == "--component_shrink_min_component_faces") settings.component_shrink_min_component_faces = argv_util::next_int(arg);
        else if (arg == "--enable_component_adaptive_simplification") settings.enable_component_adaptive_simplification = true;
        else if (arg == "--disable_component_adaptive_simplification") settings.enable_component_adaptive_simplification = false;
        else if (arg == "--component_simplify_min_component_faces") settings.component_simplify_min_component_faces = argv_util::next_int(arg);
        else if (arg == "--component_simplify_centroid_radius_factor") settings.component_simplify_centroid_radius_factor = argv_util::next_double(arg);
        else if (arg == "--component_simplify_normal_dot") settings.component_simplify_normal_dot = argv_util::next_double(arg);
        else if (arg == "--component_simplify_preserve_boundary_faces") settings.component_simplify_preserve_boundary_faces = true;
        else if (arg == "--component_simplify_allow_boundary_thinning") settings.component_simplify_preserve_boundary_faces = false;
        else if (arg == "--component_growth_keep_dirty_after_mesh") settings.component_growth_clear_dirty_after_mesh = false;
        else if (arg == "--component_growth_clear_dirty_after_mesh") settings.component_growth_clear_dirty_after_mesh = true;
        else if (arg == "--mesh_output")          settings.mesh_output          = argv_util::next_raw(arg);
        else if (arg == "--mesh_dump_dir")        settings.mesh_dump_dir        = argv_util::next_raw(arg);
        else if (arg == "--mesh_every")           settings.mesh_every           = argv_util::next_int(arg);
        else if (arg == "--mesh_submap_scans")    settings.mesh_submap_scans    = argv_util::next_int(arg);
        else if (arg == "--mesh_min_hit_count")   settings.mesh_min_hit_count   = argv_util::next_int(arg);
        else if (arg == "--mesh_max_sqrt_residual") settings.mesh_max_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--mesh_min_consistency") settings.mesh_min_normal_consistency = argv_util::next_double(arg);
        else if (arg == "--mesh_normal_dot")      settings.mesh_normal_dot      = argv_util::next_double(arg);
        else if (arg == "--mesh_neighbor_radius_factor") settings.mesh_neighbor_radius_factor = argv_util::next_double(arg);
        else if (arg == "--mesh_max_edge_factor") settings.mesh_max_edge_factor = argv_util::next_double(arg);
        else if (arg == "--mesh_triangle_normal_dot") settings.mesh_triangle_normal_dot = argv_util::next_double(arg);
        else if (arg == "--mesh_min_area_factor") settings.mesh_min_area_factor = argv_util::next_double(arg);
        else if (arg == "--mesh_max_boundary_ratio") settings.mesh_max_boundary_ratio = argv_util::next_double(arg);
        else if (arg == "--mesh_max_fan_angle")   settings.mesh_max_fan_angle   = argv_util::next_double(arg);
        else if (arg == "--mesh_no_free_reject")  settings.mesh_reject_free_centroid = false;
        else if (arg == "--enable_vertex_smoothing") settings.enable_vertex_smoothing = true;
        else if (arg == "--disable_vertex_smoothing") settings.enable_vertex_smoothing = false;
        else if (arg == "--vertex_smooth_iters") settings.vertex_smooth_iters = argv_util::next_int(arg);
        else if (arg == "--vertex_smooth_apply_to_confirmed") settings.vertex_smooth_apply_to_confirmed = true;
        else if (arg == "--vertex_smooth_no_apply_to_confirmed") settings.vertex_smooth_apply_to_confirmed = false;
        else if (arg == "--vertex_smooth_allow_confirmed_flat") settings.vertex_smooth_allow_confirmed_flat = true;
        else if (arg == "--vertex_smooth_protect_confirmed_flat") settings.vertex_smooth_allow_confirmed_flat = false;
        else if (arg == "--vertex_smooth_preserve_edges") settings.vertex_smooth_preserve_edges = true;
        else if (arg == "--vertex_smooth_no_preserve_edges") settings.vertex_smooth_preserve_edges = false;
        else if (arg == "--vertex_smooth_normal_only") settings.vertex_smooth_normal_only = true;
        else if (arg == "--vertex_smooth_full_vector") settings.vertex_smooth_normal_only = false;
        else if (arg == "--vertex_smooth_qem_project") settings.vertex_smooth_qem_project = true;
        else if (arg == "--vertex_smooth_no_qem_project") settings.vertex_smooth_qem_project = false;
        else if (arg == "--vertex_smooth_min_degree") settings.vertex_smooth_min_degree = argv_util::next_int(arg);
        else if (arg == "--vertex_smooth_min_hit_count") settings.vertex_smooth_min_hit_count = argv_util::next_int(arg);
        else if (arg == "--vertex_smooth_radius_factor") settings.vertex_smooth_radius_factor = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_sigma_spatial_factor") settings.vertex_smooth_sigma_spatial_factor = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_sigma_normal") settings.vertex_smooth_sigma_normal = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_sigma_plane_factor") settings.vertex_smooth_sigma_plane_factor = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_normal_dot") settings.vertex_smooth_normal_dot = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_min_consistency") settings.vertex_smooth_min_normal_consistency = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_min_sqrt_residual") settings.vertex_smooth_min_sqrt_residual = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_max_move_factor") settings.vertex_smooth_max_move_factor = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_anchor_scale") settings.vertex_smooth_anchor_scale = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_candidate_scale") settings.vertex_smooth_candidate_scale = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_residual_tol_factor") settings.vertex_smooth_residual_tol_factor = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_max_bel_free") settings.vertex_smooth_max_bel_free = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_max_conflict") settings.vertex_smooth_max_conflict = argv_util::next_double(arg);
        else if (arg == "--vertex_smooth_max_boundary_ratio") settings.vertex_smooth_max_boundary_ratio = argv_util::next_double(arg);
        else if (arg == "--help" || arg == "-h")  { print_usage(); return 0; }
        else {
            std::cerr << "Unknown arg: " << arg << "\n\n";
            print_usage();
            return 1;
        }
    }
    if (pcd_folder.empty() || pose_file.empty()) {
        print_usage();
        return 1;
    }
    if (!fs::exists(pcd_folder)) {
        std::cerr << "Error: pcd_folder " << pcd_folder << " does not exist\n";
        return 1;
    }
    if (!fs::exists(pose_file)) {
        std::cerr << "Error: pose_file " << pose_file << " does not exist\n";
        return 1;
    }

    if (saw_disable_scanline && saw_pandar_qt64 && !settings.enable_scanline) {
        std::fprintf(stderr, "  [WARN] --pandar_qt64/--sensor pandar_qt64 requested but scan-line geometry is disabled; PandarQT64 beam-neighbor logic will not run.\n");
    }
    if (saw_pandar_uniform && saw_pandar_infer) {
        std::fprintf(stderr, "  [WARN] both --pandar_qt64_uniform and --pandar_qt64_infer were provided; last parsed flag wins.\n");
    }
    if (settings.pandar_qt64_mode && settings.pandar_qt64_channels != 64 && settings.pandar_qt64_calib.empty() &&
        !settings.pandar_qt64_uniform && !settings.pandar_qt64_infer_rings) {
        std::fprintf(stderr, "  [WARN] PandarQT64 design table has 64 channels; --pandar_qt64_channels is ignored unless using --pandar_qt64_calib, --pandar_qt64_uniform, or --pandar_qt64_infer.\n");
    }

    // Sparse-region scaffold structuring: turn on the hierarchical scaffold
    // inheritance + fill machinery so voxels too sparse to be filled by raw
    // points at the working resolution receive virtual inherited QEM planes /
    // normals / evidence from a coarse level. Power users who want manual
    // control over the scaffold flags can pass --disable_sparse_scaffold and
    // configure --enable_hierarchical_scaffold / --scaffold_* directly.
    if (settings.sparse_region_scaffold) {
        settings.enable_hierarchical_scaffold = true;       // inherit coarse QEM into sparse cells
        settings.enable_hierarchical_scaffold_fill = true;  // tile inherited cells into faces
        // Singular scaffold: exactly one fallback level at 2x the voxel size
        // (0.1 m voxel -> 0.2 m parent, 2^3 = 8 children). When a 0.1 m voxel
        // lacks evidence, the 0.2 m parent's QEM/plane is redistributed to all 8
        // children; children that later accumulate real hits override the
        // inherited QEM (via inherited_decay_scale), while un-updated children
        // keep the inherited QEM. No coarser (0.4 m+) levels are used.
        settings.scaffold_base_factor = 2;
        settings.scaffold_max_level   = 1;
        settings.scaffold_inherit_into_real_cells = true;   // redistribute to all 8 children
    }

    int n_threads = 1;
    #ifdef HAS_OPENMP
    n_threads = omp_get_max_threads();
    #endif
    std::printf("[qem_voxel_map] threads=%d  Eigen=%d.%d.%d\n",
                n_threads,
                EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    std::printf("  voxel_size=%.3f range_precision=%.4f min_incident_cos=%.2f "
                "process_every_n=%d carve=%s\n",
                settings.voxel_size, settings.range_precision,
                settings.min_incident_cos, settings.process_every_n,
                settings.carve_rays ? "yes" : "no");
    std::printf("  Probabilistic planes: %s bearing_sigma=%.6f pose_trans_sigma=%.4f pose_rot_sigma=%.6f gate_sigma=%.2f sigma=[%.3f, %.3f] qem_ref=%.4f sample_cap=50\n",
                settings.enable_probabilistic_planes ? "on" : "off",
                settings.bearing_sigma_rad, settings.pose_trans_sigma, settings.pose_rot_sigma_rad,
                settings.prob_gate_sigma, settings.prob_min_sigma, settings.prob_max_sigma,
                settings.prob_qem_ref_sigma);
    std::printf("  Seed voxels: %s min_incident=%.2f weight_scale=%.2f promote_hits=%d neigh_promote_hits=%d export=%s\n",
                settings.enable_seed_voxels ? "on" : "off",
                settings.seed_min_incident_cos, settings.seed_hit_weight_scale,
                settings.seed_promote_min_hits, settings.seed_promote_neighbor_min_hits,
                settings.export_seed_vertices ? "yes" : "no");
    std::printf("  NVT/BEO normals: %s  scope=%s  k=%d iters=%d rho_cos=%.2f tau=%.2f damping=%.2f min_conf=%.2f soft_floor=%.2f max_radius=%.3f\n",
                settings.enable_nvt ? "on" : "off",
                settings.nvt_only_for_pca ? "pca-only" : "all-normals",
                settings.nvt_k, settings.nvt_iters, settings.nvt_rho_cos,
                settings.nvt_tau, settings.nvt_damping, settings.nvt_min_conf,
                settings.nvt_conf_soft_floor,
                settings.nvt_max_radius > 0.0 ? settings.nvt_max_radius : 5.0 * settings.voxel_size);
    if (settings.nvt_iters > 1)
        std::fprintf(stderr, "  [NVT] warning: nvt_iters > 1 may oversmooth sharp LiDAR features.\n");
    std::printf("  Scan-line geometry: %s normals=%s rows=%d cols=%d jump=%.2fm+%.2fr boundary_conf=%.2f\n",
                settings.enable_scanline ? "on" : "off",
                settings.scanline_normals ? "on" : "off",
                settings.scanline_rows, settings.scanline_cols,
                settings.scanline_jump_abs, settings.scanline_jump_rel,
                settings.scanline_boundary_conf);
    if (settings.pandar_qt64_mode) {
        std::printf("  PandarQT64 geometry: on  source=%s  calib=%s\n",
                    settings.pandar_qt64_calib.empty()
                        ? (settings.pandar_qt64_infer_rings ? "inferred" : (settings.pandar_qt64_uniform ? "uniform" : "datasheet-design"))
                        : "unit-calibration-file",
                    settings.pandar_qt64_calib.empty() ? "none" : settings.pandar_qt64_calib.c_str());
    }

    std::printf("  Hierarchical scaffold: %s fill=%s max_level=%d base_factor=%d min_children=%d edge=%.2f inherit_weight=%.3f decay_ref=%.3f inherited_corner_sign=%s sign_w=%.3f fill_crossing=%s\n",
                (settings.enable_hierarchical_scaffold || settings.enable_planar_scaffold_inherit || settings.enable_planar_scaffold) ? "on" : "off",
                settings.enable_hierarchical_scaffold_fill ? "on" : "off",
                settings.scaffold_max_level, settings.scaffold_base_factor,
                settings.scaffold_min_parent_children, settings.scaffold_max_edge_factor,
                settings.scaffold_inherit_weight, settings.scaffold_inherit_decay_weight_ref,
                settings.scaffold_inherited_corner_sign ? "on" : "off",
                settings.scaffold_inherited_sign_weight,
                settings.scaffold_fill_require_plane_crossing ? "yes" : "no");
    std::printf("  Mesh mode: %s  dc_require_free=%s dc_normal_dot=%.2f dc_max_edge=%.2f\n",
                settings.mesh_mode.c_str(), settings.dc_require_free ? "yes" : "no",
                settings.dc_normal_dot, settings.dc_max_edge_factor);
    std::string halo_desc = settings.persistent_mesh_halo_voxels > 0
                                ? std::to_string(settings.persistent_mesh_halo_voxels)
                                : std::string("auto");
    std::printf("  Persistent incremental mesh: %s  halo_voxels=%s\n",
                settings.persistent_incremental_mesh ? "on (remesh dirty regions only)" : "off (full rebuild each export)",
                halo_desc.c_str());
    std::printf("  Seam-aware hole closing: %s  iters=%d  retire_free_faces=%s\n",
                settings.enable_seam_closing ? "on" : "off", settings.seam_close_iters,
                settings.persistent_retire_free_faces ? "on" : "off");
    std::printf("  Sparse-region scaffold: %s  singular fallback @ %.3fm (=%dx voxel, %d children), max_level=%d\n",
                settings.sparse_region_scaffold ? "on" : "off",
                settings.voxel_size * settings.scaffold_base_factor,
                settings.scaffold_base_factor,
                settings.scaffold_base_factor * settings.scaffold_base_factor * settings.scaffold_base_factor,
                settings.scaffold_max_level);
    std::printf("  Component growth: %s iters=%d edge=%.2f radius=%.2f/%.2f normal_dot=%.2f comp_dot=%.2f plane=%.2f confirmed_anchor=%s persistent=%s dirty_only=%s dirty_rad=%d search=%s\n",
                settings.enable_component_growth ? "on" : "off",
                settings.component_growth_iters,
                settings.component_growth_max_edge_factor,
                settings.component_growth_boundary_radius_factor,
                settings.component_growth_flat_radius_factor,
                settings.component_growth_normal_dot,
                settings.component_growth_component_normal_dot,
                settings.component_growth_plane_dist_factor,
                settings.component_growth_require_confirmed_anchor ? "yes" : "no",
                settings.component_growth_persistent_state ? "yes" : "no",
                settings.component_growth_dirty_only ? "yes" : "no",
                settings.component_growth_dirty_radius_voxels,
                settings.component_growth_use_rrs_boundary_search ? "rrs" : "edge");
    std::printf("  Component persistence reuse: %s dirty_faces=%s retire_free=%.2f retire_conflict=%.2f retire_plaus=%.2f fis_delete=%s shrink=%s\n",
                settings.component_persistent_reuse_faces ? "on" : "off",
                settings.component_persistent_reuse_dirty_faces ? "reuse_if_not_retired" : "skip",
                settings.component_persistent_reuse_max_bel_free,
                settings.component_persistent_reuse_max_conflict,
                settings.component_persistent_reuse_min_plaus_surface,
                settings.enable_component_fis_delete ? "on" : "off",
                settings.enable_component_radius_shrink ? "on" : "off");
    std::printf("  Vertex smoothing: %s iters=%d normal_only=%s qem_project=%s apply_confirmed=%s protect_confirmed_flat=%s preserve_edges=%s\n",
                settings.enable_vertex_smoothing ? "on" : "off",
                settings.vertex_smooth_iters,
                settings.vertex_smooth_normal_only ? "yes" : "no",
                settings.vertex_smooth_qem_project ? "yes" : "no",
                settings.vertex_smooth_apply_to_confirmed ? "yes" : "no",
                settings.vertex_smooth_allow_confirmed_flat ? "no" : "yes",
                settings.vertex_smooth_preserve_edges ? "yes" : "no");
    if ((settings.mesh_mode == "dual" || settings.mesh_mode == "dc") && settings.dc_require_free && !settings.carve_rays) {
        std::fprintf(stderr, "  [WARN] mesh_mode=dual with --no_carve and dc_require_free=yes will usually produce few/no faces. Use --dc_no_require_free or enable carving.\n");
    }

    auto dataset = load_dataset(pcd_folder, pose_file);
    if (dataset.empty()) {
        std::cerr << "No PCDs matched to poses; nothing to do.\n";
        return 1;
    }
    if (settings.num_scans > 0 && settings.num_scans < (int)dataset.size())
        dataset.resize(settings.num_scans);
    std::printf("Processing %d scans\n", (int)dataset.size());

    VoxelQEMMap map(settings);
    g_t0 = now_sec();
    g_map = &map;
    g_output = output;
    std::signal(SIGINT, on_sigint);

    std::string snap_base = fs::path(output).stem().string();
    std::string snap_ext  = fs::path(output).extension().string();
    std::string snap_dir  = fs::path(output).parent_path().string();
    if (snap_dir.empty()) snap_dir = ".";

    for (int i = 0; i < (int)dataset.size(); i++) {
        std::printf("\n[%d/%d] %s\n", i + 1, (int)dataset.size(),
                    fs::path(dataset[i].pcd_path).filename().c_str());
        auto pc = read_pcd(dataset[i].pcd_path);
        std::printf("  %d pts%s\n", (int)pc.points.size(),
                    pc.normals.empty() ? " (PCA normals)" : " + normals");
        map.process_scan(pc, dataset[i].pose, i);
        map.dump_scan_viz(i);
        map.maybe_dump_mesh_snapshot(i);

        if (settings.snapshot_interval > 0 &&
            (i + 1) % settings.snapshot_interval == 0) {
            char snap_path[1024];
            std::snprintf(snap_path, sizeof(snap_path), "%s/%s_snap_%04d%s",
                          snap_dir.c_str(), snap_base.c_str(),
                          i + 1, snap_ext.c_str());
            map.export_ply_and_csv(snap_path);
        }
        if (g_interrupted) break;
    }

    std::printf("\nMain loop: %.1fs\n", now_sec() - g_t0);
    long final_promoted = map.finalize_seed_promotions(6);
    if (final_promoted > 0) {
        std::printf("Final seed promotion cascade: %ld cells\n", final_promoted);
    }
    map.print_summary();
    map.export_ply_and_csv(output);
    if (!settings.mesh_output.empty()) {
        map.export_mesh_ply(settings.mesh_output);
    }
    std::printf("Total: %.1fs\n", now_sec() - g_t0);
    return 0;
}