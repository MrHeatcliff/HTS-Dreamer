import csv
import json
from pathlib import Path


FIELDS = [
    "config_name",
    "latent_anchor",
    "latent_anchor_dim",
    "latent_anchor_source",
    "num_levels",
    "head_dim",
    "total_dictionary_width",
    "sparse_active_budget",
    "dense_active_dimensions",
    "activation_mode",
    "topk_per_level",
    "decoder_count",
    "flat_reconstruction",
    "prefix_reconstruction",
    "prefix_stop_gradient",
    "reconstruction_loss_enabled",
    "horizons",
    "action_subsequence_encoder",
    "temporal_loss",
    "temporal_k_pos",
    "temporal_temperature",
    "far_negative_default",
    "vc_loss",
    "sparse_loss",
    "beta_hier",
    "alpha_sdyn",
    "projector",
    "projector_dim",
    "predictor_hidden",
    "action_units",
    "training_regime",
    "hier_module_instantiated",
    "hier_loss_enabled",
    "sdyn_module_instantiated",
    "sdyn_loss_enabled",
    "temp_module_instantiated",
    "temp_loss_enabled",
    "vc_module_instantiated",
    "vc_loss_enabled",
    "loss_enabled",
    "gradient_expected",
    "module_instantiated",
    "param_count_source",
    "target_addon_params",
    "selected_width",
    "actual_addon_params",
    "relative_param_gap",
    "search_status",
    "total_params",
    "implementation_status",
    "actual_initialization_verified",
    "smoke_status",
    "unit_test_status",
]

V4_FIELDS = [
    "config_name", "latent_anchor", "latent_anchor_dim",
    "latent_anchor_source", "num_levels", "head_dim",
    "total_dictionary_width", "sparse_active_budget",
    "dense_active_dimensions", "activation_mode", "topk_per_level",
    "decoder_count", "flat_reconstruction", "prefix_reconstruction",
    "prefix_stop_gradient", "reconstruction_loss_enabled", "horizons",
    "action_subsequence_encoder", "temporal_loss", "temporal_k_pos",
    "temporal_temperature", "far_negative_default", "vc_loss",
    "sparse_loss", "beta_hier", "alpha_sdyn", "projector",
    "projector_dim", "predictor_hidden", "action_units",
    "training_regime", "hier_module_instantiated", "hier_loss_enabled",
    "sdyn_module_instantiated", "sdyn_loss_enabled",
    "temp_module_instantiated", "temp_loss_enabled",
    "vc_module_instantiated", "vc_loss_enabled", "loss_enabled",
    "gradient_expected", "implementation_exists",
    "debug_init_smoke_verified", "size12m_init_verified",
    "forward_verified", "backward_verified", "optimizer_step_verified",
    "checkpoint_save_verified", "checkpoint_reload_verified",
    "artifact_write_verified", "param_count_source",
    "analytical_addon_params", "actual_addon_params",
    "actual_total_params", "target_addon_params", "relative_param_gap",
    "selected_width", "search_status", "unit_test_status",
]


NA = "N/A"
T = "true"
F = "false"
LATENT_ANCHOR = "rssm_repfeat"
LATENT_SOURCE = "dreamerv3.rssm.RSSM.loss"
LATENT_DIM_SIZE12M = 2560
DREAMER_SIZE12M_PARAMS_ALIEN = 10498772
LEVELS = 6
HEAD_DIM = 32
WIDTH = LEVELS * HEAD_DIM
TOPKS = [8, 8, 8, 8, 8, 8]
STRIDES = [32, 16, 8, 4, 2, 1]
BETA = [0.1666666667] * LEVELS
ALPHA = [0.1666666667] * LEVELS


def linear_params(insize, outsize, bias=True):
  return int(insize) * int(outsize) + (int(outsize) if bias else 0)


def norm_params(units):
  return int(units)


def mlp_params(insize, units, layers):
  total = 0
  cur = int(insize)
  for _ in range(int(layers)):
    total += linear_params(cur, units)
    total += norm_params(units)
    cur = int(units)
  return total


def hts_full_addon_params(
    feat_dim=LATENT_DIM_SIZE12M, action_dim=18, hidden=256,
    action_units=128, proj_dim=64):
  total = mlp_params(feat_dim, hidden, 2)
  total += LEVELS * linear_params(hidden, HEAD_DIM)
  for level in range(LEVELS):
    prefix = (level + 1) * HEAD_DIM
    total += mlp_params(prefix, hidden, 2)
    total += linear_params(hidden, feat_dim)
    total += mlp_params(action_dim * STRIDES[level], action_units, 1)
    total += mlp_params(prefix + action_units, hidden, 2)
    total += linear_params(hidden, feat_dim)
  total += mlp_params(HEAD_DIM, proj_dim, 1)
  total += linear_params(proj_dim, proj_dim)
  return total


def flat_sae_params(feat_dim=LATENT_DIM_SIZE12M, code_width=WIDTH, hidden=256):
  return (
      mlp_params(feat_dim, hidden, 2) +
      linear_params(hidden, code_width) +
      mlp_params(code_width, hidden, 2) +
      linear_params(hidden, feat_dim))


def flat_mh_params(
    feat_dim=LATENT_DIM_SIZE12M, action_dim=18, code_width=WIDTH,
    hidden=256, action_units=128):
  total = mlp_params(feat_dim, hidden, 2) + linear_params(hidden, code_width)
  for horizon in [1, 2, 4, 8, 16, 32]:
    total += mlp_params(action_dim * horizon, action_units, 1)
    total += mlp_params(code_width + action_units, hidden, 2)
    total += linear_params(hidden, feat_dim)
  return total


def sgf_style_params(feat_dim=LATENT_DIM_SIZE12M, proj_dim=64, hidden=256):
  # Raw one-hot action is concatenated directly; no learned action encoder.
  action_dim = 18
  return (
      mlp_params(feat_dim, hidden, 2) +
      linear_params(hidden, proj_dim) +
      mlp_params(proj_dim + action_dim, hidden, 2) +
      linear_params(hidden, feat_dim))


def dense_hierarchy_params(feat_dim=LATENT_DIM_SIZE12M, hidden=256):
  total = mlp_params(feat_dim, hidden, 2)
  total += LEVELS * linear_params(hidden, HEAD_DIM)
  for level in range(LEVELS):
    prefix = (level + 1) * HEAD_DIM
    total += mlp_params(prefix, hidden, 2)
    total += linear_params(hidden, feat_dim)
  return total


ADDON_HTS_EST = hts_full_addon_params()
ADDON_FLAT_MH_EST = flat_mh_params()


def row(name, **kw):
  base = {
      "config_name": name,
      "latent_anchor": LATENT_ANCHOR,
      "latent_anchor_dim": LATENT_DIM_SIZE12M,
      "latent_anchor_source": LATENT_SOURCE,
      "num_levels": LEVELS,
      "head_dim": HEAD_DIM,
      "total_dictionary_width": WIDTH,
      "sparse_active_budget": NA,
      "dense_active_dimensions": NA,
      "activation_mode": "dense",
      "topk_per_level": "[]",
      "decoder_count": 0,
      "flat_reconstruction": F,
      "prefix_reconstruction": F,
      "prefix_stop_gradient": F,
      "reconstruction_loss_enabled": F,
      "horizons": "[]",
      "action_subsequence_encoder": F,
      "temporal_loss": "none",
      "temporal_k_pos": 0,
      "temporal_temperature": 0.0,
      "far_negative_default": "none",
      "vc_loss": F,
      "sparse_loss": F,
      "beta_hier": "[]",
      "alpha_sdyn": "[]",
      "projector": "none",
      "projector_dim": 0,
      "predictor_hidden": 0,
      "action_units": 0,
      "training_regime": "joint",
      "hier_module_instantiated": F,
      "hier_loss_enabled": F,
      "sdyn_module_instantiated": F,
      "sdyn_loss_enabled": F,
      "temp_module_instantiated": F,
      "temp_loss_enabled": F,
      "vc_module_instantiated": F,
      "vc_loss_enabled": F,
      "loss_enabled": F,
      "gradient_expected": F,
      "module_instantiated": F,
      "param_count_source": "not_available",
      "target_addon_params": NA,
      "selected_width": NA,
      "actual_addon_params": NA,
      "relative_param_gap": NA,
      "search_status": NA,
      "total_params": NA,
      "implementation_status": "not_implemented_official",
      "actual_initialization_verified": F,
      "smoke_status": "not_run",
      "unit_test_status": "not_run",
  }
  base.update(kw)
  return base


def hts_common(**kw):
  data = dict(
      sparse_active_budget=sum(TOPKS),
      activation_mode="sparse_topk",
      topk_per_level=json.dumps(TOPKS),
      decoder_count=LEVELS,
      flat_reconstruction=F,
      prefix_reconstruction=T,
      prefix_stop_gradient=T,
      reconstruction_loss_enabled=T,
      horizons=json.dumps(STRIDES),
      action_subsequence_encoder="concat_actions_mlp",
      temporal_loss="masked_infonce_projected_z1",
      temporal_k_pos=4,
      temporal_temperature=0.1,
      far_negative_default="none",
      vc_loss=T,
      sparse_loss="level_topk+l1",
      beta_hier=json.dumps(BETA),
      alpha_sdyn=json.dumps(ALPHA),
      projector="mlp_on_z1",
      projector_dim=64,
      predictor_hidden=256,
      action_units=128,
      hier_module_instantiated=T,
      hier_loss_enabled=T,
      sdyn_module_instantiated=T,
      sdyn_loss_enabled=T,
      temp_module_instantiated=T,
      temp_loss_enabled=T,
      vc_module_instantiated=T,
      vc_loss_enabled=T,
      loss_enabled=T,
      gradient_expected=T,
      module_instantiated=T,
      param_count_source="analytical_estimate",
      target_addon_params=ADDON_HTS_EST,
      actual_addon_params=NA,
      total_params=NA,
  )
  data.update(kw)
  return data


def rows():
  return [
      row(
          "dreamer_anchor",
          num_levels=0, head_dim=0, total_dictionary_width=0,
          activation_mode="none", param_count_source="initialized_model",
          actual_addon_params=0, total_params=DREAMER_SIZE12M_PARAMS_ALIEN,
          implementation_status="implemented",
          actual_initialization_verified=T, smoke_status="pass",
          unit_test_status="RT-01 pending"),
      row(
          "hts_full",
          **hts_common(
              implementation_status="implemented",
              actual_initialization_verified=F,
              smoke_status="debug_smoke_pass",
              unit_test_status="UT-01..UT-11 pass")),
      row(
          "flat_sae",
          num_levels=1, head_dim=WIDTH, total_dictionary_width=WIDTH,
          sparse_active_budget=48, activation_mode="sparse_topk",
          topk_per_level="[48]", decoder_count=1,
          flat_reconstruction=T, prefix_reconstruction=F,
          prefix_stop_gradient=NA, reconstruction_loss_enabled=T,
          sparse_loss="topk+l1", projector="flat_sparse_code",
          hier_module_instantiated=F, hier_loss_enabled=F,
          sdyn_module_instantiated=F, sdyn_loss_enabled=F,
          temp_module_instantiated=F, temp_loss_enabled=F,
          vc_module_instantiated=F, vc_loss_enabled=F,
          loss_enabled=T, gradient_expected=T, module_instantiated=T,
          param_count_source="analytical_estimate",
          target_addon_params=flat_sae_params(), total_params=NA,
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass"),
      row(
          "flat_mh",
          num_levels=1, head_dim=WIDTH, total_dictionary_width=WIDTH,
          dense_active_dimensions=WIDTH, activation_mode="dense",
          horizons=json.dumps([1, 2, 4, 8, 16, 32]),
          action_subsequence_encoder="concat_actions_mlp",
          projector="flat_dense_code", predictor_hidden=256,
          action_units=128, sdyn_module_instantiated=T,
          sdyn_loss_enabled=T, loss_enabled=T, gradient_expected=T,
          param_count_source="analytical_estimate",
          target_addon_params=ADDON_FLAT_MH_EST,
          module_instantiated=T,
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass"),
      row(
          "flat_partition_dim_matched",
          num_levels=6, head_dim=HEAD_DIM, total_dictionary_width=WIDTH,
          dense_active_dimensions=WIDTH, activation_mode="dense_partitioned",
          topk_per_level="[]", decoder_count=1,
          flat_reconstruction=T, prefix_reconstruction=F,
          prefix_stop_gradient=NA, reconstruction_loss_enabled=T,
          horizons="[]", action_subsequence_encoder=F,
          temporal_loss="none", vc_loss=F, sparse_loss=F,
          projector="six_equal_dense_partitions",
          projector_dim=WIDTH, predictor_hidden=0, action_units=0,
          hier_module_instantiated=F, hier_loss_enabled=F,
          sdyn_module_instantiated=F, sdyn_loss_enabled=F,
          temp_module_instantiated=F, temp_loss_enabled=F,
          vc_module_instantiated=F, vc_loss_enabled=F,
          loss_enabled=T, gradient_expected=T, module_instantiated=T,
          param_count_source="analytical_estimate",
          target_addon_params=mlp_params(LATENT_DIM_SIZE12M, 256, 2) + linear_params(256, WIDTH),
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass",
          unit_test_status="UT-15-P0 pending"),
      row(
          "sgf_style_flat_same_code",
          num_levels=1, head_dim=64, total_dictionary_width=64,
          dense_active_dimensions=64, activation_mode="dense",
          horizons="[1]", action_subsequence_encoder="raw_a_t",
          vc_loss=T, vc_module_instantiated=T, vc_loss_enabled=T,
          projector="dense_projector", projector_dim=64,
          predictor_hidden=256, action_units="action_dim",
          sdyn_module_instantiated=T, sdyn_loss_enabled=T,
          loss_enabled=T, gradient_expected=T,
          param_count_source="analytical_estimate",
          target_addon_params=sgf_style_params(),
          module_instantiated=T,
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass"),
      row(
          "recon_only_hierarchy",
          dense_active_dimensions=WIDTH, activation_mode="dense",
          decoder_count=LEVELS, flat_reconstruction=F,
          prefix_reconstruction=T, prefix_stop_gradient=T,
          reconstruction_loss_enabled=T, beta_hier=json.dumps(BETA),
          projector="dense_heads", hier_module_instantiated=T,
          hier_loss_enabled=T, loss_enabled=T, gradient_expected=T,
          param_count_source="analytical_estimate",
          target_addon_params=dense_hierarchy_params(),
          module_instantiated=T,
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass"),
      row(
          "matryoshka_only",
          sparse_active_budget=sum(TOPKS), activation_mode="sparse_topk",
          topk_per_level=json.dumps(TOPKS), decoder_count=LEVELS,
          flat_reconstruction=F, prefix_reconstruction=T,
          prefix_stop_gradient=T, reconstruction_loss_enabled=T,
          beta_hier=json.dumps(BETA), sparse_loss="level_topk+l1",
          projector="sparse_heads", hier_module_instantiated=T,
          hier_loss_enabled=T, loss_enabled=T, gradient_expected=T,
          param_count_source="analytical_estimate",
          target_addon_params=dense_hierarchy_params(),
          module_instantiated=T,
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass"),
      row(
          "dense_multistride_no_sparse",
          **hts_common(
              sparse_active_budget=NA, dense_active_dimensions=WIDTH,
              activation_mode="dense", topk_per_level="[]",
              sparse_loss=F, implementation_status="implemented",
              actual_initialization_verified=T,
              smoke_status="debug_smoke_pass",
              unit_test_status="RT-07 pending")),
      row(
          "larger_flat_param",
          num_levels=1, head_dim="searched",
          total_dictionary_width="searched",
          dense_active_dimensions="searched", activation_mode="dense",
          horizons=json.dumps([1, 2, 4, 8, 16, 32]),
          action_subsequence_encoder="concat_actions_mlp",
          projector="flat_dense_code", predictor_hidden=256,
          action_units=128, sdyn_module_instantiated=T,
          sdyn_loss_enabled=T, loss_enabled=T, gradient_expected=T,
          module_instantiated=T, param_count_source="target_placeholder",
          target_addon_params=ADDON_HTS_EST, selected_width=2648,
          actual_addon_params=NA, relative_param_gap=NA,
          search_status="analytical_selected_actual_count_pending",
          total_params=NA,
          implementation_status="implemented",
          actual_initialization_verified=T,
          smoke_status="debug_smoke_pass"),
      row(
          "larger_flat_flops",
          num_levels=1, head_dim=NA, total_dictionary_width=NA,
          dense_active_dimensions=NA, activation_mode=NA,
          param_count_source="not_available", target_addon_params=NA,
          selected_width=NA, actual_addon_params=NA,
          relative_param_gap=NA, search_status="P1_pending",
          total_params=NA, implementation_status="P1_pending"),
      row(
          "hts_no_temp",
          **hts_common(
              temporal_loss="disabled", temporal_k_pos=0,
              temporal_temperature=0.0, temp_module_instantiated=T,
              temp_loss_enabled=F, implementation_status="implemented",
              actual_initialization_verified=T,
              smoke_status="debug_smoke_pass",
              unit_test_status="RT-03 pending")),
      row(
          "hts_no_vc",
          **hts_common(
              vc_loss=F, vc_module_instantiated=T, vc_loss_enabled=F,
              implementation_status="implemented",
              actual_initialization_verified=T,
              smoke_status="debug_smoke_pass",
              unit_test_status="RT-04 pending")),
      row(
          "hts_no_hier",
          **hts_common(
              prefix_stop_gradient="dormant_module",
              reconstruction_loss_enabled=F,
              hier_module_instantiated=T, hier_loss_enabled=F,
              implementation_status="implemented",
              actual_initialization_verified=T,
              smoke_status="debug_smoke_pass",
              unit_test_status="RT-05 pending")),
      row(
          "hts_no_sdyn",
          **hts_common(
              horizons=json.dumps(STRIDES),
              sdyn_module_instantiated=T, sdyn_loss_enabled=F,
              gradient_expected="predictor_grad_zero",
              implementation_status="implemented",
              actual_initialization_verified=T,
              smoke_status="debug_smoke_pass",
              unit_test_status="RT-06 pending")),
  ]


def write(output):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  data = rows()
  with (output / "component_matrix.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(data)
  with (output / "component_matrix.json").open("w") as file:
    json.dump(data, file, indent=2)
  with (output / "component_matrix_v3.json").open("w") as file:
    json.dump(data, file, indent=2)
  with (output / "component_matrix_v3.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(data)
  with (output / "component_matrix.md").open("w") as file:
    file.write(f"Exact row count: {len(data)}\n\n")
    file.write("| " + " | ".join(FIELDS) + " |\n")
    file.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
    for item in data:
      file.write("| " + " | ".join(str(item.get(field, "")) for field in FIELDS) + " |\n")
  with (output / "component_matrix_v3.md").open("w") as file:
    file.write(f"Exact row count: {len(data)}\n\n")
    file.write("| " + " | ".join(FIELDS) + " |\n")
    file.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
    for item in data:
      file.write("| " + " | ".join(str(item.get(field, "")) for field in FIELDS) + " |\n")
  _write_parity(output, data)
  v4 = [_to_v4(row) for row in data]
  with (output / "component_matrix_v4.json").open("w") as file:
    json.dump(v4, file, indent=2)
  with (output / "component_matrix_v4.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=V4_FIELDS)
    writer.writeheader()
    writer.writerows(v4)
  with (output / "component_matrix_v4.md").open("w") as file:
    file.write(f"Exact row count: {len(v4)}\n\n")
    file.write("| " + " | ".join(V4_FIELDS) + " |\n")
    file.write("| " + " | ".join(["---"] * len(V4_FIELDS)) + " |\n")
    for item in v4:
      file.write("| " + " | ".join(str(item.get(field, "")) for field in V4_FIELDS) + " |\n")
  _write_parity_v4(output, v4)


def _bool(value):
  return str(value).lower() == "true"


def _to_v4(row):
  impl = row.get("implementation_status") not in (
      "not_implemented_official", "P1_pending")
  debug = row.get("smoke_status") == "debug_smoke_pass" or row["config_name"] == "dreamer_anchor"
  analytical = row.get("target_addon_params", NA)
  item = {field: row.get(field, NA) for field in V4_FIELDS}
  item.update({
      "implementation_exists": T if impl else F,
      "debug_init_smoke_verified": T if debug else F,
      "size12m_init_verified": T if row["config_name"] == "dreamer_anchor" else F,
      "forward_verified": T if debug else F,
      "backward_verified": F,
      "optimizer_step_verified": F,
      "checkpoint_save_verified": T if debug else F,
      "checkpoint_reload_verified": F,
      "artifact_write_verified": T if debug else F,
      "analytical_addon_params": analytical if row.get("param_count_source") in (
          "analytical_estimate", "target_placeholder") else NA,
      "actual_addon_params": row.get("actual_addon_params", NA),
      "actual_total_params": row.get("total_params", NA),
  })
  return item


def _write_parity(output, data):
  csv_path = output / "component_matrix_v3.csv"
  json_path = output / "component_matrix_v3.json"
  with json_path.open() as file:
    json_rows = json.load(file)
  with csv_path.open() as file:
    csv_rows = list(csv.DictReader(file))
  json_names = [row["config_name"] for row in json_rows]
  csv_names = [row["config_name"] for row in csv_rows]
  json_schema = list(json_rows[0].keys()) if json_rows else []
  csv_schema = list(csv_rows[0].keys()) if csv_rows else []
  report = {
      "json_row_count": len(json_rows),
      "csv_row_count": len(csv_rows),
      "json_config_names": json_names,
      "csv_config_names": csv_names,
      "json_schema_columns": json_schema,
      "csv_schema_columns": csv_schema,
      "assertions": {
          "json_row_count_eq_15": len(json_rows) == 15,
          "csv_row_count_eq_15": len(csv_rows) == 15,
          "config_names_match": json_names == csv_names,
          "schema_columns_match": json_schema == csv_schema,
      },
  }
  report["parity_pass"] = all(report["assertions"].values())
  with (output / "component_matrix_v3_parity_report.json").open("w") as file:
    json.dump(report, file, indent=2)
  if not report["parity_pass"]:
    raise SystemExit("component_matrix_v3 parity failed")


def _write_parity_v4(output, data):
  csv_path = output / "component_matrix_v4.csv"
  json_path = output / "component_matrix_v4.json"
  json_rows = json.load(json_path.open())
  csv_rows = list(csv.DictReader(csv_path.open()))
  report = {
      "json_row_count": len(json_rows),
      "csv_row_count": len(csv_rows),
      "json_config_names": [row["config_name"] for row in json_rows],
      "csv_config_names": [row["config_name"] for row in csv_rows],
      "json_schema_columns": list(json_rows[0].keys()) if json_rows else [],
      "csv_schema_columns": list(csv_rows[0].keys()) if csv_rows else [],
  }
  report["assertions"] = {
      "json_row_count_eq_15": report["json_row_count"] == 15,
      "csv_row_count_eq_15": report["csv_row_count"] == 15,
      "config_names_match": report["json_config_names"] == report["csv_config_names"],
      "schema_columns_match": report["json_schema_columns"] == report["csv_schema_columns"],
  }
  report["parity_pass"] = all(report["assertions"].values())
  with (output / "component_matrix_v4_parity_report.json").open("w") as file:
    json.dump(report, file, indent=2)
  if not report["parity_pass"]:
    raise SystemExit("component_matrix_v4 parity failed")


if __name__ == "__main__":
  write("paper_artifacts")
