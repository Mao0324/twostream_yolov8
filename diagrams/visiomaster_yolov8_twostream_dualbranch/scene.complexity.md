# Visiomaster Complexity Report: YOLOv8 Two-stream OBB ASSA Dual-branch Fusion Architecture

## Summary
- Style profile: `clean_white`
- Page: 18.00 x 10.00 in, aspect 1.80
- Visible semantic nodes: 29
- Edges: 36
- Regions: 3
- Region-covered visible nodes: 28/29
- Cross-region edges: 4
- Region plan entries: 0
- Validation warnings: 30
- Validation errors: 0

## Source Region Plan
- Not in exact reconstruction mode.

## Recommended Build Mode
- Use `region_first` or `tiled_subscenes`: rebuild each logical module/crop, validate it, then assemble the full-page scene.
- Add invisible `audit_region` boxes for source areas that do not have visible dashed frames.
- Freeze shared style tokens before assembly: body font, small label font, operator font, frame title font, and arrow weight.

## Region Load
- `backbone_region`: 16 visible nodes, density=0.23/sqin, center=(5.90, 3.84) `ok`
- `head_region`: 9 visible nodes, density=0.24/sqin, center=(14.67, 3.84) `ok`
- `legend_region`: 3 visible nodes, density=0.07/sqin, center=(9.00, 8.38) `ok`
- Uncovered visible nodes: `title`

## Font Scale
- `operator_node`: 7.0-14.0 pt across 4 nodes
- `process_box`: 8.0-9.0 pt across 19 nodes
- `rounded_process`: 8.5-10.0 pt across 5 nodes
- `text_block`: 18.0-18.0 pt across 1 nodes

## Text Fit Risks
- `title` 7.57x0.29 in estimated vs 16.90x0.25 in

## Dense Region Risks
- No region exceeds the default density threshold.

## Paper Detail Grammar Risks
- `operator_node`: 4
- Long explicit path `e_note_minus3` length=3.45 in; check for missing bus/junction/boundary port.
- Long explicit path `e_note_fusion` length=3.45 in; check for missing bus/junction/boundary port.

## Validation Snapshot
- WARN: Operator node `split` has a multi-character symbol; set `symbol_text_fit: "single_line"` and tune symbol_box_* if the source shows a compact operator.
- WARN: Complex scene has 29 visible nodes and 36 edges; set metadata.region_strategy to `region_first`, `tiled_subscenes`, or `module_first`, then build/review the figure module-by-module before whole-page assembly.
- WARN: Region `backbone_region` has no source_bbox_px/source_aspect_ratio; visual review cannot distinguish source scale drift from renderer issues.
- WARN: Region `head_region` has no source_bbox_px/source_aspect_ratio; visual review cannot distinguish source scale drift from renderer issues.
- WARN: Region `legend_region` has no source_bbox_px/source_aspect_ratio; visual review cannot distinguish source scale drift from renderer issues.
- WARN: Font sizes for `operator_node` vary from 7.0pt to 14.0pt (small: split; large: add_p3, add_p4, add_p5). Large figures should keep each component family on a small role-based font scale.
- WARN: Text in node `title` may not fit (7.57x0.29 in estimated vs 16.90x0.25 in available). Wrap text, enlarge the node, or assign a smaller role font before rendering.
- WARN: Edge `e_rgb_p3_down` looks like a dashed/loss/backprop feedback route but uses `arrow_connector`. Use `dashed_feedback_path` so the path is audited as one continuous feedback route.
- WARN: Edge `e_ir_p3_down` looks like a dashed/loss/backprop feedback route but uses `arrow_connector`. Use `dashed_feedback_path` so the path is audited as one continuous feedback route.
- WARN: Edge `e_fusion_p4_add` intersects non-endpoint node `add_p3`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- WARN: Edge `e_fusion_p4_rgb_down` looks like a dashed/loss/backprop feedback route but uses `arrow_connector`. Use `dashed_feedback_path` so the path is audited as one continuous feedback route.
- WARN: Edge `e_fusion_p4_ir_down` looks like a dashed/loss/backprop feedback route but uses `arrow_connector`. Use `dashed_feedback_path` so the path is audited as one continuous feedback route.
- WARN: Edge `e_fusion_p4_ir_down` intersects non-endpoint node `add_p3`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- WARN: Edge `e_rgb_p5_down` crosses container boundary (backbone_region -> head_region). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `e_rgb_p5_down` directly connects components across module boundary (backbone_region -> head_region). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `e_ir_p5_down` crosses container boundary (backbone_region -> head_region). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `e_ir_p5_down` directly connects components across module boundary (backbone_region -> head_region). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `e_ir_p5_fusion` intersects non-endpoint node `add_p5`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- WARN: Edge `e_addp4_fpn` crosses container boundary (backbone_region -> head_region). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `e_addp4_fpn` directly connects components across module boundary (backbone_region -> head_region). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `e_addp4_fpn` intersects non-endpoint node `rgb_p5`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- WARN: Edge `e_addp3_fpn` crosses container boundary (backbone_region -> head_region). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `e_addp3_fpn` directly connects components across module boundary (backbone_region -> head_region). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `e_addp3_fpn` intersects non-endpoint node `fusion_p5`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- Additional validation items suppressed; run `scene_validate.py` for the full list.

