# Visiomaster Complexity Report: YOLOv8 Two-stream OBB ASSADualBranchRIFusion Architecture

## Summary
- Style profile: `clean_white`
- Page: 18.00 x 10.00 in, aspect 1.80
- Visible semantic nodes: 22
- Edges: 25
- Regions: 5
- Region-covered visible nodes: 19/22
- Cross-region edges: 11
- Region plan entries: 0
- Validation warnings: 30
- Validation errors: 0

## Source Region Plan
- Not in exact reconstruction mode.

## Recommended Build Mode
- Whole-scene authoring is acceptable, but still run module audit before final Visio render.

## Region Load
- `rgb_lane`: 3 visible nodes, density=0.20/sqin, center=(6.95, 1.62) `ok`
- `neck_lane`: 6 visible nodes, density=0.21/sqin, center=(14.73, 3.52) `ok`
- `fusion_lane`: 4 visible nodes, density=0.25/sqin, center=(6.95, 3.50) `ok`
- `ir_lane`: 3 visible nodes, density=0.20/sqin, center=(6.95, 5.43) `ok`
- `notes_lane`: 3 visible nodes, density=0.08/sqin, center=(8.95, 8.02) `ok`
- Uncovered visible nodes: `title`, `input`, `split`

## Font Scale
- `operator_node`: 12.0-12.0 pt across 3 nodes
- `process_box`: 7.5-9.0 pt across 12 nodes
- `rounded_process`: 8.0-9.0 pt across 5 nodes
- `text_block`: 8.3-18.0 pt across 2 nodes

## Text Fit Risks
- `title` 7.32x0.29 in estimated vs 16.90x0.25 in
- `minus4_note` 3.29x0.95 in estimated vs 1.85x4.45 in

## Dense Region Risks
- No region exceeds the default density threshold.

## Paper Detail Grammar Risks
- `operator_node`: 3
- Long explicit path `minus4_stage_p3` length=2.87 in; check for missing bus/junction/boundary port.
- Long explicit path `minus4_stage_p4` length=2.87 in; check for missing bus/junction/boundary port.
- Long explicit path `minus4_stage_p5` length=2.87 in; check for missing bus/junction/boundary port.

## Validation Snapshot
- WARN: Text in node `title` may not fit (7.32x0.29 in estimated vs 16.90x0.25 in available). Wrap text, enlarge the node, or assign a smaller role font before rendering.
- WARN: Text in node `minus4_note` may not fit (3.29x0.95 in estimated vs 1.85x4.45 in available). Wrap text, enlarge the node, or assign a smaller role font before rendering.
- WARN: Edge `split_to_rgb` crosses container boundary (None -> rgb_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `split_to_rgb` directly connects components across module boundary (None -> rgb_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `split_to_ir` crosses container boundary (None -> ir_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `split_to_ir` directly connects components across module boundary (None -> ir_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `rgb_p3_to_fusion` crosses container boundary (rgb_lane -> fusion_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `rgb_p3_to_fusion` directly connects components across module boundary (rgb_lane -> fusion_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `ir_p3_to_fusion` crosses container boundary (ir_lane -> fusion_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `ir_p3_to_fusion` directly connects components across module boundary (ir_lane -> fusion_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `rgb_p4_to_fusion` crosses container boundary (rgb_lane -> fusion_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `rgb_p4_to_fusion` directly connects components across module boundary (rgb_lane -> fusion_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `ir_p4_to_fusion` crosses container boundary (ir_lane -> fusion_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `ir_p4_to_fusion` directly connects components across module boundary (ir_lane -> fusion_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `rgb_p5_to_fusion` crosses container boundary (rgb_lane -> fusion_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `rgb_p5_to_fusion` directly connects components across module boundary (rgb_lane -> fusion_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `ir_p5_to_fusion` crosses container boundary (ir_lane -> fusion_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `ir_p5_to_fusion` directly connects components across module boundary (ir_lane -> fusion_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `fusion_p3_to_add` crosses container boundary (fusion_lane -> neck_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `fusion_p3_to_add` directly connects components across module boundary (fusion_lane -> neck_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `fusion_p3_to_add` intersects non-endpoint node `rgb_p5`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- WARN: Edge `fusion_p4_to_add` crosses container boundary (fusion_lane -> neck_lane). Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, or mark `allow_cross_container: true` for deliberate callouts.
- WARN: Edge `fusion_p4_to_add` directly connects components across module boundary (fusion_lane -> neck_lane). For exact replicas, route through `boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component.
- WARN: Edge `fusion_p4_to_add` intersects non-endpoint node `fusion_p5`. Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node.
- Additional validation items suppressed; run `scene_validate.py` for the full list.

