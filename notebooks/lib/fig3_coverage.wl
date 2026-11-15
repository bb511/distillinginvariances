(* ::Package:: *)

(* fig3_coverage.wl -- Figure 3, where the orbit lies: against the coverage AUC (0.5: g.x
   stays on the data's support, 1: every g.x is recognisable), left the canonical gain
   over the baseline of the seven arms of each cell, right the top-1 non-invariance of
   the distilled students in each magnitude bin. Loaded after paper_common.wl.
   Reads tables/metrics_iii/folds_2026-11-11/coverage_folds.csv (auc per fold, whole
   draw, canonical rows, the cell's own probe), tables/metrics_iii/transfer_test.csv at
   the points of hyperparameters/hyperparameters.csv (gain_mean, canonical regime), and
   tables/orchestration_v/coverage/coverage_bins.csv (the auc and delta_s_distilled
   columns of the bins with interpreted True). *)

(* Top tagging's coverage was not measured. *)
fig3Cells = Most[cellOrder];
coverageFolds[c_] := Select[
    ofCell[readTable["tables/metrics_iii/folds_2026-11-11/coverage_folds.csv"], c],
    #magnitude == "" && #["eval_data"] == "canonical" && #["probe_group"] == cells[c, "probe"] &][[All, "auc"]];
coverage[c_] := foldInterval[coverageFolds[c]];
(* Bins under 500 test samples are not read (interpreted False). *)
binRows[c_] := Select[readTable["tables/orchestration_v/coverage/coverage_bins.csv"],
   #teacher == cells[c, "teacher"] && #interpreted == "True" &];
binPaths[c_] := SortBy[#["bin_low"] &] /@ Values[GroupBy[binRows[c], #magnitude &]];

fig3XRange = {0.47, 1.03};
fig3XTicks = decTicks[{0.5, 1}, 0.1, 1];
(* Left: a dot per arm at the cell's AUC. *)
fig3Dot[c_] := marker["circle", cellColour[c], True, 3.5, 0.75];
fig3CellLayer[c_] := Inset[fig3Dot[c], {coverage[c]["mean"], #["mean"]}] & /@ gains[c, "canonical"];
fig3CellPanel[size_, pad_] := panel[{zeroGuide, fig3CellLayer /@ fig3Cells},
   {fig3XRange, {-0.03, 0.18}}, {fig3XTicks, decTicks[{-0.03, 0.18}, 0.05, 2]}, "cells", size, pad];

(* Right: per bin, the range over the seven arms (vertical) and the AUC's interval
   (horizontal), the bins of one magnitude joined in their order. *)
binCross[b_] := With[{y = Mean[{b["delta_s_distilled_min"], b["delta_s_distilled_max"]}]},
   {Line[{{b["auc_mean"], b["delta_s_distilled_min"]}, {b["auc_mean"], b["delta_s_distilled_max"]}}],
    Line[{{b["auc_ci_low"], y}, {b["auc_ci_high"], y}}]}];
binJoin[path_] := Line[{#["auc_mean"], Mean[{#["delta_s_distilled_min"], #["delta_s_distilled_max"]}]} & /@ path];
fig3BinLayer[c_] := {cellColour[c], {AbsoluteThickness[0.3], Opacity[0.5], binJoin /@ binPaths[c]},
   {AbsoluteThickness[0.7], binCross /@ binRows[c]}};
fig3BinPanel[size_, pad_] := panel[fig3BinLayer /@ fig3Cells, {fig3XRange, {0, 1.04}},
   {fig3XTicks, decTicks[{0, 1}, 0.2, 1]}, "magnitude bins, reported only", size, pad];

fig3Legend[x_] := legendCentred[{cellEntries[fig3Cells]}, x, {6}];
fig3[] := With[{w = 160, h = 120, left = 34, gap = 38, base = 46, pads = {{{34, 2}, {16, 2}}, {{24, 2}, {16, 2}}}},
   canvas[{
     place[fig3CellPanel[{w, h}, pads[[1]]], {left, base}, pads[[1]]],
     place[fig3BinPanel[{w, h}, pads[[2]]], {left + w + gap, base}, pads[[2]]],
     yLabel["invariance gain over the baseline", {6, base + h/2}],
     yLabel["non\:2010invariance of the students", {left + w + 9, base + h/2}],
     xLabel["coverage AUC (1: the orbit leaves the data's support)", {left + w + gap/2, base - 24}],
     fig3Legend[left + w + gap/2]}, base + h + 4]];

(* Readings: the cells with the AUC to five decimals, since two of them lie within
   0.002 of 1, and every bin drawn. *)
fig3CellReading[c_] := With[{a = coverage[c], g = gains[c, "canonical"]},
   <|"cell" -> c, "AUC" -> Round[a["mean"], 0.00001], "interval" -> Round[{a["lo"], a["hi"]}, 0.00001],
    "gain, 7 arms" -> r3[MinMax[Lookup[g, "mean"]]],
    "fraction, 7 arms" -> If[First[g]["defined"], r3[MinMax[Lookup[g, "fraction"]]], "undefined"]|>];
fig3CellReadings[] := Dataset[fig3CellReading /@ fig3Cells];
fig3BinReading[c_][b_] := <|"cell" -> c, "magnitude" -> b["magnitude"], "bin" -> {b["bin_low"], b["bin_high"]},
   "AUC" -> b["auc_mean"], "interval" -> {b["auc_ci_low"], b["auc_ci_high"]},
   "non-invariance, 7 arms" -> {b["delta_s_distilled_min"], b["delta_s_distilled_max"]}|>;
fig3BinReadings[] := Dataset[Flatten[Table[fig3BinReading[c] /@ Flatten[binPaths[c]], {c, fig3Cells}]]];
