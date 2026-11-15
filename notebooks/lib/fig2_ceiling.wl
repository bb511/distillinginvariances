(* ::Package:: *)

(* fig2_ceiling.wl -- Figure 2, how much transfers: the invariance every distilled arm
   gains over the baseline of its regime, against the gain of the augmented ceiling on
   the same rows, on canonical rows (top) and on transformed rows (bottom). Loaded after
   paper_common.wl.
   Reads, through gains: tables/metrics_iii/transfer_test.csv (five cells) and
   tables/metrics_iv/transfer_test.csv (top tagging) at the points of
   hyperparameters/hyperparameters.csv and hyperparameters_top3.csv (the gain and
   fraction columns, tost_p, margin); and results.csv for the ceiling: inv_top1 of the
   baseline rows, test split, folds 1 to 5, no variant, under the cell's own probe. *)

baselineRows[] := Select[readLines["results.csv", ",baseline,"],
   #method == "baseline" && #["eval_split"] == "test" && IntegerQ[#fold] && #variant == "" &];
foldValues[c_, regime_, data_] := SortBy[Select[ofCell[baselineRows[], c], #regime == regime &&
       #["eval_data"] == data && #["probe_group"] == cells[c, "probe"] &], #fold &][[All, "inv_top1"]];
(* The augmented baseline minus the baseline of the regime, both on that regime's rows,
   paired by fold: the denominator of the transfer fraction on the test folds. *)
ceilingGap[c_, regime_] := foldInterval[foldValues[c, "augmented", regime] - foldValues[c, regime, regime]];
(* False where the ceiling's gap on the main run lies below the margin (transfer.py). *)
fractionDefined[c_, regime_] := First[gains[c, regime]]["defined"];

fig2YRange = <|"canonical" -> {-0.07, 0.84}, "transformed" -> {-0.075, 0.155}|>;
fig2Ticks = <|"canonical" -> {0.2, 1}, "transformed" -> {0.05, 2}|>;
(* The ceiling spans its cell, with its interval at the right end; faded where the
   transfer fraction is undefined. *)
ceilingLayer[c_, k_, regime_] := With[{g = ceilingGap[c, regime],
    o = If[fractionDefined[c, regime], 1, 0.35]},
   {Opacity[o], Black, AbsoluteThickness[0.8], Line[{{k - 0.95, g["mean"]}, {k - 0.05, g["mean"]}}],
    whisker[k - 0.09, {g["lo"], g["hi"]}, Black, o]}];
fig2Layers[regime_, dx_] := {
   MapIndexed[marginBand[#2[[1]], First[gains[#1, regime]]["margin"]] &, cellOrder], zeroGuide,
   MapIndexed[ceilingLayer[#1, #2[[1]], regime] &, cellOrder],
   MapIndexed[cellPoints[gains[#1, regime], #2[[1]], dx, cellColour[#1]] &, cellOrder]};
(* 5.2 pt between the arms of a cell, in units of the cell's width. *)
fig2Panel[regime_, labelled_, {w_, h_}, pad_] := panel[fig2Layers[regime, 5.2 Length[cellOrder]/w],
   {{0, Length[cellOrder]}, fig2YRange[regime]},
   {If[labelled, cellTicks[cellOrder, 1], {}], decTicks[fig2YRange[regime], Sequence @@ fig2Ticks[regime]]},
   regime <> " rows", {w, h}, pad];

fig2Legend[x_] := legendCentred[{methodEntries[methodOrder], Join[fillEntries, {fadedEntry, bandEntry}],
    {{lineGlyph[Directive[Black, AbsoluteThickness[0.8]]], "augmented ceiling"},
     {lineGlyph[Directive[Black, Opacity[0.35], AbsoluteThickness[0.8]]], "ceiling, fraction undefined"}}},
   x, {26, 16, 6}];

fig2[] := With[{w = 358, h = 104, left = 34, gap = 6, base = 50, pad = {{34, 2}, {14, 2}}},
   canvas[{
     place[fig2Panel["transformed", True, {w, h}, pad], {left, base}, pad],
     place[fig2Panel["canonical", False, {w, h}, pad], {left, base + h + gap}, pad],
     yLabel["invariance gain over the baseline", {6, base + h + gap/2}],
     fig2Legend[left + w/2]}, base + 2 h + gap + 4]];

(* Readings: every point, with its transfer fraction, and every ceiling. *)
fig2Reading[regime_][p_] := Join[<|"rows" -> regime|>, reading[p],
   <|"fraction" -> If[p["defined"], r3[p["fraction"]], "undefined"]|>];
fig2Readings[] := Dataset[Flatten[Table[fig2Reading[regime] /@ gains[c, regime],
     {regime, {"canonical", "transformed"}}, {c, cellOrder}]]];
fig2Ceilings[] := Dataset[Flatten[Table[With[{g = ceilingGap[c, regime]},
      <|"rows" -> regime, "cell" -> c, "baseline" -> r3[Mean[foldValues[c, regime, regime]]],
       "ceiling" -> r3[Mean[foldValues[c, "augmented", regime]]], "gap" -> r3[g["mean"]],
       "interval" -> r3[{g["lo"], g["hi"]}], "fraction defined" -> fractionDefined[c, regime]|>],
     {regime, {"canonical", "transformed"}}, {c, cellOrder}]]];
