(* ::Package:: *)

(* fig4_transformed.wl -- Figure 4, the non-invariant teacher's harm: on transformed rows,
   the invariance each KD and Hint student gains over the baseline, in colour for the
   student of the cell's teacher and in black for the student of the non-invariant
   control at the same T and alpha. Loaded after paper_common.wl.
   Reads, at the points of hyperparameters/hyperparameters.csv and
   hyperparameters_top3.csv: the gain columns of tables/metrics_iii/transfer_test.csv
   and tables/metrics_iv/transfer_test.csv (the cell's students), and
   tables/orchestration_vi/gaps/out/control_gain_over_baseline.csv (the control's
   students: gain_inv_top1_mean, _lo, _hi, tost_p, margin; the JetFormer control in
   LLoCa-z). The readings add the paired gain over the control's students of Figure 1's
   tables, transformed regime. *)

fig4Methods = {"kd", "hint"};
ownGains[c_, regime_] := Select[gains[c, regime], MemberQ[fig4Methods, #method] &];
controlBaselineGains[c_, regime_] := sortArms[
   point[c, #, {"gain_inv_top1_mean", "gain_inv_top1_lo", "gain_inv_top1_hi"}] & /@ atSelected[
     Select[readTable["tables/orchestration_vi/gaps/out/control_gain_over_baseline.csv"],
      #control == cells[c, "control"] &], c, regime]];

fig4YRange = {-0.26, 0.27};
(* An arm's two students sit 2.6 pt either side of its place, the arms 13 pt apart; both
   in units of the cell's width. *)
fig4Pair[c_, k_, i_, {own_, control_}, unit_] := With[{x = armX[k, i, 4, 13 unit]},
   {pointLayer[own, x - 2.6 unit, cellColour[c]], pointLayer[control, x + 2.6 unit, Black]}];
fig4Cell[c_, k_, unit_] := MapIndexed[fig4Pair[c, k, #2[[1]], #1, unit] &,
   Transpose[{ownGains[c, "transformed"], controlBaselineGains[c, "transformed"]}]];
fig4Panel[{w_, h_}, pad_] := panel[
   {MapIndexed[marginBand[#2[[1]], First[ownGains[#1, "transformed"]]["margin"]] &, cellOrder], zeroGuide,
    MapIndexed[fig4Cell[#1, #2[[1]], Length[cellOrder]/w] &, cellOrder]},
   {{0, Length[cellOrder]}, fig4YRange}, {cellTicks[cellOrder, 1], decTicks[fig4YRange, 0.1, 1]},
   "transformed rows", {w, h}, pad];

fig4Arms = {{"kd", 0, alphaZero}, {"kd", 1, alphaPositive}, {"hint", 0, alphaZero},
   {"hint", 1, alphaPositive}};
fig4Legend[x_] := legendCentred[{
    {marker[methodShape[#[[1]]], Black, #[[2]] == 0], Row[{methodLabel[#[[1]]], ", ", #[[3]]}]} & /@ fig4Arms,
    {{swatch[cellColour /@ {"deepsets", "lloca_so3", "lorentznet"}], "student of the cell's teacher"},
     {swatch[{Black, Black}], "student of the control"}, fadedEntry, bandEntry}}, x, {16, 6}];

fig4[] := With[{w = 358, h = 130, left = 34, base = 40, pad = {{34, 2}, {14, 2}}},
   canvas[{place[fig4Panel[{w, h}, pad], {left, base}, pad],
     yLabel["invariance gain over the baseline", {6, base + h/2}],
     fig4Legend[left + w/2]}, base + h + 4]];

(* Readings: both students of every arm and, from the paired test, the cell's student
   minus the control's. *)
fig4Reading[who_][p_] := Join[<|"student of" -> who|>, reading[p]];
fig4Readings[] := Dataset[Flatten[Table[{fig4Reading["cell's teacher"] /@ ownGains[c, "transformed"],
      fig4Reading["control"] /@ controlBaselineGains[c, "transformed"],
      fig4Reading["cell's teacher minus control"] /@ controlGains[c, "transformed"]}, {c, cellOrder}]]];
