(* ::Package:: *)

(* fig1_attribution.wl -- Figure 1, the money plot: on canonical rows, the invariance each
   KD and Hint student gains over the student of the cell's non-invariant control at the
   same T and alpha, the cells split by whether the group is a symmetry of the task.
   Loaded after paper_common.wl.
   Reads, through controlGains: tables/orchestration_vi/gaps/out/item1_control_gains.csv
   (five cells; the JetFormer control in LLoCa-z) and tables/metrics_iv/control_{kd,hint}.csv
   (top tagging), metric inv_top1, test split, canonical regime, at the points of
   hyperparameters/hyperparameters.csv and hyperparameters_top3.csv. *)

fig1Classes = {{"canonical rows: group is a symmetry of the task", Select[cellOrder, cells[#, "symmetry"] &]},
   {"canonical rows: group is not a symmetry of the task", Select[cellOrder, ! cells[#, "symmetry"] &]}};
fig1YRange = {-0.09, 0.21};

fig1Bands[cs_] := MapIndexed[
   marginBand[#2[[1]], First[controlGains[#1, "canonical"]]["margin"]] &, cs];
fig1Points[cs_, dx_] := MapIndexed[
   cellPoints[controlGains[#1, "canonical"], #2[[1]], dx, cellColour[#1]] &, cs];
(* 9 pt between the arms of a cell, in units of the cell's width. *)
fig1Panel[{label_, cs_}, k_, {w_, h_}, pad_] := panel[
   {fig1Bands[cs], zeroGuide, fig1Points[cs, 9 Length[cs]/w]},
   {{0, Length[cs]}, fig1YRange}, {cellTicks[cs, 2], decTicks[fig1YRange, 0.05, 2, k == 1]},
   label, {w, h}, pad];

fig1Arms = {{"kd", 0, alphaZero}, {"kd", 1, alphaPositive}, {"hint", 0, alphaZero},
   {"hint", 1, alphaPositive}};
fig1Legend[x_] := legendCentred[{
    {marker[methodShape[#[[1]]], Black, #[[2]] == 0], Row[{methodLabel[#[[1]]], ", ", #[[3]]}]} & /@ fig1Arms,
    {fadedEntry, bandEntry}}, x, {16, 6}];

fig1[] := With[{w = 177, h = 150, left = 32, gap = 8, base = 44,
    pads = {{{32, 2}, {24, 2}}, {{2, 2}, {24, 2}}}},
   canvas[{
     MapIndexed[place[fig1Panel[#1, #2[[1]], {w, h}, pads[[#2[[1]]]]],
        {left + (#2[[1]] - 1) (w + gap), base}, pads[[#2[[1]]]]] &, fig1Classes],
     yLabel["invariance gain over the control's students", {6, base + h/2}],
     fig1Legend[left + w + gap/2]}, base + h + 4]];

(* Readings: every point of the figure. *)
fig1Readings[] := Dataset[reading /@ Flatten[controlGains[#, "canonical"] & /@ cellOrder]];
