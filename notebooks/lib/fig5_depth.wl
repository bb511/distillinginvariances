(* ::Package:: *)

(* fig5_depth.wl -- Figure 5, where invariance arises inside the student: the
   orbit-invariance fraction of each layer of the canonical KD and Hint students at
   alpha = 0 and of the student at initialisation, under SO(2) about the beam and under
   SO(3), in the three cells that read [pT, eta, phi]. Loaded after paper_common.wl.
   Reads tables/metrics_iii/profile_probes_selected/profile_folds.csv (mean, ci_low,
   ci_high per layer; regime and eval_data canonical, probe_group so2 or so3, method kd
   or hint at alpha 0 with init trained, and the init random rows). *)

fig5Cells = {"lloca_z", "lloca_so3", "lorentznet"};
fig5Probes = {"so2", "so3"};
fig5Layers = {"hidden_1", "hidden_2", "hidden_3", "logits"};
fig5Models = {"kd", "hint", "init"};
profileRows[c_] := Select[
   ofCell[readTable["tables/metrics_iii/profile_probes_selected/profile_folds.csv"], c],
   #regime == "canonical" && #["eval_data"] == "canonical" && MemberQ[fig5Probes, #["probe_group"]] &];
(* The initialisation is stored under the KD arm at alpha 0, one model per regime. *)
fig5Model[r_] := Which[r["init"] == "random", "init", r["alpha"] == 0, r["method"], True, None];
profile[c_, probe_, model_] := SortBy[
   Select[profileRows[c], #["probe_group"] == probe && fig5Model[#] == model &],
   First[FirstPosition[fig5Layers, #layer]] &];

(* KD and Hint sit 0.07 of a layer either side of it, so that their intervals stay apart. *)
fig5Dodge = <|"kd" -> -0.07, "hint" -> 0.07, "init" -> 0|>;
fig5XY[rows_, model_] := MapIndexed[{#2[[1]] + fig5Dodge[model], #1["mean"]} &, rows];
fig5Line[rows_, "init", colour_] := {colour, Opacity[0.6], AbsoluteThickness[0.6], AbsoluteDashing[{2, 2}],
   Line[fig5XY[rows, "init"]]};
fig5Line[rows_, model_, colour_] := {{colour, AbsoluteThickness[0.6], Line[fig5XY[rows, model]]},
   MapThread[{whisker[#1[[1]], {#2["ci_low"], #2["ci_high"]}, colour],
      Inset[marker[methodShape[model], colour, True], #1]} &, {fig5XY[rows, model], rows}]};
fig5Series[c_] := Table[fig5Line[profile[c, probe, model], model, groupColour[probe]],
   {model, Reverse[fig5Models]}, {probe, fig5Probes}];

fig5XTicks = MapThread[{#1, text[#2, 8], {0.025, 0}} &, {Range[4], {"1", "2", "3", "logits"}}];
(* The vertical guide marks the second hidden layer, which the hint guides. *)
fig5Panel[c_, k_, size_, pad_] := panel[{{guideStyle, InfiniteLine[{{2, 0}, {2, 1}}]}, fig5Series[c]},
   {{0.5, 4.5}, {0, 1.04}}, {fig5XTicks, decTicks[{0, 1}, 0.2, 1, k == 1]}, cells[c, "label"], size, pad];

fig5Legend[x_] := legendCentred[{{
     {marker["circle", groupColour["so2"], True], "SO(2) about the beam"},
     {marker["circle", groupColour["so3"], True], "SO(3)"},
     {marker["circle", Black, True], "KD"}, {marker["square", Black, True], "Hint"},
     {lineGlyph[Directive[Black, Opacity[0.6], AbsoluteThickness[0.6], AbsoluteDashing[{2, 2}]]],
      "initialisation"}, {Graphics[{guideStyle, Line[{{0, -3}, {0, 3}}]}, PlotRange -> {{-4, 4}, {-3, 3}}, ImageSize -> {8, 6}],
      "guided layer"}}}, x, {6}];
fig5[] := With[{w = 116, h = 110, left = 30, gap = 8, base = 46, lefts = {30, 2, 2}},
   canvas[{
     MapIndexed[With[{pad = {{lefts[[#2[[1]]]], 2}, {16, 2}}},
        place[fig5Panel[#1, #2[[1]], {w, h}, pad], {left + (#2[[1]] - 1) (w + gap), base}, pad]] &, fig5Cells],
     yLabel["orbit\:2010invariance fraction", {6, base + h/2}],
     xLabel["hidden layer (canonical rows)", {left + 1.5 w + gap, base - 24}],
     fig5Legend[left + 1.5 w + gap]}, base + h + 4]];

(* Readings: every mean and interval drawn, and SO(2) minus SO(3) of the fold means at
   the last hidden layer and the logits (H2c; its intervals are in h23_selected.txt). *)
fig5Reading[c_, probe_, model_][r_] := <|"cell" -> c, "probe" -> probe, "model" -> model,
   "layer" -> r["layer"], "fraction" -> r3[r["mean"]],
   "interval" -> If[model == "init", "single model", r3[{r["ci_low"], r["ci_high"]}]]|>;
fig5Readings[] := Dataset[Flatten[Table[fig5Reading[c, probe, model] /@ profile[c, probe, model],
     {c, fig5Cells}, {model, fig5Models}, {probe, fig5Probes}]]];
fig5Gap[c_, model_] := With[{d = Lookup[profile[c, "so2", model], "mean"] - Lookup[profile[c, "so3", model], "mean"]},
   <|"cell" -> c, "model" -> model, "last hidden layer" -> r3[d[[3]]], "logits" -> r3[d[[4]]]|>];
fig5Gaps[] := Dataset[Flatten[Table[fig5Gap[c, model], {c, fig5Cells}, {model, {"kd", "hint"}}]]];
