(* ::Package:: *)

(* distill_figures.wl -- panel builders and figure definitions for the
   distilling-invariances paper. Requires distill_style.wl and the
   globals dataDir and plotsDir. Loaded with Get[]; symbols in Global`. *)

(* --- data ------------------------------------------------------------- *)

(* CSV to a list of row associations. Lines starting with '#' are comments;
   empty cells become Missing[]. *)
loadRows[path_] := Module[{rows, hdr},
  rows = Select[Import[path, "CSV"],
    !(StringQ[#[[1]]] && StringStartsQ[StringTrim[#[[1]]], "#"]) &];
  hdr = First[rows];
  (* Import drops trailing empty cells, so rows are padded to the header
     length. The lorentz_temperature_*.csv curve rows carry one spurious empty
     field after the temperature (an export bug), so over-long rows lose their
     first empty cell until they fit. *)
  Table[AssociationThread[hdr -> (fitRow[r, Length[hdr]] /. "" -> Missing["Empty"])],
   {r, Rest[rows]}]];

fitRow[r_, n_] := Module[{row = r, pos},
  While[Length[row] > n && (pos = FirstPosition[row, "", Missing[], {1}]) =!= Missing[],
   row = Delete[row, pos]];
  PadRight[Take[row, UpTo[n]], n, ""]];

dataFile[name_] := FileNameJoin[{dataDir, name}];

rowWhere[rows_, test_] := With[{sel = Select[rows, test]},
  If[sel === {}, Missing["NoRow"], First[sel]]];

(* mean/std pair for a metric column; std is None if the column is absent *)
ms[row_, col_] := {row[col <> "_mean"],
   If[KeyExistsQ[row, col <> "_std"], row[col <> "_std"], None]};

(* --- axes -------------------------------------------------------------- *)

catTicks[labels_] := Table[{i, Style[labels[[i]], heros[22]], {0.014, 0}}, {i, Length[labels]}];
log2Ticks[] := Table[{Log2[t], Style[ToString[t], heros[22]], {0.014, 0}}, {t, {1, 2, 4, 8, 16}}];
tempLabel = Row[{"Temperature ", Style["T", Italic]}];

tr[logQ_][y_] := If[logQ, Log10[y], y];
mapTicks[ticks_, f_] := {f[#[[1]]], #[[2]], #[[3]]} & /@ ticks;

(* range of all plotted values (means and band edges), padded by 8 % *)
yRangeOf[vals_, logQ_, pad_: 0.08] := Module[{v, lo, hi, d},
  v = DeleteMissing[Flatten[vals]];
  If[logQ, v = Log10[Select[v, # > 0 &]]];
  {lo, hi} = MinMax[v]; d = hi - lo;
  If[d == 0, d = Max[Abs[lo] 0.1, 0.01]];
  {lo - pad d, hi + pad d}];

bandEdges[s_] := If[s["s"] === None, {s["y"]},
   {s["y"], s["y"] + s["s"], s["y"] - s["s"]}];

(* --- primitives -------------------------------------------------------- *)

(* series: <|"x", "y", "s" (list or None), "colour"|> *)
seriesPrimitives[s_, logQ_] := Module[{x = s["x"], y = s["y"], sd = s["s"], pts, lo, hi, band},
  pts = Transpose[{x, tr[logQ] /@ y}];
  band = If[sd === None, {},
    lo = If[logQ, Max[#, 10.^-6] & /@ (y - sd), y - sd]; hi = y + sd;
    {s["colour"], Opacity[0.18], EdgeForm[None],
     Polygon[Join[Transpose[{x, tr[logQ] /@ hi}], Reverse[Transpose[{x, tr[logQ] /@ lo}]]]]}];
  {band, {s["colour"], AbsoluteThickness[2.8], Line[pts], PointSize[0.014], Point[pts]}}];

(* refs: <|"y", "s" (number, None or Missing), "colour", "label" (optional)|> *)
refPrimitives[r_, {x0_, x1_}, logQ_] := Module[{y = tr[logQ][r["y"]], sd = r["s"], band, note},
  band = If[sd === None || MissingQ[sd] || sd == 0, {},
    {r["colour"], Opacity[0.12], EdgeForm[None],
     Rectangle[{x0, tr[logQ][If[logQ, Max[r["y"] - sd, 10.^-6], r["y"] - sd]]},
      {x1, tr[logQ][r["y"] + sd]}]}];
  note = If[KeyExistsQ[r, "label"] && r["label"] =!= None,
    Text[Style[r["label"], heros[19], r["colour"]], {x1 - 0.02 (x1 - x0), y}, {1, -1.1}], {}];
  {band, {r["colour"], AbsoluteThickness[2.2], AbsoluteDashing[{9, 5}], Line[{{x0, y}, {x1, y}}]}, note}];

cornerNote[text_, colour_: Black, pos_: {0.97, 0.95}, align_: {Right, Top}] :=
  Inset[Style[text, heros[19], colour], Scaled[pos], align];

(* --- panels ------------------------------------------------------------ *)

Options[seriesPanel] = {"XLabel" -> None, "YLabel" -> None, "XTicks" -> Automatic,
   "XRange" -> Automatic, "LogY" -> False, "YRange" -> Automatic, "Extra" -> {},
   "Size" -> Automatic, "Padding" -> Automatic, "YLeftColour" -> Black};

seriesPanel[series_List, refs_List, OptionsPattern[]] := Module[{logQ, xr, yr},
  logQ = OptionValue["LogY"];
  xr = Replace[OptionValue["XRange"],
    Automatic :> With[{xs = Flatten[#["x"] & /@ series]}, {Min[xs] - 0.5, Max[xs] + 0.5}]];
  yr = Replace[OptionValue["YRange"], Automatic :> yRangeOf[
      {bandEdges /@ series, Table[{r["y"], If[r["s"] === None || MissingQ[r["s"]], {},
          {r["y"] + r["s"], r["y"] - r["s"]}]}, {r, refs}]}, logQ]];
  framePanel[{Table[refPrimitives[r, xr, logQ], {r, refs}],
    Table[seriesPrimitives[s, logQ], {s, series}]}, xr, yr,
   "XLabel" -> OptionValue["XLabel"], "YLabel" -> OptionValue["YLabel"],
   "XTicks" -> OptionValue["XTicks"],
   "YTicks" -> If[logQ, logTicks[yr[[1]], yr[[2]]], Automatic],
   "Extra" -> OptionValue["Extra"], "Size" -> OptionValue["Size"],
   "Padding" -> OptionValue["Padding"], "YLeftColour" -> OptionValue["YLeftColour"]]];

(* Two metrics in one frame: the right-hand series are mapped affinely onto
   the left range and the right ticks are mapped the same way, so the two
   axes align by construction. *)
twinPadding = {{118, 118}, {84, 16}};
Options[twinPanel] = {"XLabel" -> None, "YLabel" -> None, "YRightLabel" -> None,
   "XTicks" -> Automatic, "XRange" -> Automatic, "LogY" -> False, "Extra" -> {},
   "Size" -> Automatic, "Padding" -> twinPadding,
   "LeftColour" -> tolVibrant["blue"], "RightColour" -> tolVibrant["orange"]};

twinPanel[left_Association, right_Association, OptionsPattern[]] := Module[
  {logQ, xr, yl, yrr, f, k, mapSeries, mapRef, lp, rp, ytl, ytr},
  logQ = OptionValue["LogY"];
  xr = Replace[OptionValue["XRange"],
    Automatic :> With[{xs = Flatten[#["x"] & /@ left["series"]]}, {Min[xs] - 0.5, Max[xs] + 0.5}]];
  yl = yRangeOf[{bandEdges /@ left["series"], refEdges /@ left["refs"]}, logQ];
  yrr = yRangeOf[{bandEdges /@ right["series"], refEdges /@ right["refs"]}, False];
  f = Rescale[#, yrr, yl] &;
  k = (yl[[2]] - yl[[1]])/(yrr[[2]] - yrr[[1]]);
  mapSeries[s_] := Append[s, {"y" -> f /@ s["y"], "s" -> If[s["s"] === None, None, k s["s"]]}];
  mapRef[r_] := Append[r, {"y" -> f[r["y"]],
     "s" -> If[r["s"] === None || MissingQ[r["s"]], r["s"], k r["s"]]}];
  lp = {Table[refPrimitives[r, xr, logQ], {r, left["refs"]}],
    Table[seriesPrimitives[s, logQ], {s, left["series"]}]};
  rp = {Table[refPrimitives[mapRef[r], xr, False], {r, right["refs"]}],
    Table[seriesPrimitives[mapSeries[s], False], {s, right["series"]}]};
  ytl = If[logQ, logTicks[yl[[1]], yl[[2]]], linTicks[yl[[1]], yl[[2]]]];
  ytr = mapTicks[linTicks[yrr[[1]], yrr[[2]]], f];
  framePanel[{lp, rp}, xr, yl,
   "XLabel" -> OptionValue["XLabel"], "YLabel" -> OptionValue["YLabel"],
   "YRightLabel" -> OptionValue["YRightLabel"], "XTicks" -> OptionValue["XTicks"],
   "YTicks" -> ytl, "YRightTicks" -> ytr,
   "YLeftColour" -> OptionValue["LeftColour"], "YRightColour" -> OptionValue["RightColour"],
   "Extra" -> OptionValue["Extra"], "Size" -> OptionValue["Size"],
   "Padding" -> OptionValue["Padding"]]];

refEdges[r_] := If[r["s"] === None || MissingQ[r["s"]], {r["y"]},
   {r["y"], r["y"] + r["s"], r["y"] - r["s"]}];

(* teacher/student compute on a log axis; values transcribed, see data/model_compute.csv *)
sciLabel[v_] := With[{e = Floor[Log10[v]]},
  Row[{NumberForm[N[v/10.^e], {3, 2}], "\[Times]", Superscript[10, e]}]];

computePanel[t_, s_, ylabel_] := Module[{lo, hi, bars, labels},
  lo = Floor[Log10[s]] - 0.4; hi = Ceiling[Log10[t]] + 0.7;
  bars = {{tolVibrant["grey"], EdgeForm[{Black, AbsoluteThickness[1.5]}],
     Rectangle[{0.65, lo}, {1.35, Log10[t]}]},
    {tolVibrant["blue"], EdgeForm[{Black, AbsoluteThickness[1.5]}],
     Rectangle[{1.65, lo}, {2.35, Log10[s]}]}};
  labels = {Text[Style[sciLabel[t], heros[20]], {1, Log10[t]}, {0, -1}],
    Text[Style[sciLabel[s], heros[20]], {2, Log10[s]}, {0, -1}]};
  framePanel[{bars, labels}, {0.4, 2.6}, {lo, hi},
   "XTicks" -> catTicks[{"Teacher", "Student"}], "YTicks" -> logTicks[lo, hi],
   "YLabel" -> ylabel, "Padding" -> {{125, 22}, {84, 16}}]];

(* dot chart: one row per configuration, value printed next to the marker *)
Options[dotRowPanel] = {"XLabel" -> None, "LogX" -> False, "XRange" -> Automatic,
   "ShowLabels" -> True, "Size" -> {760, 620}, "Padding" -> Automatic,
   "Format" -> (ToString[NumberForm[#, {4, 3}]] &)};
dotRowPanel[labels_, values_, colour_, OptionsPattern[]] := Module[
  {n, ys, logQ, xs, xr, guides, marks, texts, yt},
  n = Length[labels]; ys = Reverse[Range[n]]; logQ = OptionValue["LogX"];
  xs = tr[logQ] /@ values;
  xr = Replace[OptionValue["XRange"], Automatic :> With[{lo = Min[xs], hi = Max[xs]},
      {lo - 0.12 (hi - lo), hi + 0.45 (hi - lo)}]];
  guides = {GrayLevel[0.45], AbsoluteThickness[0.7], AbsoluteDashing[{3, 3}],
    Table[Line[{{xr[[1]], y}, {xr[[2]], y}}], {y, ys}]};
  marks = {colour, AbsolutePointSize[11], Point[Transpose[{xs, ys}]]};
  texts = Table[Text[Style[OptionValue["Format"][values[[i]]], heros[20]],
     {xs[[i]], ys[[i]]}, {-1.4, 0}], {i, n}];
  yt = Table[{ys[[i]], If[OptionValue["ShowLabels"], Style[labels[[i]], heros[22]], ""],
     {0.008, 0}}, {i, n}];
  framePanel[{guides, marks, texts}, xr, {0.4, n + 0.6},
   "XLabel" -> OptionValue["XLabel"],
   "XTicks" -> If[logQ, logTicks[xr[[1]], xr[[2]]], linTicks[xr[[1]], xr[[2]]]],
   "YTicks" -> yt, "Size" -> OptionValue["Size"],
   "Padding" -> Replace[OptionValue["Padding"],
     Automatic -> If[OptionValue["ShowLabels"], {{330, 30}, {84, 16}}, {{40, 30}, {84, 16}}]]]];

(* --- assembly ---------------------------------------------------------- *)

panelImageSize[p_] := Replace[ImageSize /. Options[p, ImageSize], Automatic -> panelSize];

(* panels of equal size laid out in ncols columns, optional legend row below;
   one plot unit = one pixel so Insets keep their native size *)
panelGrid[panels_, ncols_, legendEntries_: None, gap_: 10] := Module[
  {w, h, nrows, legH, W, H, insets, legend},
  {w, h} = panelImageSize[First[panels]];
  nrows = Ceiling[Length[panels]/ncols];
  legH = If[legendEntries === None, 0, 70];
  W = ncols w + (ncols - 1) gap; H = nrows h + (nrows - 1) gap + legH;
  insets = Table[Inset[panels[[i]],
     {Mod[i - 1, ncols] (w + gap), H - Quotient[i - 1, ncols] (h + gap)}, {Left, Top}, {w, h}],
    {i, Length[panels]}];
  legend = If[legendEntries === None, {},
    Inset[legendRow[legendEntries, 26], {W/2, 12}, {Center, Bottom}]];
  Graphics[{insets, legend}, PlotRange -> {{0, W}, {0, H}}, ImageSize -> {W, H},
   AspectRatio -> H/W, PlotRangePadding -> None, ImagePadding -> None, Background -> White]];

(* --- figure family 1: canonical data, x = method ------------------------- *)

(* spec keys: file, methods (row names in order), labels, teacher (row name),
   acc, nll, ece, fidTop1, fidJsd, invTop1, invJsd (column stems), invName, out *)
canonicalFigures[spec_] := Module[{rows, get, teacher, xs, xt, series, ref, panels},
  rows = loadRows[dataFile[spec["file"]]];
  get[m_] := rowWhere[rows, #["method"] === m &];
  teacher = get[spec["teacher"]];
  xs = Range[Length[spec["methods"]]];
  xt = catTicks[spec["labels"]];
  series[col_, colour_] := <|"x" -> xs, "y" -> (ms[get[#], col][[1]] & /@ spec["methods"]),
    "s" -> (ms[get[#], col][[2]] & /@ spec["methods"]), "colour" -> colour|>;
  ref[col_, colour_, label_] := <|"y" -> ms[teacher, col][[1]], "s" -> ms[teacher, col][[2]],
    "colour" -> colour, "label" -> label|>;
  panels = <|
    "acc" -> seriesPanel[{series[spec["acc"], tolVibrant["blue"]]},
      {ref[spec["acc"], Black, "Teacher"]}, "YLabel" -> "Accuracy", "XTicks" -> xt,
      "Padding" -> {{125, 22}, {84, 16}}],
    "nll_ece" -> twinPanel[
      <|"series" -> {series[spec["nll"], tolVibrant["blue"]]},
        "refs" -> {ref[spec["nll"], tolVibrant["blue"], "Teacher"]}|>,
      <|"series" -> {series[spec["ece"], tolVibrant["orange"]]},
        "refs" -> {ref[spec["ece"], tolVibrant["orange"], "Teacher"]}|>,
      "YLabel" -> "NLL", "YRightLabel" -> "ECE", "XTicks" -> xt],
    "fid" -> twinPanel[
      <|"series" -> {series[spec["fidTop1"], tolVibrant["blue"]]}, "refs" -> {}|>,
      <|"series" -> {series[spec["fidJsd"], tolVibrant["orange"]]}, "refs" -> {}|>,
      "YLabel" -> "Top-1 agreement (teacher)", "YRightLabel" -> "1 \[Minus] JSD (teacher)",
      "XTicks" -> xt],
    "inv" -> twinPanel[
      <|"series" -> {series[spec["invTop1"], tolVibrant["blue"]]}, "refs" -> {}|>,
      <|"series" -> {series[spec["invJsd"], tolVibrant["orange"]]}, "refs" -> {}|>,
      "YLabel" -> "Top-1 agreement (" <> spec["invName"] <> ")",
      "YRightLabel" -> "1 \[Minus] JSD (" <> spec["invName"] <> ")", "XTicks" -> xt]|>;
  KeyValueMap[exportFigure[#2, spec["out"] <> "_" <> #1] &, panels];
  panels];

canonicalSpecs = {
   <|"file" -> "noshuffle_jet__canonically_ordered_jets.csv",
     "methods" -> {"Baseline", "KD", "Hint"}, "labels" -> {"Baseline", "KD", "Hint"},
     "teacher" -> "Teacher", "acc" -> "accu", "nll" -> "nll", "ece" -> "ece",
     "fidTop1" -> "fid_top1", "fidJsd" -> "fid_1mjsd", "invTop1" -> "pi_agree",
     "invJsd" -> "pi_1mjsd", "invName" -> "permutation", "out" -> "noshuffle_jet"|>,
   <|"file" -> "noshuffle_mnist__canonically_ordered_mnist.csv",
     "methods" -> {"Baseline", "KD", "Hint"}, "labels" -> {"Baseline", "KD", "Hint"},
     "teacher" -> "Teacher", "acc" -> "accu", "nll" -> "nll", "ece" -> "ece",
     "fidTop1" -> "fid_top1", "fidJsd" -> "fid_1mjsd", "invTop1" -> "ti_agree",
     "invJsd" -> "ti_1mjsd", "invName" -> "translation", "out" -> "noshuffle_mnist"|>,
   <|"file" -> "lorentz_canonical_methods.csv",
     "methods" -> {"baseline", "kd", "hint"},
     "labels" -> {"Baseline", Row[{"KD (", Style["T", Italic], " = 1)"}],
       Row[{"Hint (", Style["T", Italic], " = 1)"}]},
     "teacher" -> "teacher", "acc" -> "accuracy", "nll" -> "nll", "ece" -> "ece",
     "fidTop1" -> "teacher_top1_agreement", "fidJsd" -> "teacher_1_minus_jsd",
     "invTop1" -> "lorentz_top1_agreement", "invJsd" -> "lorentz_1_minus_jsd",
     "invName" -> "Lorentz", "out" -> "bestT_jet"|>};

(* --- figure family 2: temperature sweeps ------------------------------- *)

sweepColours = <|"KD" -> tolVibrant["blue"], "Hint" -> tolVibrant["orange"],
   "HintBeta" -> tolVibrant["teal"], "Baseline" -> tolVibrant["red"],
   "Relational" -> tolVibrant["magenta"], "Teacher" -> Black|>;
sweepLegend = {{sweepColours["KD"], "KD"}, {sweepColours["Hint"], "Hint"},
   {sweepColours["HintBeta"], Row[{"Hint, ", Style["\[Beta]", Italic], " = 0.25"}]},
   {sweepColours["Baseline"], "Baseline", True},
   {sweepColours["Relational"], "Relational distillation", True},
   {sweepColours["Teacher"], "Teacher", True}};

(* spec keys: file, curves (assoc method name -> row selector value), refs
   (assoc name -> row selector), isRef (row -> bool), tOf (row -> T),
   metrics (list of {column stem, y label, logQ}), compute (assoc unit, teacher, student),
   hasStd, out *)
sweepFigure[spec_] := Module[{rows, curveRows, refRow, val, mkSeries, mkRef, panels},
  rows = loadRows[dataFile[spec["file"]]];
  curveRows[m_] := SortBy[Select[rows, !spec["isRef"][#] && #["method"] === m &], spec["tOf"]];
  refRow[m_] := rowWhere[rows, spec["isRef"][#] && #["method"] === m &];
  val[row_, col_] := If[spec["hasStd"], ms[row, col], {row[col], None}];
  mkSeries[name_, col_] := With[{rs = curveRows[spec["curves"][name]]},
    <|"x" -> (Log2[spec["tOf"][#]] & /@ rs), "y" -> (val[#, col][[1]] & /@ rs),
      "s" -> If[spec["hasStd"], val[#, col][[2]] & /@ rs, None], "colour" -> sweepColours[name]|>];
  (* a reference is drawn as a line when it lies near the curves, else quoted in a corner *)
  mkRef[name_, col_, curveVals_] := With[{r = refRow[spec["refs"][name]]},
    If[MissingQ[r] || MissingQ[r[If[spec["hasStd"], col <> "_mean", col]]], Nothing,
     With[{v = val[r, col]},
      If[refNearQ[v[[1]], curveVals],
       <|"y" -> v[[1]], "s" -> v[[2]], "colour" -> sweepColours[name]|>,
       <|"note" -> Row[{name, ": ", NumberForm[v[[1]], {4, 3}]}], "colour" -> sweepColours[name]|>]]]];
  panels = Table[Module[{col, ylab, logQ, series, allRefs, refs, notes, extra},
     {col, ylab, logQ} = m;
     series = Table[mkSeries[name, col], {name, Keys[spec["curves"]]}];
     allRefs = Table[mkRef[name, col, Flatten[#["y"] & /@ series]], {name, Keys[spec["refs"]]}];
     refs = Select[allRefs, KeyExistsQ[#, "y"] &];
     notes = Select[allRefs, KeyExistsQ[#, "note"] &];
     (* notes go to the least occupied corner *)
     extra = With[{c = freeCorner[series, refs]},
       Table[cornerNote[notes[[i]]["note"], notes[[i]]["colour"],
         c[[1]] + {0, If[c[[2, 2]] === Top, -0.07, 0.07] (i - 1)}, c[[2]]],
        {i, Length[notes]}]];
     seriesPanel[series, refs, "XLabel" -> tempLabel, "YLabel" -> ylab,
      "XTicks" -> log2Ticks[], "XRange" -> {-0.35, 4.35}, "LogY" -> logQ,
      "Extra" -> extra, "Padding" -> {{125, 22}, {84, 16}}]],
    {m, spec["metrics"]}];
  AppendTo[panels, computePanel[spec["compute"]["teacher"], spec["compute"]["student"],
     spec["compute"]["unit"]]];
  With[{fig = panelGrid[panels, 4, sweepLegend]}, exportFigure[fig, spec["out"]]; fig]];

(* Corner with the fewest curve ends and reference lines in its band (the
   outer 25 % of the y range); ties go to top-right, then top-left. *)
freeCorner[series_, refs_] := Module[{vals, lo, hi, band, inTop, inBot, ends, corners, counts},
  vals = DeleteMissing[Flatten[{bandEdges /@ series, refEdges /@ refs}]];
  {lo, hi} = MinMax[vals]; band = 0.25 (hi - lo);
  inTop[y_] := y > hi - band; inBot[y_] := y < lo + band;
  ends[side_] := Flatten[Table[If[side === "R", Take[s["y"], -2], Take[s["y"], 2]], {s, series}]];
  corners = {{"R", inTop, {{0.97, 0.95}, {Right, Top}}}, {"L", inTop, {{0.03, 0.95}, {Left, Top}}},
    {"R", inBot, {{0.97, 0.05}, {Right, Bottom}}}, {"L", inBot, {{0.03, 0.05}, {Left, Bottom}}}};
  counts = Table[Count[ends[c[[1]]], y_ /; c[[2]][y]] + Count[#["y"] & /@ refs, y_ /; c[[2]][y]],
    {c, corners}];
  corners[[First[Ordering[counts, 1]], 3]]];

(* a reference within 1.5 curve spans of the curves is drawn as a line *)
refNearQ[v_, curveVals_] := With[{lo = Min[curveVals], hi = Max[curveVals]},
  With[{span = Max[hi - lo, 10.^-6]}, lo - 1.5 span <= v <= hi + 1.5 span]];

compute = With[{rows = loadRows[dataFile["model_compute.csv"]]},
   Association[Table[r["experiment"] -> <|"unit" -> r["unit"], "teacher" -> r["teacher"],
       "student" -> r["student"]|>, {r, rows}]]];

jetMetrics = {{"accu", "Accuracy", False}, {"pi_agree", "Top-1 agreement (permutation)", False},
   {"pi_jsd", "1 \[Minus] JSD (permutation)", False}, {"fid_top1", "Top-1 agreement (teacher)", False},
   {"fid_jsd", "1 \[Minus] JSD (teacher)", False}, {"nlll", "NLL", False}, {"ecel", "ECE", False}};
mnistMetrics = {{"accu", "Accuracy", False}, {"ti_agree", "Top-1 agreement (translation)", False},
   {"ti_1mjsd", "1 \[Minus] JSD (translation)", False}, {"fid_top1", "Top-1 agreement (teacher)", False},
   {"fid_1mjsd", "1 \[Minus] JSD (teacher)", False}, {"nlll", "NLL", False}, {"ecel", "ECE", False}};
lorentzMetrics = {{"accuracy", "Accuracy", False},
   {"lorentz_top1_agreement", "Top-1 agreement (Lorentz)", False},
   {"lorentz_1_minus_jsd", "1 \[Minus] JSD (Lorentz)", False},
   {"teacher_top1_agreement", "Top-1 agreement (teacher)", False},
   {"teacher_1_minus_jsd", "1 \[Minus] JSD (teacher)", False}, {"nll", "NLL", False}, {"ece", "ECE", False}};

gridCurves = <|"KD" -> "KD", "Hint" -> "Hint", "HintBeta" -> "Hint_beta025"|>;
gridIsRef = MissingQ[#["T"]] &;
gridT = #["T"] &;

sweepSpecs = {
   <|"file" -> "shuffled_jet_grid_a0__without_CE.csv", "curves" -> gridCurves,
     "refs" -> <|"Baseline" -> "Baseline_MLP", "Relational" -> "Relational_Distillation", "Teacher" -> "Teacher"|>,
     "isRef" -> gridIsRef, "tOf" -> gridT, "metrics" -> jetMetrics, "hasStd" -> True,
     "compute" -> compute["deepsets_jets"], "out" -> "sweep_perm_T_plot_d3w64h2_grid_a0"|>,
   <|"file" -> "shuffled_jet_grid_a1__with_CE.csv", "curves" -> gridCurves,
     "refs" -> <|"Baseline" -> "Baseline_MLP", "Relational" -> "Relational_Distillation", "Teacher" -> "Teacher"|>,
     "isRef" -> gridIsRef, "tOf" -> gridT, "metrics" -> jetMetrics, "hasStd" -> True,
     "compute" -> compute["deepsets_jets"], "out" -> "sweep_perm_T_plot_d3w64h2_grid_a1"|>,
   <|"file" -> "translated_mnist_grid_a0__without_CE.csv", "curves" -> gridCurves,
     "refs" -> <|"Baseline" -> "Baseline", "Relational" -> "Relational_Distillation", "Teacher" -> "Teacher"|>,
     "isRef" -> gridIsRef, "tOf" -> gridT, "metrics" -> mnistMetrics, "hasStd" -> True,
     "compute" -> compute["cnn_mnist"], "out" -> "sweep_mnist_T_plot_grid"|>,
   <|"file" -> "translated_mnist_grid_a2__with_CE.csv", "curves" -> gridCurves,
     "refs" -> <|"Baseline" -> "Baseline", "Relational" -> "Relational_Distillation", "Teacher" -> "Teacher"|>,
     "isRef" -> gridIsRef, "tOf" -> gridT, "metrics" -> mnistMetrics, "hasStd" -> True,
     "compute" -> compute["cnn_mnist"], "out" -> "sweep_mnist_T_plot_grid_a2"|>,
   <|"file" -> "lorentz_temperature_without_cross_entropy.csv",
     "curves" -> <|"KD" -> "kd", "Hint" -> "hint", "HintBeta" -> "hint_beta"|>,
     "refs" -> <|"Baseline" -> "baseline", "Relational" -> "relational", "Teacher" -> "teacher"|>,
     "isRef" -> (#["series_type"] === "reference" &), "tOf" -> (#["temperature"] &),
     "metrics" -> lorentzMetrics, "hasStd" -> False,
     "compute" -> compute["lorentznet_jets"], "out" -> "temperature_no_ce"|>,
   <|"file" -> "lorentz_temperature_with_cross_entropy.csv",
     "curves" -> <|"KD" -> "kd", "Hint" -> "hint", "HintBeta" -> "hint_beta"|>,
     "refs" -> <|"Baseline" -> "baseline", "Relational" -> "relational", "Teacher" -> "teacher"|>,
     "isRef" -> (#["series_type"] === "reference" &), "tOf" -> (#["temperature"] &),
     "metrics" -> lorentzMetrics, "hasStd" -> False,
     "compute" -> compute["lorentznet_jets"], "out" -> "temperature_with_ce"|>};

(* --- figure family 3: NLL and ECE across methods ------------------------- *)

methodLabels = {"Baseline", "KD", "Hint",
   Column[{"Hint,", Row[{Style["\[Beta]", Italic], " = 0.25"}]}, Alignment -> Center, Spacings -> 0.1],
   "Relational"};

nllEceFigure[rowsIn_, nllCol_, eceCol_, out_, logQ_] := Module[{xs, mk, ref, fig},
  (* rowsIn: {teacherRow, r1, r2, r3, r4, r5} in methodLabels order *)
  xs = Range[5];
  mk[col_, colour_] := <|"x" -> xs, "y" -> (ms[#, col][[1]] & /@ Rest[rowsIn]),
    "s" -> (ms[#, col][[2]] & /@ Rest[rowsIn]), "colour" -> colour|>;
  ref[col_, colour_] := <|"y" -> ms[First[rowsIn], col][[1]], "s" -> ms[First[rowsIn], col][[2]],
    "colour" -> colour, "label" -> "Teacher"|>;
  fig = twinPanel[
    <|"series" -> {mk[nllCol, tolVibrant["blue"]]}, "refs" -> {ref[nllCol, tolVibrant["blue"]]}|>,
    <|"series" -> {mk[eceCol, tolVibrant["orange"]]}, "refs" -> {ref[eceCol, tolVibrant["orange"]]}|>,
    "YLabel" -> "NLL", "YRightLabel" -> "ECE", "XTicks" -> catTicks[methodLabels],
    "LogY" -> logQ, "Size" -> {860, 520}, "Padding" -> {{118, 118}, {84, 16}}];
  exportFigure[fig, out]; fig];

nllEceFigures[] := Module[{jet, mnist, pick, figs = {}},
  Do[jet = loadRows[dataFile[f[[1]]]];
   pick = {rowWhere[jet, #["method"] === "Teacher" &],
     rowWhere[jet, #["method"] === "Baseline_MLP" &],
     rowWhere[jet, #["method"] === "KD" && #["T"] === 1 &],
     rowWhere[jet, #["method"] === "Hint" && #["T"] === 1 &],
     rowWhere[jet, #["method"] === "Hint_beta025" && #["T"] === 1 &],
     rowWhere[jet, #["method"] === "Relational_Distillation" &]};
   AppendTo[figs, nllEceFigure[pick, "nlll", "ecel", f[[2]], False]],
   {f, {{"shuffled_jet_grid_a0__without_CE.csv", "jet_nll_ece_methods_a0"},
     {"shuffled_jet_grid_a1__with_CE.csv", "jet_nll_ece_methods_a1"}}}];
  mnist = loadRows[dataFile["logit_explosion__nll_ece_and_msecos.csv"]];
  Do[pick = Table[rowWhere[mnist, #["regime"] === r[[1]] && #["method"] === m &],
     {m, {"Teacher", "MLP_baseline", "KD", "Hint", "Hint_beta025", "Relational_Distillation"}}];
   AppendTo[figs, nllEceFigure[pick, "nll", "ece", r[[2]], True]],
   {r, {{"a0_without_CE", "mnist_nll_ece_methods_a0"}, {"a2_with_CE", "mnist_nll_ece_methods_a2"}}}];
  figs];

(* --- figure family 4: relational discrepancy functions ------------------- *)

discName = <|"cosine" -> "Cosine", "mse" -> "MSE", "smooth_l1" -> "Smooth L1",
   "distance_mse" -> "Pairwise-distance MSE"|>;

relationalFigure[] := Module[{rows, keep, labels, acc, nll, a, b, fig},
  rows = loadRows[dataFile["lorentz_relational_discrepancies.csv"]];
  (* the distance-MSE run without cross-entropy collapsed (26.9 % accuracy) and is
     reported in the text instead of the figure *)
  keep = SortBy[Select[rows, !(#["discrepancy"] === "distance_mse" && #["cross_entropy"] === "without_cross_entropy") &],
    -#["accuracy"] &];
  labels = (discName[#["discrepancy"]] <> If[#["cross_entropy"] === "with_cross_entropy", " / CE", " / no CE"]) & /@ keep;
  acc = 100 #["accuracy"] & /@ keep; nll = #["nll"] & /@ keep;
  a = dotRowPanel[labels, acc, tolVibrant["blue"], "XLabel" -> "Accuracy (%)",
    "Format" -> (ToString[NumberForm[#, {5, 2}]] &)];
  b = dotRowPanel[labels, nll, tolVibrant["orange"], "XLabel" -> "NLL", "LogX" -> True,
    "ShowLabels" -> False, "Size" -> {560, 620}];
  fig = panelGrid[{a, b}, 2];
  exportFigure[fig, "relational_accuracy_nll"]; fig];

(* --- everything --------------------------------------------------------- *)

buildAllFigures[] := Module[{},
  canonicalFigures /@ canonicalSpecs;
  sweepFigure /@ sweepSpecs;
  nllEceFigures[];
  relationalFigure[];
  FileNames["*.pdf", plotsDir]];
