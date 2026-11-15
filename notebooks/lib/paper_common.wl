(* ::Package:: *)

(* paper_common.wl -- what the five figures of paper_figures.nb share: the cells, the
   table readers, each arm at its selected point, the fold interval, and the panels,
   markers and legends of the house style. Loaded after style.wl; reads the tables under
   resultsDir. Unless a figure says otherwise, invariance is I_top1 under the cell's own
   probe, a fold mean on the test split with the corrected 95 % interval. *)

(* The cells in the order of every categorical axis. "symmetry" is the classification of
   the observed finding (results.md): whether the labels reward the invariance on the
   data's support. Cool colours where they do, warm colours where they do not. *)
cells = <|
   "deepsets" -> <|"dataset" -> "jets", "teacher" -> "deepsets", "probe" -> "permutation",
     "control" -> "mlp", "label" -> "DeepSets", "group" -> "permutations",
     "symmetry" -> True, "colour" -> tol["teal"]|>,
   "mnist" -> <|"dataset" -> "mnist", "teacher" -> "cnn_circular", "probe" -> "translation",
     "control" -> "cnn_zeros", "label" -> "MNIST", "group" -> "translations",
     "symmetry" -> True, "colour" -> tol["cyan"]|>,
   "lloca_z" -> <|"dataset" -> "jets", "teacher" -> "lloca_z", "probe" -> "z",
     "control" -> "jetformer", "label" -> "LLoCa\:2010z", "group" -> "beam group",
     "symmetry" -> True, "colour" -> tol["blue"]|>,
   "lloca_so3" -> <|"dataset" -> "jets", "teacher" -> "lloca_so3", "probe" -> "so3",
     "control" -> "jetformer", "label" -> "LLoCa\:2010SO(3)", "group" -> "SO(3)",
     "symmetry" -> False, "colour" -> tol["magenta"]|>,
   "lorentznet" -> <|"dataset" -> "jets", "teacher" -> "lorentznet_m2", "probe" -> "lorentz_iso",
     "control" -> "jetformer", "label" -> "LorentzNet", "group" -> "isotropic boosts",
     "symmetry" -> False, "colour" -> tol["orange"]|>,
   "top" -> <|"dataset" -> "top_tagging", "teacher" -> "lorentznet_m2", "probe" -> "lorentz_iso",
     "control" -> "jetformer", "label" -> "top tagging", "group" -> "isotropic boosts",
     "symmetry" -> False, "colour" -> tol["red"]|>|>;
cellOrder = Keys[cells];
cellColour[c_] := cells[c, "colour"];

(* Tables. Top tagging was selected and analysed apart from the other five cells, so
   its points and its fold tables sit in files of their own. *)
Clear[readTable, readLines]; (* both memoise: loading the library again reads afresh *)
readTable[path_] := readTable[path] = With[{csv = Import[FileNameJoin[{resultsDir, path}], "CSV"]},
    AssociationThread[First[csv], #] & /@ Rest[csv]];
(* results.csv is 19 MB, which Import takes minutes to parse whole: only the lines
   holding the pattern are parsed. *)
readLines[path_, pattern_] := readLines[path, pattern] =
   With[{lines = ReadList[FileNameJoin[{resultsDir, path}], String]},
    With[{csv = ImportString[StringRiffle[
         Prepend[Select[Rest[lines], StringContainsQ[pattern]], First[lines]], "\n"], "CSV"]},
     AssociationThread[First[csv], #] & /@ Rest[csv]]];
ofCell[rows_, c_] := Select[rows, #dataset == cells[c, "dataset"] && #teacher == cells[c, "teacher"] &];
hyperFile[c_] := "hyperparameters/" <> If[c == "top", "hyperparameters_top3.csv", "hyperparameters.csv"];
transferFile[c_] := "tables/" <> If[c == "top", "metrics_iv", "metrics_iii"] <> "/transfer_test.csv";

(* An arm is a channel at one alpha, read at its selected temperature and beta. A table
   without a beta column holds KD and Hint rows alone, whose beta is 0. *)
num[x_] := If[NumericQ[x], N[x], x];
armKey[r_] := num /@ Lookup[r, {"method", "alpha", "temperature", "beta"}, 0.];
selectedKeys[c_, regime_] := armKey /@ Select[ofCell[readTable[hyperFile[c]], c], #regime == regime &];
atSelected[rows_, c_, regime_] := With[{keys = selectedKeys[c, regime]},
   Select[ofCell[rows, c], #regime == regime && MemberQ[keys, armKey[#]] &]];
methodOrder = {"kd", "hint", "hint_beta", "relational", "logit_matching"};
sortArms[points_] := SortBy[points, {First[FirstPosition[methodOrder, #method]], #alpha} &];

(* A point is one arm's paired difference over the folds: mean, 95 % interval, TOST p. *)
point[c_, r_, {mean_, lo_, hi_}] := <|"cell" -> c, "method" -> r["method"], "alpha" -> r["alpha"],
   "temperature" -> r["temperature"], "mean" -> r[mean], "lo" -> r[lo], "hi" -> r[hi],
   "tost" -> r["tost_p"], "margin" -> r["margin"]|>;
(* The test's outcome against the margin, as results.md defines the three labels. *)
outcome[p_] := Which[p["tost"] < 0.05, "equivalent", p["lo"] > 0 || p["hi"] < 0, "difference",
   True, "inconclusive"];

(* Gain over the baseline of the cell's own students (transfer.py). *)
gainPoint[c_][r_] := Append[point[c, r, {"gain_mean", "gain_ci_low", "gain_ci_high"}],
   {"fraction" -> r["fraction_mean"], "defined" -> r["fraction_defined"] == "True"}];
gains[c_, regime_] := sortArms[gainPoint[c] /@ atSelected[readTable[transferFile[c]], c, regime]];

(* Gain over the students of the non-invariant control at the same T and alpha
   (paired_differences.py --reference-teacher). *)
controlTable[c_] := If[c == "top",
   Join @@ (readTable["tables/metrics_iv/control_" <> # <> ".csv"] & /@ {"kd", "hint"}),
   Select[readTable["tables/orchestration_vi/gaps/out/item1_control_gains.csv"],
    #control == cells[c, "control"] &]];
controlGains[c_, regime_] := sortArms[point[c, #, {"mean_delta", "ci_low", "ci_high"}] & /@
    atSelected[Select[controlTable[c], #metric == "inv_top1" && #["eval_split"] == "test" &], c, regime]];

(* The corrected resampled t-interval of the registration (Nadeau and Bengio 2003,
   eq. 22): the variance of the fold mean is inflated by 0.25, the ratio of test to
   training rows of one of five folds. *)
foldInterval[d_] := With[{half = Quantile[StudentTDistribution[Length[d] - 1], 0.975]*
      Sqrt[(1/Length[d] + 0.25) Variance[d]]},
   <|"mean" -> Mean[d], "lo" -> Mean[d] - half, "hi" -> Mean[d] + half|>];

(* Markers in points; open markers are white inside. *)
markerShape = <|"circle" -> Disk[{0, 0}, 1],
   "square" -> Rectangle[{-0.85, -0.85}, {0.85, 0.85}],
   "diamond" -> Polygon[{{0, 1.25}, {1.05, 0}, {0, -1.25}, {-1.05, 0}}],
   "triangle" -> Polygon[{{0, 1.2}, {1.1, -0.75}, {-1.1, -0.75}}],
   "star" -> Polygon[Table[(If[EvenQ[k], 1.45, 0.6]) {Sin[k Pi/5], Cos[k Pi/5]}, {k, 0, 9}]]|>;
marker[shape_, colour_, filled_, size_ : 4.5, opacity_ : 1, edge_ : 0.5] := Graphics[{
    FaceForm[If[filled, Directive[Opacity[opacity], colour], White]],
    EdgeForm[Directive[Opacity[opacity], If[filled, Darker[colour, 0.25], colour],
      AbsoluteThickness[edge]]],
    markerShape[shape]}, ImageSize -> size, PlotRangePadding -> Scaled[0.12]];

(* Greek in the math italic of TeX Gyre Termes Math, which the text font lacks. *)
greek = Style[#, FontFamily -> "TeX Gyre Termes Math"] & /@ <|"alpha" -> "\|01D6FC", "beta" -> "\|01D6FD"|>;
alphaZero = Row[{greek["alpha"], " = 0"}];
alphaPositive = Row[{greek["alpha"], " > 0"}];

(* One grammar for every arm: the shape is the channel, filled is alpha = 0 (the teacher
   alone), open is alpha = alpha_CE > 0, and a faded point is one whose test is inconclusive. *)
methodShape = <|"kd" -> "circle", "hint" -> "square", "hint_beta" -> "diamond",
   "relational" -> "triangle", "logit_matching" -> "star"|>;
methodLabel = <|"kd" -> "KD", "hint" -> "Hint", "hint_beta" -> Row[{"Hint\:2010", greek["beta"]}],
   "relational" -> "relational", "logit_matching" -> "logit matching"|>;
markerSize[method_] := If[method == "logit_matching", 5.5, 4.5];
fade[p_] := If[outcome[p] == "inconclusive", 0.35, 1];
armMarker[p_, colour_] := marker[methodShape[p["method"]], colour, p["alpha"] == 0,
   markerSize[p["method"]], fade[p]];
whisker[x_, {lo_, hi_}, colour_, opacity_ : 1] :=
  {Opacity[opacity], colour, AbsoluteThickness[0.5], Line[{{x, lo}, {x, hi}}]};
pointLayer[p_, x_, colour_] := {whisker[x, {p["lo"], p["hi"]}, colour, fade[p]],
   Inset[armMarker[p, colour], {x, p["mean"]}]};

(* A categorical axis: cell k fills [k - 1, k], and its n arms sit dx apart about the
   centre. The grey band is the cell's equivalence margin. *)
armX[k_, i_, n_, dx_] := k - 0.5 + (i - (n + 1)/2) dx;
cellPoints[points_, k_, dx_, colour_] := MapIndexed[
   pointLayer[#1, armX[k, #2[[1]], Length[points], dx], colour] &, points];
marginBand[k_, m_] := {GrayLevel[0.9], Rectangle[{k - 0.95, -m}, {k - 0.05, m}]};
zeroGuide = {guideStyle, InfiniteLine[{{0, 0}, {1, 0}}]};
cellLabel[c_, 1] := text[cells[c, "label"], 8];
cellLabel[c_, 2] := Column[{text[cells[c, "label"], 8], text[cells[c, "group"], 7]}, Center,
   Spacings -> -0.1];
cellTicks[cs_, lines_] := MapIndexed[{#2[[1]] - 0.5, cellLabel[#1, lines], {0, 0}} &, cs];

(* Decimal ticks with the typographic minus, which the operator substitution switched
   off in style.wl leaves alone. *)
minus[s_] := StringReplace[s, "-" -> "\:2212"];
decimal[y_, digits_] := If[Chop[y] == 0, "0", minus[ToString[NumberForm[N[y], {Infinity, digits}]]]];
decTicks[{lo_, hi_}, step_, digits_, labelled_ : True] := Table[
   {y, If[labelled, text[decimal[y, digits], 8], ""], {0.025, 0}},
   {y, step Ceiling[lo/step], hi, step}];

(* The regime or the cell in a tab that shares the frame's top-left corner. *)
tab[label_] := Framed[text[label, 7], FrameMargins -> {{2, 2}, {1, 1}},
   FrameStyle -> frameStyle, Background -> White, RoundingRadius -> 0];

(* A framed panel of size {w, h} in points with fixed paddings, so that panels placed
   side by side or stacked keep their frames aligned. *)
panel[layers_, {xr_, yr_}, {xt_, yt_}, label_, {w_, h_}, pad_] := Graphics[layers,
   PlotRange -> {xr, yr}, PlotRangePadding -> None, PlotRangeClipping -> True,
   AspectRatio -> Full, Frame -> True, FrameStyle -> frameStyle, FrameTicks -> {{yt, None}, {xt, None}},
   FrameTicksStyle -> Directive[Black, 8], BaseStyle -> baseStyle,
   Epilog -> Inset[tab[label], Scaled[{0, 1}], {Left, Top}],
   ImageSize -> {w, h} + Total /@ pad, ImagePadding -> pad];

(* A figure is a canvas in points, as wide as the text block; a panel is placed by the
   bottom-left corner of its frame. *)
canvas[items_, height_] := Graphics[items, PlotRange -> {{0, fullWidth}, {0, height}},
   PlotRangePadding -> None, ImageSize -> {fullWidth, height}, ImagePadding -> 0];
place[g_, {x_, y_}, pad_] := Inset[g, {x - pad[[1, 1]], y - pad[[2, 1]]}, {Left, Bottom}];
xLabel[label_, {x_, y_}] := Text[text[label], {x, y}];
yLabel[label_, {x_, y_}] := Text[text[label], {x, y}, Center, {0, 1}];

(* Legend rows, each centred on x on its own: an entry is its glyph, 3 pt and its label,
   with 12 pt between entries. *)
glyphWidth[g_] := First[Flatten[{ImageSize /. Options[g, ImageSize]}]];
labelWidth[label_] := First[Rasterize[text[label, 7], "RasterSize", ImageResolution -> 72]];
entryWidth[{g_, label_}] := glyphWidth[g] + 3 + labelWidth[label];
legendEntry[{g_, label_}, x_, y_] := {Inset[g, {x + glyphWidth[g]/2, y + 0.7}],
   Text[text[label, 7], {x + glyphWidth[g] + 3, y}, {-1, 0}]};
legendRow[entries_, x_, y_] := With[{widths = entryWidth /@ entries},
   MapThread[legendEntry[#1, #2, y] &, {entries,
     x - (Total[widths] + 12 (Length[widths] - 1))/2 + Most[FoldList[Plus, 0, widths + 12]]}]];
legendCentred[rows_, x_, ys_] := MapThread[legendRow[#1, x, #2] &, {rows, ys}];

(* Legend glyphs shared by the figures: a filled or open pair standing for any shape, a
   faded marker, the margin band and a line. *)
pairGlyph[filled_] := Graphics[{Inset[marker["circle", Black, filled], {-2.6, 0}],
    Inset[marker["square", Black, filled], {2.6, 0}]}, PlotRange -> {{-5, 5}, {-3, 3}}, ImageSize -> {10, 6}];
(* A colour key is a swatch, so that it is not read as one of the channels' markers. *)
swatch[colours_] := Graphics[MapIndexed[{#1, Rectangle[{4 #2[[1]] - 4, -2.5}, {4 #2[[1]], 2.5}]} &, colours],
   PlotRange -> {{0, 4 Length[colours]}, {-3, 3}}, ImageSize -> {4 Length[colours], 6}];
bandGlyph = Graphics[{GrayLevel[0.9], Rectangle[{-4, -2.5}, {4, 2.5}]},
   PlotRange -> {{-4, 4}, {-3, 3}}, ImageSize -> {8, 6}];
lineGlyph[style_] := Graphics[{style, Line[{{-4, 0}, {4, 0}}]}, PlotRange -> {{-4, 4}, {-3, 3}},
   ImageSize -> {8, 6}];
fillEntries = {{pairGlyph[True], alphaZero}, {pairGlyph[False], alphaPositive}};
fadedEntry = {marker["circle", Black, True, 4.5, 0.35], "inconclusive"};
bandEntry = {bandGlyph, "equivalence margin"};
methodEntries[methods_] := {marker[methodShape[#], Black, True, markerSize[#]], methodLabel[#]} & /@ methods;
cellEntries[cs_] := {marker["circle", cellColour[#], True], cells[#, "label"]} & /@ cs;

(* Readings: three decimals, as the text quotes them. *)
r3[x_] := Round[x, 0.001];
reading[p_] := <|"cell" -> p["cell"], "arm" -> {p["method"], p["alpha"], p["temperature"]},
   "mean" -> r3[p["mean"]], "interval" -> r3[{p["lo"], p["hi"]}], "margin" -> p["margin"],
   "TOST p" -> Round[p["tost"], 0.0001], "outcome" -> outcome[p]|>;
