(* ::Package:: *)

(* distill_style.wl -- shared figure style for the distilling-invariances
   paper. Loaded with Get[]; all symbols live in Global`.

   The helpers below are copied from
   adl1t_datamaker/notebooks/nature_descriptor_plots.nb so the published
   figure style cannot drift between papers: TeX Gyre Heros, framed
   640x520 panels with ticks on all four sides, tol_vibrant palette.
   The only additions are the right-axis options of framePanel (used by
   the twin-axis panels) and a dashed legend swatch. The CMS/TeV header
   of finishPanel is deliberately absent. *)

tolVibrant = <|
   "orange" -> RGBColor["#EE7733"], "blue" -> RGBColor["#0077BB"],
   "cyan" -> RGBColor["#33BBEE"], "magenta" -> RGBColor["#EE3377"],
   "red" -> RGBColor["#CC3311"], "teal" -> RGBColor["#009988"],
   "grey" -> RGBColor["#BBBBBB"], "black" -> RGBColor["#000000"]|>;

(* --- fonts and ticks ---------------------------------------------------- *)

heros[sz_] := Directive[Black, FontFamily -> "TeX Gyre Heros", FontSize -> sz,
   PrivateFontOptions -> {"OperatorSubstitution" -> False}];
herosFont = {FontFamily -> "TeX Gyre Heros"};
tickLabel[x_] := If[Round[x] == x, ToString[Round[x]],
   ToString[NumberForm[N[x], {Infinity, 3}]]];
(* label with the number of decimals implied by the tick step, zero-padded *)
tickLabelD[x_, dec_] := If[dec <= 0, ToString[Round[x]],
   ToString[NumberForm[N[x], {Infinity, dec}, NumberPadding -> {"", "0"}]]];
blankTicks[t_] := ReplacePart[#, 2 -> ""] & /@ t;

linTicks[min_, max_, label_: True] := Module[{maj, d, dec, minor},
  maj = Select[N[FindDivisions[{min, max}, 6]], min - 10.^-9 <= # <= max + 10.^-9 &];
  If[maj === {}, maj = N[Subdivide[min, max, 5]]];
  d = If[Length[maj] > 1, maj[[2]] - maj[[1]], (max - min)/5.];
  dec = Max[0, -Floor[Log10[d] + 10.^-9]];
  minor = Select[Flatten[Table[m + d k/5, {m, Prepend[maj, First[maj] - d]},
      {k, 1, 4}]], min <= # <= max &];
  Join[Table[{m, If[label, Style[tickLabelD[m, dec], heros[22]], ""], {0.014, 0}}, {m, maj}],
   Table[{m, "", {0.007, 0}}, {m, minor}]]];

intTicks[lo_, hi_] := With[{d = Max[1, Ceiling[(hi - lo + 1)/7]]},
  Table[{k, Style[ToString[k], heros[22]], {0.014, 0}}, {k, lo, hi, d}]];

logTicks[emin_, emax_, label_: True] := Module[{es, every, subQ, maj, minor},
  es = Range[Ceiling[emin], Floor[emax]];
  subQ = Length[es] < 2;
  every = If[Length[es] > 8, 2, 1];
  maj = If[Length[es] == 0, {},
    Table[{e, If[label && Mod[e - First[es], every] == 0,
       Style[Superscript[10, e], heros[22]], ""], {0.014, 0}}, {e, es}]];
  minor = Flatten[Table[{e + Log10[k],
      If[label && subQ && MemberQ[{2, 3, 5}, k],
       Style[Row[{k, "\[Times]", Superscript[10, e]}], heros[22]], ""],
      If[subQ && MemberQ[{2, 3, 5}, k], {0.014, 0}, {0.007, 0}]},
     {e, Floor[emin], Ceiling[emax]}, {k, 2, 9}], 1];
  Join[maj, Select[minor, emin <= #[[1]] <= emax &]]];

(* --- framed panel ------------------------------------------------------- *)

panelSize = {640, 520};
panelPadding = {{112, 22}, {84, 16}};
frameLabelOf[x_, col_: Black] := If[x === None, None, Style[x, heros[30], col]];

Options[framePanel] = {"XLabel" -> None, "YLabel" -> None, "XTicks" -> Automatic,
   "YTicks" -> Automatic, "Size" -> Automatic, "Padding" -> Automatic,
   "Extra" -> {}, "Clip" -> True,
   (* additions for twin-axis panels *)
   "YRightTicks" -> None, "YRightLabel" -> None,
   "YLeftColour" -> Black, "YRightColour" -> Black};

framePanel[prims_, {x0_, x1_}, {y0_, y1_}, OptionsPattern[]] := Module[{xt, yt, ytr},
  xt = Replace[OptionValue["XTicks"], Automatic :> linTicks[x0, x1]];
  yt = Replace[OptionValue["YTicks"], Automatic :> linTicks[y0, y1]];
  ytr = Replace[OptionValue["YRightTicks"], None :> blankTicks[yt]];
  Graphics[{prims, OptionValue["Extra"]},
   PlotRange -> {{x0, x1}, {y0, y1}}, Frame -> True, Axes -> False,
   FrameStyle -> Directive[Black, AbsoluteThickness[1.5]],
   FrameTicks -> {{yt, ytr}, {xt, blankTicks[xt]}},
   FrameTicksStyle -> {{Directive[heros[22], OptionValue["YLeftColour"]],
      Directive[heros[22], OptionValue["YRightColour"]]}, {heros[22], heros[22]}},
   FrameLabel -> {{frameLabelOf[OptionValue["YLabel"], OptionValue["YLeftColour"]],
      frameLabelOf[OptionValue["YRightLabel"], OptionValue["YRightColour"]]},
     {frameLabelOf[OptionValue["XLabel"]], None}},
   LabelStyle -> heros[24], BaseStyle -> herosFont,
   ImageSize -> Replace[OptionValue["Size"], Automatic -> panelSize],
   AspectRatio -> Full, PlotRangePadding -> None,
   ImagePadding -> Replace[OptionValue["Padding"], Automatic -> panelPadding],
   PlotRangeClipping -> OptionValue["Clip"], Background -> White]];

(* --- legends ------------------------------------------------------------ *)

lineSwatch[colour_, dashed_: False] := Graphics[{colour, AbsoluteThickness[4],
    If[dashed, AbsoluteDashing[{7, 4}], Nothing], Line[{{0, 0}, {1, 0}}]},
   ImageSize -> {34, 14}, AspectRatio -> Full, PlotRange -> {{0, 1}, {-1, 1}},
   PlotRangePadding -> None, ImagePadding -> None];

(* entries: {colour, label} or {colour, label, dashedQ} *)
Options[legendBox] = {"Pos" -> {0.97, 0.96}, "Align" -> {Right, Top}, "FontSize" -> 21};
legendBox[entries_, OptionsPattern[]] := Inset[legendGrid[entries, OptionValue["FontSize"]],
   Scaled[OptionValue["Pos"]], OptionValue["Align"]];

legendGrid[entries_, fs_: 21] := Grid[
   Table[{lineSwatch[e[[1]], Length[e] > 2 && e[[3]]], Style[e[[2]], heros[fs]]}, {e, entries}],
   Alignment -> {Left, Center}, Spacings -> {0.6, 0.3}];

legendRow[entries_, fs_: 24] := Row[
   Table[Row[{lineSwatch[e[[1]], Length[e] > 2 && e[[3]]], Style[e[[2]], heros[fs]]},
     Spacer[8]], {e, entries}], Spacer[34]];

(* --- export and preview ------------------------------------------------- *)

exportFigure[fig_, name_] :=
  Export[FileNameJoin[{plotsDir, name <> "." <> ToLowerCase[#]}], fig, #] & /@ {"SVG", "PDF"};

preview[panels_, ncols_: 3] := Magnify[
   GraphicsGrid[Partition[panels, UpTo[ncols]], Background -> White, ImageSize -> 300 ncols], 0.9];
