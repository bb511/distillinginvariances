#!/usr/bin/env wolframscript
(* Headless rebuild of every paper figure into notebooks/plots/. *)
nbDir = DirectoryName[$InputFileName];
dataDir = FileNameJoin[{nbDir, "data"}];
plotsDir = FileNameJoin[{nbDir, "plots"}];
If[!DirectoryQ[plotsDir], CreateDirectory[plotsDir]];
Get[FileNameJoin[{nbDir, "lib", "distill_style.wl"}]];
Get[FileNameJoin[{nbDir, "lib", "distill_figures.wl"}]];
files = buildAllFigures[];
Print[Length[files], " PDF figures written to ", plotsDir];
Print[Column[FileNameTake /@ files]];
