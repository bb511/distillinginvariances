(* ::Package:: *)

(* style.wl -- the house style of the paper's figures, shared by every notebook under
   notebooks/. Figures are sized in points at their final width in the paper (the text
   block is 5.5 in = 396 pt), so they are included without scaling and the 7-8 pt text
   stays 7-8 pt on the page. *)

fontFamily = "TeX Gyre Termes"; (* the newtx body font of the paper *)
fullWidth = 396;

(* Paul Tol's "vibrant" scheme, used throughout the project *)
tol = <|"blue" -> RGBColor["#0077BB"], "cyan" -> RGBColor["#33BBEE"],
   "teal" -> RGBColor["#009988"], "orange" -> RGBColor["#EE7733"],
   "red" -> RGBColor["#CC3311"], "magenta" -> RGBColor["#EE3377"],
   "grey" -> RGBColor["#BBBBBB"]|>;

(* One colour per group, named as in the text. *)
groupColour = <|"so2" -> tol["blue"], "lorentz" -> tol["orange"], "so3" -> tol["magenta"],
   "all" -> tol["red"]|>;
groupLabel = <|"so2" -> "SO(2)", "lorentz" -> "SO(1,1)", "so3" -> "SO(3)",
   "all" -> "SO(1,3)"|>;

text[s_, size_ : 8] := Style[s, size, FontFamily -> fontFamily, Black,
   PrivateFontOptions -> {"OperatorSubstitution" -> False}];
frameStyle = Directive[Black, AbsoluteThickness[0.6]];
guideStyle = Directive[GrayLevel[0.6], AbsoluteThickness[0.5], AbsoluteDashing[{2, 2}]];
baseStyle = {FontFamily -> fontFamily, FontSize -> 8, Black,
   PrivateFontOptions -> {"OperatorSubstitution" -> False}};

(* Export at true size: the PDF export otherwise applies the printout magnification and
   shrinks the figure, fonts included, to 72 %. *)
exportFigure[stem_, g_] := With[{sized = Style[g, Magnification -> 1]},
   Export[stem <> ".pdf", sized]; Export[stem <> ".png", sized, ImageResolution -> 300]; stem];
