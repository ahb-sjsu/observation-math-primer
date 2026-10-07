# bibtex runs inside -outdir, so point it back at the repository root for refs/*.bib.
# TeX Live's bibtex honours BIBINPUTS. MiKTeX's ignores it but accepts
# --include-directory, so that flag is added on Windows only.
use Cwd;
my $root = getcwd();
ensure_path('BIBINPUTS', $root);
ensure_path('TEXINPUTS', $root);
if ($^O eq 'MSWin32') {
  $bibtex = "bibtex --include-directory=\"$root\" %O %S";
}
