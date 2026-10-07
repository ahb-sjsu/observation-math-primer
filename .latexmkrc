# bibtex runs inside -outdir, so point it back at the repository root for refs/*.bib.
# TeX Live's bibtex honours BIBINPUTS. MiKTeX's ignores it but accepts
# --include-directory, so that flag is added whenever the bibtex on PATH is
# MiKTeX's. (Detecting the OS is unreliable: latexmk may run under an MSYS Perl.)
use Cwd;
my $root = getcwd();
ensure_path('BIBINPUTS', $root);
ensure_path('TEXINPUTS', $root);
my $bibtex_version = `bibtex --version 2>&1`;
if (defined $bibtex_version && $bibtex_version =~ /MiKTeX/) {
  $bibtex = "bibtex --include-directory=\"$root\" %O %S";
}
