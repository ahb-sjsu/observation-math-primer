# bibtex runs inside -outdir, so point it back at the repository root for refs/*.bib.
# MiKTeX's bibtex ignores BIBINPUTS but accepts --include-directory.
use Cwd;
my $root = getcwd();
ensure_path('BIBINPUTS', $root);
ensure_path('TEXINPUTS', $root);
$bibtex = "bibtex --include-directory=\"$root\" %O %S";
