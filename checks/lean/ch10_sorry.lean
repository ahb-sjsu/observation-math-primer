/- Chapter 10. A theorem closed with `sorry` compiles with a warning, and
`#print axioms` exposes it. Core Lean 4 only. -/
theorem unfinished (n : Nat) : n * n ≥ n := by
  sorry

#print axioms unfinished
