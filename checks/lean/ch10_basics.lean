/-
Chapter 10 Lean snippets, propositional logic, quantifiers and proof patterns.
Core Lean 4 only (no Mathlib). Checked with Lean v4.32.2.
-/

-- A proof of `p → q` is a function from proofs of `p` to proofs of `q`.
theorem modus_ponens (p q : Prop) (hp : p) (hpq : p → q) : q :=
  hpq hp

-- Contrapositive, the constructive direction.
theorem contrapositive (p q : Prop) (hpq : p → q) : ¬q → ¬p := by
  intro hnq hp
  exact hnq (hpq hp)

-- The converse direction needs proof by contradiction, a classical principle.
theorem of_contrapositive (p q : Prop) (h : ¬q → ¬p) : p → q := by
  intro hp
  apply Classical.byContradiction
  intro hnq
  exact h hnq hp

-- Taking a conjunction apart with `obtain`.
theorem and_swap (p q : Prop) (h : p ∧ q) : q ∧ p := by
  obtain ⟨hp, hq⟩ := h
  exact ⟨hq, hp⟩

-- An existential is proved by a witness and a proof about it.
theorem exists_even_above (n : Nat) : ∃ m, m > n ∧ m % 2 = 0 :=
  ⟨2 * n + 2, by omega⟩

-- A universal claim is refuted by one counterexample.
theorem not_all_square_gt : ¬ ∀ n : Nat, n * n > n := by
  intro h
  have h1 := h 1
  exact absurd h1 (by decide)

-- Linear arithmetic over Nat, decided by `omega`.
theorem odd_succ_even (n : Nat) (h : n % 2 = 1) : (n + 1) % 2 = 0 := by
  omega

-- Induction.
def sumTo : Nat → Nat
  | 0 => 0
  | n + 1 => sumTo n + (n + 1)

theorem two_mul_sumTo (n : Nat) : 2 * sumTo n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [sumTo, Nat.mul_add, ih, Nat.add_mul, Nat.mul_one, Nat.one_mul]
    omega

-- A closed computation, checked by evaluation.
example : sumTo 10 = 55 := by decide

#print axioms contrapositive
#print axioms of_contrapositive
#print axioms two_mul_sumTo

-- Contradictory hypotheses prove anything. The check passes, and says nothing.
theorem from_false (h : False) : 1 = 2 := h.elim

theorem from_inconsistent (n : Nat) (h1 : n < 3) (h2 : n > 5) : n = 100 := by
  omega

#print axioms from_false
