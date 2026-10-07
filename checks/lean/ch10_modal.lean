/-
Chapter 10 Lean snippets, Kripke semantics and the modal ontological argument.
Core Lean 4 only (no Mathlib). A core-only restatement of
counter-apologetics/formal/Formal/Modal.lean, with the frame conditions written
out as quantified hypotheses instead of Mathlib's `Symmetric` and `Reflexive`.
Checked with Lean v4.32.2.
-/

namespace Primer.Modal

variable {W : Type} (R : W → W → Prop)

/-- Necessity at `w`: true at every world accessible from `w`. -/
def box (p : W → Prop) (w : W) : Prop := ∀ v, R w v → p v

/-- Possibility at `w`: true at some world accessible from `w`. -/
def dia (p : W → Prop) (w : W) : Prop := ∃ v, R w v ∧ p v

/-- Axiom T holds on every reflexive frame. -/
theorem T_of_refl (hrefl : ∀ a, R a a) (p : W → Prop) (w : W)
    (h : box R p w) : p w :=
  h w (hrefl w)

/-- Premise 2 at `w`: necessarily, if G then necessarily G. -/
def Prem2 (G : W → Prop) (w : W) : Prop := box R (fun u => G u → box R G u) w

/-- The modal ontological argument is valid on every symmetric frame. -/
theorem ontological_valid_B (hsym : ∀ a b, R a b → R b a)
    (G : W → Prop) (w : W) (h1 : dia R G w) (h2 : Prem2 R G w) : G w := by
  obtain ⟨v, hwv, hGv⟩ := h1
  exact h2 v hwv hGv w (hsym w v hwv)

/-- Two worlds. `false` sees `true`, and each world sees itself. -/
def R4 (a b : Bool) : Prop := a = false ∨ b = true

/-- An S4 countermodel: reflexive and transitive, not symmetric, both premises
true at `false`, and the conclusion false there. -/
theorem S4_countermodel :
    (∀ a, R4 a a) ∧ (∀ a b c, R4 a b → R4 b c → R4 a c) ∧
    ¬ (∀ a b, R4 a b → R4 b a) ∧
    dia R4 (fun b => b = true) false ∧
    Prem2 R4 (fun b => b = true) false ∧ ¬ (false = true) := by
  refine ⟨?_, ?_, ?_, ⟨true, Or.inl rfl, rfl⟩, ?_, by decide⟩
  · intro a; cases a <;> simp [R4]
  · intro a b c hab hbc; cases a <;> cases b <;> cases c <;> simp_all [R4]
  · intro h
    have h' := h false true (Or.inl rfl)
    simp [R4] at h'
  · intro u _ hu v huv
    subst hu
    cases v <;> simp_all [R4]

end Primer.Modal

#print axioms Primer.Modal.ontological_valid_B
#print axioms Primer.Modal.S4_countermodel
