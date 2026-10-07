/- Chapter 10 exercise answers in Lean. Core Lean 4 only. -/
namespace Primer.Ex

variable {W : Type} (R : W → W → Prop)

def box (p : W → Prop) (w : W) : Prop := ∀ v, R w v → p v

/-- Axiom 4 holds on every transitive frame. -/
theorem four_of_trans (htr : ∀ a b c, R a b → R b c → R a c)
    (p : W → Prop) (w : W) (h : box R p w) : box R (box R p) w :=
  fun v hv u hu => h u (htr w v u hv hu)

theorem and_comm' (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  obtain ⟨hp, hq⟩ := h
  exact ⟨hq, hp⟩

theorem add_zero' : ∀ n : Nat, n + 0 = n := by
  intro n
  rfl

end Primer.Ex

#print axioms Primer.Ex.four_of_trans
#print axioms Primer.Ex.and_comm'
#print axioms Primer.Ex.add_zero'
